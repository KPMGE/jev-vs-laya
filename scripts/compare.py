"""Score Laya base, the two Laya fine-tunes and Jev on the same messages.

    ./scripts/serve_laya.sh                      # in another terminal
    uv run scripts/compare.py                    # gold set (human-reviewed labels)
    uv run scripts/compare.py --split val        # val set (Opus labels): check the pipeline first
    uv run scripts/compare.py --replay runs/<timestamp>    # recompute metrics from saved responses
    uv run scripts/compare.py --only laya-ft-full --add-to runs/<timestamp>   # add a contender to a run

Every message is one /v1/systemone call carrying all six questions. Raw responses are saved
to runs/<timestamp>/<contender>.jsonl, metrics to results.json and a table to results.md.
Contenders that aren't available are skipped: Jev without TYPESAFE_API_KEY, a fine-tune
whose model isn't registered on the local server. See docs/evaluation.md.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from triage import metrics
from triage.client import Backend, DecisionClient, jev, laya
from triage.schema import FIELDS, QUESTIONS, state

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
LAYA_MODELS = [("laya-base", "english"), ("laya-ft-head", "laya-ft-head"), ("laya-ft-full", "laya-ft-full")]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def reference(split: str, allow_unreviewed: bool) -> dict[str, dict]:
    ids = json.loads((DATA / "splits.json").read_text())[split]
    opus = {r["id"]: r["labels"] for r in read_jsonl(DATA / "labels.opus.jsonl")}
    if split == "val":
        return {i: opus[i] for i in ids}
    gold = {r["id"]: r["labels"] for r in read_jsonl(DATA / "gold.jsonl")}  # last line per id wins
    missing = [i for i in ids if i not in gold]
    if missing and not allow_unreviewed:
        sys.exit(f"{len(missing)} of {len(ids)} gold messages aren't reviewed yet. Finish them in "
                 "`uv run scripts/review.py`, or pass --allow-unreviewed to fall back to Opus labels.")
    if missing:
        print(f"warning: {len(missing)} unreviewed gold messages scored against Opus labels", file=sys.stderr)
    return {i: gold.get(i, opus[i]) for i in ids}


def served_by(decision) -> str:
    """The checkpoint that actually answered: Laya reports it under `routing`, Jev under `model`."""
    return (decision.raw.get("routing") or {}).get("model") or decision.model


def available(backend: Backend, probe: str) -> bool:
    try:
        with DecisionClient(backend, timeout=120) as c:
            d = c.decide(probe, {"issue_type": QUESTIONS["issue_type"]})  # also warms the model up
    except httpx.HTTPError as e:
        print(f"skipping {backend.name}: {e}", file=sys.stderr)
        return False
    # laya-serve silently auto-routes unknown model names to a default checkpoint
    if backend.name.startswith("laya") and served_by(d) != backend.model:
        print(f"skipping {backend.name}: model {backend.model!r} isn't registered on the server "
              f"(it routed to {served_by(d)!r}); see docs/training.md", file=sys.stderr)
        return False
    return True


def contenders(probe: str, only: set[str] | None = None) -> list[Backend]:
    names = [n for n, _ in LAYA_MODELS] + ["jev"]
    if only and (unknown := only - set(names)):
        sys.exit(f"unknown contender(s) {sorted(unknown)}; choose from {names}")
    out = [b for name, model in LAYA_MODELS
           if (not only or name in only) and available(b := laya(model, name=name), probe)]
    if only and "jev" not in only:
        return out
    if (j := jev()) is None:
        print("skipping jev: TYPESAFE_API_KEY not set", file=sys.stderr)
    elif available(j, probe):
        out.append(j)
    return out


def run(backend: Backend, messages: dict[str, str], out: Path) -> list[dict]:
    workers = 1 if backend.name.startswith("laya") else 8  # laya-serve answers one request at a time

    def one(item):
        cid, text = item
        with DecisionClient(backend, timeout=120) as c:
            for attempt in range(3):
                try:
                    d = c.decide(text, QUESTIONS)
                    return {"id": cid, "model": served_by(d), "answers": d.answers,
                            "latency_s": d.latency_s, "input_tokens": d.input_tokens}
                except httpx.HTTPStatusError as e:
                    if e.response.status_code not in (429, 529) or attempt == 2:
                        raise
                    time.sleep(float(e.response.headers.get("retry-after", 2 ** attempt)))

    start = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, messages.items()))
    print(f"{backend.name:13} {len(rows)} messages in {time.perf_counter() - start:.0f}s", flush=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def score(name: str, rows: list[dict], refs: dict[str, dict], price: float) -> dict:
    rows = [r for r in rows if r["id"] in refs]
    labels = [refs[r["id"]] for r in rows]
    answers = [r["answers"] for r in rows]
    fields = {
        f: metrics.score_field(f, [metrics.reference_value(f, l) for l in labels], [a[f] for a in answers])
        for f in FIELDS
    }
    tokens = sum(r["input_tokens"] for r in rows) / len(rows)
    return {
        "contender": name,
        "model": rows[0]["model"],
        "n": len(rows),
        "fields": fields,
        "latency": metrics.latency([r["latency_s"] for r in rows]),
        "mean_input_tokens": tokens,
        "usd_per_1m_messages": tokens * price,  # tokens/msg × $/MTok × 1M msgs / 1M tokens
        "trust_gate": metrics.trust_gate(labels, answers),
    }


def fmt(v, kind="pct"):
    if v is None:
        return "–"
    return f"{v:.1%}" if kind == "pct" else f"{v:.2f}" if kind == "num" else f"{v:,.0f}"


def table(results: list[dict], split: str, n: int) -> str:
    names = [r["contender"] for r in results]
    head = f"| Field | Metric | {' | '.join(names)} |\n|---|---|{'---|' * len(names)}\n"
    lines = []
    for f in FIELDS:
        for metric, kind in [("accuracy", "pct"), ("macro_f1", "num"), ("mae", "num"), ("f1_yes", "num"),
                             ("auc", "num"), ("ece", "num")]:
            if metric in results[0]["fields"][f]:
                vals = [fmt(r["fields"][f][metric], kind) for r in results]
                lines.append(f"| {f} | {metric} | {' | '.join(vals)} |")
    lines.append(f"| all | latency p50 (ms) | {' | '.join(fmt(r['latency']['p50_ms'], 'int') for r in results)} |")
    lines.append(f"| all | latency p95 (ms) | {' | '.join(fmt(r['latency']['p95_ms'], 'int') for r in results)} |")
    lines.append(f"| all | input tokens / msg | {' | '.join(fmt(r['mean_input_tokens'], 'int') for r in results)} |")
    lines.append(f"| all | $ per 1M messages | {' | '.join('$' + fmt(r['usd_per_1m_messages'], 'num') for r in results)} |")

    gate_head = f"\n\n**Trust gate** (min confidence of issue_type and team; ≥{metrics.GATE_AUTO} auto, " \
                f"≥{metrics.GATE_CONFIRM} confirm, else human), coverage · joint accuracy:\n\n" \
                f"| Band | {' | '.join(names)} |\n|---|{'---|' * len(names)}\n"
    gate = "\n".join(
        f"| {b} | " + " | ".join(f"{fmt(r['trust_gate'][b]['coverage'])} · {fmt(r['trust_gate'][b]['accuracy'])}"
                                 for r in results) + " |"
        for b in ("auto", "confirm", "human")
    )
    title = f"## {split} set, {n} messages\n\n"
    return title + head + "\n".join(lines) + gate_head + gate + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["gold", "val"], default="gold")
    ap.add_argument("--replay", type=Path, help="a runs/<timestamp> folder to rescore without calling models")
    ap.add_argument("--allow-unreviewed", action="store_true", help="score unreviewed gold messages against Opus")
    ap.add_argument("--limit", type=int, help="only the first N messages (smoke test)")
    ap.add_argument("--only", help="comma-separated contenders to run, e.g. laya-ft-full")
    ap.add_argument("--add-to", type=Path, help="a runs/<timestamp> folder to add contenders to, on its messages")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    only = set(args.only.split(",")) if args.only else None

    if args.add_to:
        # Run the new contenders on exactly the run's messages, then score everything it holds.
        # Useful when the machine can't hold every Laya checkpoint in memory at once.
        out = args.add_to.resolve()
        meta = json.loads((out / "meta.json").read_text())
        split, prices = meta["split"], meta["prices"]
        refs = reference(split, args.allow_unreviewed)
        ids = [r["id"] for r in read_jsonl(next(out.glob("*.jsonl")))]
        cands = {c["id"]: c for c in read_jsonl(DATA / "candidates.jsonl")}
        messages = {i: state(cands[i]["body"], cands[i]["subject"]) for i in ids}
        backends = contenders(next(iter(messages.values())), only)
        if not backends:
            sys.exit("none of the requested contenders is reachable")
        prices |= {b.name: b.usd_per_mtok_in for b in backends}
        (out / "meta.json").write_text(json.dumps(dict(meta, prices=prices), indent=1))
        for b in backends:
            run(b, messages, out / f"{b.name}.jsonl")
        saved = {p.stem: read_jsonl(p) for p in sorted(out.glob("*.jsonl"))}
    elif args.replay:
        meta = json.loads((args.replay / "meta.json").read_text())
        split, prices = meta["split"], meta["prices"]
        refs = reference(split, args.allow_unreviewed)
        saved = {p.stem: read_jsonl(p) for p in sorted(args.replay.glob("*.jsonl"))}
        out = args.replay.resolve()
    else:
        split = args.split
        refs = reference(split, args.allow_unreviewed)
        cands = {c["id"]: c for c in read_jsonl(DATA / "candidates.jsonl")}
        ids = sorted(refs)[: args.limit] if args.limit else sorted(refs)
        refs = {i: refs[i] for i in ids}
        messages = {i: state(cands[i]["body"], cands[i]["subject"]) for i in ids}
        backends = contenders(next(iter(messages.values())), only)
        if not backends:
            sys.exit("no contender is reachable: start ./scripts/serve_laya.sh")
        out = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True)
        prices = {b.name: b.usd_per_mtok_in for b in backends}
        (out / "meta.json").write_text(json.dumps({"split": split, "n": len(ids), "prices": prices}, indent=1))
        saved = {b.name: run(b, messages, out / f"{b.name}.jsonl") for b in backends}

    order = [n for n, _ in LAYA_MODELS] + ["jev"]
    results = [score(name, saved[name], refs, prices[name]) for name in order if name in saved]
    (out / "results.json").write_text(json.dumps(results, indent=1))
    md = table(results, split, results[0]["n"])
    (out / "results.md").write_text(md)
    print("\n" + md + f"\nsaved to {out.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
