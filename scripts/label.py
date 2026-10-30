"""Label candidates with Claude.

    uv run scripts/label.py pilot   --model haiku [--n 20]        # synchronous sample → data/labels.<model>.pilot.jsonl
    uv run scripts/label.py submit  --model haiku                  # every unlabeled candidate as one Message Batch
    uv run scripts/label.py submit  --model opus --ids data/eval_ids.txt
    uv run scripts/label.py collect --model haiku                  # wait, append to data/labels.<model>.jsonl

Batches cost 50% of the normal price. Opus opts into server-side refusal fallbacks on
the synchronous path; the Batches API rejects that parameter, so refused or failed
batch items are retried synchronously. See docs/labeling.md.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from triage.labeling import MODELS, cost_usd, parse, request_params

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CANDIDATES = DATA / "candidates.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def append(path: Path, rows: list[dict]):
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def record(candidate_id: str, labels: dict, message) -> dict:
    u = message.usage
    return {
        "id": candidate_id,
        "labels": labels,
        "model": message.model,
        "usage": {
            "input": u.input_tokens,
            "cache_read": u.cache_read_input_tokens or 0,
            "cache_write": u.cache_creation_input_tokens or 0,
            "output": u.output_tokens,
        },
    }


def label_sync(client: anthropic.Anthropic, candidate: dict, alias: str) -> dict | None:
    params = request_params(candidate, alias)
    if alias == "haiku":  # no server-side fallback for Haiku
        message = client.messages.create(**params)
    else:
        message = client.beta.messages.create(**params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    labels = parse(message)
    return record(candidate["id"], labels, message) if labels else None


def label_many(client, candidates: list[dict], alias: str, workers: int = 8) -> list[dict]:
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda c: label_sync(client, c, alias), candidates))
    failed = [c["id"] for c, r in zip(candidates, results) if r is None]
    if failed:
        print(f"{len(failed)} not labeled (refusal or truncation): {failed}", file=sys.stderr)
    return [r for r in results if r]


def pilot(client, alias: str, n: int):
    candidates = read_jsonl(CANDIDATES)
    tobi = [c for c in candidates if c["source"] == "tobi"][: n - n * 3 // 10]
    twcs = [c for c in candidates if c["source"] == "twcs"][: n * 3 // 10]
    t = time.perf_counter()
    rows = label_many(client, tobi + twcs, alias)
    out = DATA / f"labels.{alias}.pilot.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    spent = sum(cost_usd(r["usage"], alias) for r in rows)
    print(f"labeled {len(rows)} in {time.perf_counter() - t:.0f}s, ${spent:.4f} at full price → {out.name}")
    print(f"projected per 1,000 via Batches (50% off): ${spent / len(rows) * 1000 / 2:.2f}")


def submit(client, alias: str, ids_path: Path | None):
    done = {r["id"] for r in read_jsonl(DATA / f"labels.{alias}.jsonl")}
    wanted = set(ids_path.read_text().split()) if ids_path else None
    todo = [c for c in read_jsonl(CANDIDATES) if c["id"] not in done and (wanted is None or c["id"] in wanted)]
    batch = client.messages.batches.create(requests=[
        Request(custom_id=c["id"], params=MessageCreateParamsNonStreaming(**request_params(c, alias))) for c in todo
    ])
    (DATA / f"labels.{alias}.batch_id").write_text(batch.id)
    print(f"submitted {len(todo)} requests as batch {batch.id} ({MODELS[alias][0]})")


def collect(client, alias: str):
    batch_id = (DATA / f"labels.{alias}.batch_id").read_text().strip()
    while (batch := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
        c = batch.request_counts
        print(f"{batch.processing_status}: {c.processing} processing, {c.succeeded} succeeded", flush=True)
        time.sleep(30)

    by_id = {c["id"]: c for c in read_jsonl(CANDIDATES)}
    rows, retry = [], []
    for result in client.messages.batches.results(batch_id):
        labels = parse(result.result.message) if result.result.type == "succeeded" else None
        if labels:
            rows.append(record(result.custom_id, labels, result.result.message))
        else:
            retry.append(by_id[result.custom_id])
    spent = sum(cost_usd(r["usage"], alias) for r in rows) / 2
    if retry:
        print(f"retrying {len(retry)} synchronously")
        extra = label_many(client, retry, alias)
        spent += sum(cost_usd(r["usage"], alias) for r in extra)
        rows += extra
    append(DATA / f"labels.{alias}.jsonl", rows)
    print(f"appended {len(rows)} labels → labels.{alias}.jsonl; cost ≈ ${spent:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pilot", "submit", "collect"])
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ids", type=Path, help="file of candidate ids (whitespace-separated) to restrict submit to")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()
    if args.command == "pilot":
        pilot(client, args.model, args.n)
    elif args.command == "submit":
        submit(client, args.model, args.ids)
    else:
        collect(client, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
