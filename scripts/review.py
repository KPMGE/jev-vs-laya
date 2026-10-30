"""Review Claude's proposed labels and turn them into gold labels.

    uv run scripts/review.py                                          # the 200 gold-set messages (Opus proposals)
    uv run scripts/review.py --items data/labels.opus.pilot.jsonl     # any labels file, e.g. the pilot

Opens http://127.0.0.1:8765. Each decision is appended to the output file
(default data/gold.jsonl; the last line per id wins), so you can stop and resume.
Ambiguous items come first. Keys: ↑/↓ pick a field, 1–7 pick a value,
Enter accept & next, ← back, → skip.
"""

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from triage.labeling import CHURN_LEVELS
from triage.schema import ISSUE_TYPES, SENTIMENTS, TEAMS, URGENCY, state

ROOT = Path(__file__).resolve().parent.parent

FIELDS = [
    {"name": "issue_type", "values": list(ISSUE_TYPES)},
    {"name": "team", "values": list(TEAMS)},
    {"name": "urgency", "values": list(range(1, len(URGENCY) + 1)), "help": URGENCY,
     "labels": ["no rush", "minor", "within a day", "blocking", "critical"]},
    {"name": "churn_risk", "values": CHURN_LEVELS},
    {"name": "sentiment", "values": list(SENTIMENTS)},
    {"name": "needs_human", "values": [False, True]},
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def load_items(items_path: Path, out_path: Path, only: set[str] | None) -> list[dict]:
    candidates = {c["id"]: c for c in read_jsonl(ROOT / "data" / "candidates.jsonl")}
    reviewed = {r["id"]: r for r in read_jsonl(out_path)}
    items = []
    for row in read_jsonl(items_path):
        if only is not None and row["id"] not in only:
            continue
        c = candidates[row["id"]]
        items.append({
            "id": row["id"],
            "source": c["source"],
            "text": state(c["body"], c["subject"]),  # exactly what the models see
            "hints": c["hints"],
            "proposed": {f["name"]: row["labels"][f["name"]] for f in FIELDS},
            "ambiguous": row["labels"].get("ambiguous", False),
            "note": row["labels"].get("note", ""),
            "review": reviewed.get(row["id"]),
        })
    items.sort(key=lambda i: not i["ambiguous"])
    return items


def make_handler(items: list[dict], out_path: Path):
    by_id = {i["id"]: i for i in items}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/state":
                self._send(json.dumps({"fields": FIELDS, "items": items}).encode(), "application/json")
            else:
                self._send(PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            item = by_id[data["id"]]
            changed = sorted(k for k, v in data["labels"].items() if item["proposed"][k] != v)
            review = {"id": data["id"], "labels": data["labels"], "status": data["status"], "changed": changed}
            item["review"] = review
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(review, ensure_ascii=False) + "\n")
            self._send(b"{}", "application/json")

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold Review</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#e2e0da;--accent:#2f5bd3;--accent-ink:#fff;
--changed:#b4530a;--changed-bg:#fdf1e6;--amb:#8a3ffc;--ok:#1f7a4d}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--mute:#9a9892;--line:#33322f;
--accent:#7c9cff;--accent-ink:#0d0d0c;--changed:#f0a35e;--changed-bg:#3a2a1b;--amb:#b78cff;--ok:#5ec48f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
header{display:flex;align-items:center;gap:16px;padding:12px 20px;border-bottom:1px solid var(--line)}
header h1{font-size:15px;margin:0;font-weight:600}.bar{flex:1;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ok)}.count{color:var(--mute);font-variant-numeric:tabular-nums}
main{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:20px;padding:20px;max-width:1300px;margin:0 auto}
@media (max-width:860px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}.tag{font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);color:var(--mute)}
.tag.amb{border-color:var(--amb);color:var(--amb)}.tag.done{border-color:var(--ok);color:var(--ok)}
.subject{font-weight:600;margin:0 0 8px}.body{white-space:pre-wrap;overflow-wrap:anywhere}
.note{margin-top:14px;padding:10px 12px;border-left:3px solid var(--amb);background:var(--bg);color:var(--mute);font-size:14px}
details{margin-top:12px;color:var(--mute);font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.field{padding:10px 12px;border-radius:8px;border:1px solid transparent;margin-bottom:6px}
.field.focus{border-color:var(--accent)}.field.changed{background:var(--changed-bg)}
.fname{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute);margin-bottom:6px;display:flex;gap:8px}
.field.changed .fname::after{content:"changed";color:var(--changed)}
.opts{display:flex;flex-wrap:wrap;gap:6px}
button.opt{font:inherit;font-size:13px;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
button.opt kbd{color:var(--mute);margin-right:5px;font-size:11px}button.opt.sel{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
button.opt.sel kbd{color:inherit;opacity:.7}button.opt.prop:not(.sel){border-style:dashed;border-color:var(--changed)}
.help{font-size:12px;color:var(--mute);margin-top:4px}
.actions{display:flex;gap:8px;margin-top:12px}.actions button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
.actions .primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.keys{margin-top:10px;font-size:12px;color:var(--mute)}
</style></head><body>
<header><h1>Gold review</h1><div class="bar"><i id="prog"></i></div><span class="count" id="count"></span></header>
<main><section class="card" id="msg"></section><section class="card"><div id="fields"></div>
<div class="actions"><button id="back">← Back</button><button id="skip">Skip →</button><button class="primary" id="accept">Accept ⏎</button></div>
<div class="keys">↑/↓ field · 1–7 value · Enter accept &amp; next · ← back · → skip. Dashed = Claude's proposal.</div></section></main>
<script>
let S, idx = 0, focus = 0, cur = {};
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const label = v => v === true ? "yes" : v === false ? "no" : v;
async function load(){ S = await (await fetch("/api/state")).json(); idx = Math.max(0, S.items.findIndex(i => !i.review)); open(); }
function open(){
  const it = S.items[idx]; cur = {...(it.review ? it.review.labels : it.proposed)}; focus = 0; render();
}
function render(){
  const it = S.items[idx], done = S.items.filter(i => i.review).length;
  $("prog").style.width = (100 * done / S.items.length) + "%";
  $("count").textContent = `${done}/${S.items.length} reviewed · item ${idx + 1}`;
  const status = it.review ? `<span class="tag done">${it.review.status}</span>` : "";
  $("msg").innerHTML = `<div class="meta"><span class="tag">${it.source}${it.hints.brand ? " · @" + esc(it.hints.brand) : ""}</span>
    ${it.ambiguous ? '<span class="tag amb">ambiguous</span>' : ""}${status}</div>
    <div class="body">${esc(it.text)}</div>
    ${it.note ? `<div class="note">Claude: ${esc(it.note)}</div>` : ""}
    <details><summary>Dataset hints</summary><pre>${esc(JSON.stringify(it.hints, null, 1))}</pre></details>`;
  $("fields").innerHTML = S.fields.map((f, fi) => {
    const changed = cur[f.name] !== it.proposed[f.name];
    const opts = f.values.map((v, vi) => `<button class="opt ${v === cur[f.name] ? "sel" : ""} ${v === it.proposed[f.name] ? "prop" : ""}"
      data-f="${fi}" data-v="${vi}"><kbd>${vi + 1}</kbd>${esc(f.labels ? f.labels[vi] : label(v))}</button>`).join("");
    const help = f.help && fi === focus ? `<div class="help">${esc(f.help[f.values.indexOf(cur[f.name])] || "")}</div>` : "";
    return `<div class="field ${fi === focus ? "focus" : ""} ${changed ? "changed" : ""}"><div class="fname">${f.name}</div><div class="opts">${opts}</div>${help}</div>`;
  }).join("");
  document.querySelectorAll("button.opt").forEach(b => b.onclick = () => { focus = +b.dataset.f; set(+b.dataset.v); });
}
function set(vi){ const f = S.fields[focus]; if (vi < f.values.length){ cur[f.name] = f.values[vi]; render(); } }
async function accept(){
  const it = S.items[idx], status = Object.keys(cur).some(k => cur[k] !== it.proposed[k]) ? "corrected" : "accepted";
  await fetch("/api/save", {method: "POST", body: JSON.stringify({id: it.id, labels: cur, status})});
  it.review = {labels: {...cur}, status}; move(1);
}
function move(d){ idx = Math.min(S.items.length - 1, Math.max(0, idx + d)); open(); }
$("accept").onclick = accept; $("skip").onclick = () => move(1); $("back").onclick = () => move(-1);
document.addEventListener("keydown", e => {
  if (e.key === "ArrowDown"){ focus = Math.min(S.fields.length - 1, focus + 1); render(); e.preventDefault(); }
  else if (e.key === "ArrowUp"){ focus = Math.max(0, focus - 1); render(); e.preventDefault(); }
  else if (/^[1-9]$/.test(e.key)) set(+e.key - 1);
  else if (e.key === "Enter"){ accept(); e.preventDefault(); }
  else if (e.key === "ArrowLeft") move(-1);
  else if (e.key === "ArrowRight") move(1);
});
load();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=ROOT / "data" / "labels.opus.jsonl")
    ap.add_argument("--split", default="gold", help="restrict to this split from data/splits.json ('all' for none)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    out = args.out or (ROOT / "data" / ("gold.pilot.jsonl" if "pilot" in args.items.name else "gold.jsonl"))
    splits = ROOT / "data" / "splits.json"
    only = None if args.split == "all" or not splits.exists() else set(json.loads(splits.read_text())[args.split])
    if "pilot" in args.items.name:
        only = None
    items = load_items(args.items, out, only)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(items, out))
    url = f"http://127.0.0.1:{args.port}"
    print(f"reviewing {len(items)} items from {args.items.name} → {out.name} at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
