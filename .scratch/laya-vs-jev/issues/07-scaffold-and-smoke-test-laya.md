# Scaffold the repo and smoke-test local Laya

Type: task
Status: resolved
Blocked by: 01

## Question

Done when: `uv` project scaffolded (`pyproject.toml`, `src/`, `docs/setup.md`), Laya base installed and served locally per ticket 01's findings, and a smoke-test script sends one Choice decision on a sample support message and prints the answer, confidence and latency. Record the exact commands, checkpoint location, port and observed latency.

## Answer

Done. Laya base runs locally on the M5 GPU and answers through the shared client.

- Scaffold: uv project `triage` (Python 3.12), `src/triage/client.py` (one raw-HTTP client for Laya and Jev, confidence recomputed with Jev's formula), `scripts/serve_laya.sh`, `scripts/smoke.py`, `docs/setup.md`.
- Install: `uv sync` → `laya[serve]==0.4.17`, torch 2.15.0, httpx, python-dotenv. MPS available.
- Serve: `./scripts/serve_laya.sh` → `laya-serve` on `127.0.0.1:8000`, `LAYA_DEVICE=mps`, model `english` (`convaiinnovations/laya` rev `7b928d82…`, cached in `~/.cache/huggingface`).
- Smoke: `uv run scripts/smoke.py` → `billing` P=0.997; Jev skipped (no key).
- Latency: p50 87 ms, p95 98 ms (7-option choice, 123 tokens, warm).
- Chose raw httpx over `typesafe-sdk` (pre-1.0, needs zstd off and a dummy key for Laya); response `model` reads `laya-rl-agent`.
