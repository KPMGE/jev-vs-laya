# Jev: API surface, question types, pricing

Type: research
Status: resolved
Blocked by:

## Question

What exactly does Jev's API offer for our fields? Pin down: endpoint(s), auth, the available typed question kinds (Choice, Score, Noul, …) and which return a confidence/probability, batching, rate limits (80 req/s, 100k tok/s for 1.13?), pricing (≈$0.042/MTok input, free output?), model pinning (`JEV_MODEL`). Start from local prior art: `~/Projects/mine/jev-talk/demo/lados.py` and `~/.claude/skills/jev-explore/scripts/jev.py`, then confirm against TypeSafe's public docs.

## Answer

Findings: branch `research/jev-api`, file `research/jev-api.md` (commit 696b782).

- API: `POST /v1/systemone` + `GET /v1/models` at `https://api.typesafe.ai`, `Authorization: Bearer`. Public OpenAPI v0.2.0.
- Question kinds: `noul` (P(yes) only, no confidence), `choice` (1–255 options → choice/confidence/probabilities), `score` (1–10 ordered levels → fractional score/confidence/probabilities). `box` is images, out of scope.
- One request = one message (`state`) + many named questions (64k tok/request). No multi-message batch → parallel requests per message.
- Pricing $0.042/MTok input, output free. Rate limits are per-org (console), 429 + `retry-after`.
- Pin `jev-1.14.0` (current; `jev-latest` → 1.14.0). SDK `typesafe-sdk` 0.7.5 (pre-1.0), env `TYPESAFE_DEFAULT_MODEL`.
- Same client for Laya needs: per-backend base URL / dummy key / model / cost model, zstd off, and Laya returning `model`, `usage`, `confidence`/`probabilities` — else use a raw-HTTP client computing confidence locally (depends on ticket 01).
