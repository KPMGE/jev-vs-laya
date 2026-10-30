# Compare script: Laya base vs Laya fine-tuned vs Jev

Type: grilling
Status: resolved
Blocked by: 06

## Question

What exactly does `scripts/compare.py` measure and print, and how does it run? Decide: per-field metrics (accuracy and macro-F1 for the choice fields; exact accuracy plus MAE for the urgency and churn scores, and whether the fractional `score` is rounded; accuracy/F1 and AUC for needs_human at a 0.5 threshold), calibration (ECE with how many bins, from Jev-formula confidence recomputed from `probabilities`), p50/p95 latency, cost per 1M messages (Jev $0.042/MTok input; Laya local $0 API cost), and whether to add the demo's trust gate (≥0.8 auto / ≥0.5 confirm / human) as a coverage-vs-accuracy row. Also: recording responses for offline replay, concurrency (laya-serve is one request at a time; Jev per-org rate limits), running on val before gold, and the output format (markdown table + `runs/<timestamp>/results.json`). Build it once decided; it runs against gold once ticket 10 is done.

## Answer

Decided with the user 2026-10-30 and built: `scripts/compare.py` + `src/triage/metrics.py`, documented in `docs/evaluation.md`.

- Score fields: accuracy after rounding the fractional score + MAE of the raw score. needs_human: accuracy/F1(yes) at 0.5 + AUC. Choice: accuracy + macro-F1. ECE (10 bins, top-option probability) everywhere.
- Trust gate row: min Jev-formula confidence of issue_type/team, bands ≥0.8 auto / ≥0.5 confirm / human, coverage · joint accuracy.
- Record + replay (`runs/<stamp>/`, `--replay`), `--split val|gold`, `--limit`, refuses incomplete gold unless `--allow-unreviewed`. Laya sequential, Jev 8 workers with 429/529 retry.
- Found and fixed: `laya-serve` silently auto-routes unknown model names to `english`, so availability is checked via the response's `routing.model`.
- Val baseline (300, Opus labels): laya-base issue_type 63.3% vs Jev 88.3%; needs_human 39.3% vs 89.3%; p50 390 vs 259 ms; Jev $46.72 per 1M messages.
