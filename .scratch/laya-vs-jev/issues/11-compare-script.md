# Compare script: Laya base vs Laya fine-tuned vs Jev

Type: grilling
Status: open
Blocked by: 06

## Question

What exactly does `scripts/compare.py` measure and print, and how does it run? Decide: per-field metrics (accuracy and macro-F1 for the choice fields; exact accuracy plus MAE for the urgency and churn scores, and whether the fractional `score` is rounded; accuracy/F1 and AUC for needs_human at a 0.5 threshold), calibration (ECE with how many bins, from Jev-formula confidence recomputed from `probabilities`), p50/p95 latency, cost per 1M messages (Jev $0.042/MTok input; Laya local $0 API cost), and whether to add the demo's trust gate (≥0.8 auto / ≥0.5 confirm / human) as a coverage-vs-accuracy row. Also: recording responses for offline replay, concurrency (laya-serve is one request at a time; Jev per-org rate limits), running on val before gold, and the output format (markdown table + `runs/<timestamp>/results.json`). Build it once decided; it runs against gold once ticket 10 is done.
