# Map: Laya vs Jev support-message classifier

Label: wayfinder:map

## Destination

A working repo (`~/Projects/mine/laya-vs-jev`) where Laya runs locally, a labeled support-message dataset exists (≈3k train / 300 val / 200 hand-verified gold test), Laya has been fine-tuned on it, everything is documented under `docs/`, and one `compare` script scores **Laya base**, **Laya fine-tuned** and **Jev** on the gold set. Jev is skipped until `TYPESAFE_API_KEY` is set.

## Notes

- **Execution is carried into this map** (override of plan-only default): tickets may build things, not just decide. The map ends when the compare script runs end to end.
- Domain: triaging inbound messages from users of a SaaS platform. Working field list, to be locked in a ticket: `issue_type` (bug / billing / cancellation / how-to / feature-request / account-access / praise), `team` (engineering / billing / customer-success / support), `urgency` 1–5, `churn_risk` low/med/high, `sentiment` neg/neu/pos, `needs_human` bool.
- Language: English for data, code and docs.
- Stack: Python + uv. A single Jev-compatible HTTP client talks to both Laya (local Jev-compatible server) and Jev (`api.typesafe.ai`).
- Ground truth: public dataset → Claude fills missing fields (`ANTHROPIC_API_KEY`) → user hand-verifies the 200-message gold set in a small local review tool. Sonnet is **not** a contender (it labeled the data).
- Training: local Apple M5 16 GB (MPS) first; documented Colab fallback.
- Metrics per field × model: accuracy, macro-F1, urgency MAE, calibration (ECE), p50/p95 latency, cost per 1M messages. Output: markdown table + JSON.
- Prior art to reuse: `~/Projects/mine/jev-talk/demo` (`lados.py` Jev question types and pricing, `triagem.py`), `~/.claude/skills/jev-explore/scripts/jev.py` (stdlib Jev client).
- Keys come later from the user: `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY` in `.env`. Every script must degrade gracefully without them.
- Always ask the user through the AskUserQuestion UI.
- Skills for grilling tickets: `grilling` + `domain-modeling`.

## Decisions so far

<!-- one line per closed ticket -->

## Not yet specified

- **Fine-tune run**: hyperparameters, epochs, single multi-head model vs one model per field, how a fine-tuned Laya is served behind the same Jev-compatible endpoint. Hangs on Laya's fine-tuning recipe and the locked schema.
- **Compare script design**: CLI shape, caching/recording of responses (like the demo's `--gravado`), concurrency and rate limits, how confidence is extracted per question type for calibration, cost model for local Laya (₀ API cost — electricity/hardware note?).
- **Docs structure**: setup, data, labeling, training, evaluation, results; how results get refreshed once the Jev key lands.
- **Colab fallback**: only if local training doesn't fit in 16 GB.
- **Trust gate / thresholds**: whether to report the demo's ≥0.8 auto / ≥0.5 confirm / human gate as an extra metric.

## Out of scope

- Sonnet (or any LLM) as a fourth contender — it produced the labels.
- Portuguese evaluation slice — possible later extra, not this effort.
