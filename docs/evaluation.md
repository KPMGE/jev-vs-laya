# Evaluation

How the four contenders are compared, and what each number means.

```bash
./scripts/serve_laya.sh                            # terminal 1: local Laya (base + any models/laya-ft-*)
uv run scripts/compare.py --split val              # terminal 2: check the pipeline on val first
uv run scripts/compare.py                          # the real run, on the human-reviewed gold set
uv run scripts/compare.py --replay runs/<stamp>    # recompute metrics from saved responses, offline
uv run scripts/compare.py --only laya-ft-full --add-to runs/<stamp>   # add a contender to an existing run
```

`--limit N` scores only the first N messages, as a smoke test. `--only` runs a comma-separated subset of contenders. `--add-to` runs them on an existing run's messages and rescores everything in it. On a 16 GB Mac, three fp32 Laya checkpoints at once can push the machine into swap, so you can serve two models, then add the third. `--allow-unreviewed` scores gold messages you haven't reviewed yet against their Opus labels; without it the script refuses to run on an incomplete gold set.

## Contenders

| Name | What it is | Skipped when |
| --- | --- | --- |
| `laya-base` | `english` checkpoint, untrained on our task | the local server isn't running |
| `laya-ft-head` | fine-tuned, final layer only | `models/laya-ft-head` isn't registered on the server |
| `laya-ft-full` | fine-tuned, all parameters | `models/laya-ft-full` isn't registered on the server |
| `jev` | TypeSafe Jev, pinned to `jev-1.14.0` | `TYPESAFE_API_KEY` isn't set |

`laya-serve` silently routes an unknown model name to its default checkpoint. Before running, the script sends a probe and checks the response's `routing.model`, so a missing fine-tune is skipped instead of quietly scoring the base model twice. The probe also warms each model up, so the first message's latency doesn't include loading weights.

## Protocol

- One `POST /v1/systemone` call per message carries all six questions from `src/triage/schema.py`, identical for every contender.
- The input text is `schema.state(body, subject)`, the same text the labeler saw.
- Laya runs one request at a time, because `laya-serve` serializes requests anyway. Jev runs 8 in parallel and retries 429/529 errors, honoring `retry-after`.
- Reference labels: **gold** uses the human-reviewed `data/gold.jsonl`, where the last line per id wins. **val** uses Opus labels. Claude labeled the data, so it isn't a contender.
- Every raw response goes to `runs/<timestamp>/<contender>.jsonl` (git-ignored), with metrics in `results.json` and the table in `results.md`. `--replay` rescores a saved run, for example after a metric change or a gold correction, with no model calls.

## Metrics

| Field type | Fields | Metrics |
| --- | --- | --- |
| choice | issue_type, team, sentiment | accuracy, macro-F1 |
| score | urgency (0–4), churn_risk (0–2) | accuracy after rounding the fractional `score` to the nearest level; MAE of the raw score against the gold level |
| noul | needs_human | accuracy and F1 for "yes" at P ≥ 0.5; AUC |

**Macro-F1** averages F1 over every class that appears in gold or predictions, so rare classes such as praise and cancellation weigh as much as bugs. It's the number to watch for whether a model handles the whole taxonomy rather than just the common cases.

**MAE** rewards being close. Predicting urgency 3 for a 4 costs 1, predicting 1 costs 3. A score of 3.4 against a gold 4 costs 0.6 even though it rounds to the wrong level.

**AUC** measures how well P(yes) ranks needs-human messages above the rest, at any threshold. Laya's and Jev's probabilities may be calibrated differently, so AUC compares them fairly where the 0.5 threshold might not.

**ECE** (expected calibration error, 10 equal-width bins) compares each answer's top-option probability with how often such answers are right. 0 means a model that says 80% is right 80% of the time.

**Latency** is the wall-clock time per message (all six questions) as the client sees it, reported as p50 and p95. Laya runs on the M5's GPU; Jev's time includes the network round trip.

**Cost per 1M messages** is mean input tokens × price. Jev costs $0.042 per million input tokens and output is free. Local Laya's API cost is $0, and hardware and electricity aren't counted. Each backend reports its own token count, so the counts aren't directly comparable.

### Trust gate

This mirrors the talk demo. The gate's confidence is the lower of the issue_type and team confidences, using Jev's formula `(n·p_max − 1)/(n − 1)` recomputed from `probabilities` for every contender. Laya's own `confidence` field uses a different formula. The bands:

| Band | Confidence | Meaning |
| --- | --- | --- |
| auto | ≥ 0.8 | route without a person |
| confirm | ≥ 0.5 | route, then have a person confirm |
| human | < 0.5 | a person triages |

For each band the table shows **coverage** (share of messages) and **joint accuracy** (issue_type and team both right). A useful model puts many messages in `auto` with high accuracy there.

## Baseline on val

This first run, `runs/20261029-230753`, used val (300 messages, Opus labels), before the fine-tunes existed:

| Field | Metric | laya-base | jev |
|---|---|---|---|
| issue_type | accuracy | 63.3% | 88.3% |
| issue_type | macro_f1 | 0.48 | 0.78 |
| team | accuracy | 61.7% | 82.3% |
| urgency | accuracy · MAE | 35.3% · 0.97 | 69.7% · 0.36 |
| churn_risk | accuracy · MAE | 42.7% · 0.58 | 77.3% · 0.26 |
| sentiment | accuracy | 72.3% | 83.7% |
| needs_human | accuracy · AUC | 39.3% · 0.68 | 89.3% · 0.95 |
| all | latency p50 / p95 (ms) | 390 / 590 | 259 / 538 |
| all | $ per 1M messages | $0 | $46.72 |
| gate | auto coverage · accuracy | 4.0% · 91.7% | 58.3% · 94.9% |

Untrained Laya is well behind on every field. It's worst on needs_human: it says "yes" far too often, giving 39% accuracy where always answering "no" would score higher. It also sends 76% of messages to the human band. That's the gap the fine-tunes have to close. The numbers on gold, with all four contenders, are in [results](results.md).
