# Results

Four contenders, scored on the same 200 gold messages. Each message gets one call with all six questions. The run is `runs/20261030-092746`.

> **Provisional reference labels.** The gold set hasn't been human-reviewed yet, so these numbers score against the Opus labels, as `--allow-unreviewed` does. After the review in `uv run scripts/review.py`, rescore the same saved responses with no model calls:
>
> ```bash
> uv run scripts/compare.py --replay runs/20261030-092746
> ```

## Accuracy

| Field | laya-base | laya-ft-head | laya-ft-full | jev |
|---|---|---|---|---|
| issue_type (accuracy · macro-F1) | 66.5% · 0.54 | 68.5% · 0.55 | 81.5% · 0.71 | **88.0% · 0.84** |
| team (accuracy · macro-F1) | 62.0% · 0.51 | 65.5% · 0.57 | 83.5% · 0.80 | **86.5% · 0.85** |
| urgency (accuracy · MAE) | 29.5% · 1.06 | 32.0% · 0.90 | **72.0% · 0.36** | 66.0% · 0.39 |
| churn_risk (accuracy · MAE) | 34.5% · 0.60 | 53.5% · 0.48 | 75.5% · 0.28 | **78.5% · 0.26** |
| sentiment (accuracy · macro-F1) | 66.5% · 0.61 | 67.0% · 0.61 | 75.0% · 0.70 | **82.0% · 0.76** |
| needs_human (accuracy · F1 yes · AUC) | 42.5% · 0.48 · 0.71 | 76.0% · 0.40 · 0.85 | 81.5% · 0.63 · 0.89 | **86.5% · 0.78 · 0.94** |

## Calibration (ECE, lower is better)

| Field | laya-base | laya-ft-head | laya-ft-full | jev |
|---|---|---|---|---|
| issue_type | 0.10 | 0.09 | 0.11 | **0.04** |
| team | 0.16 | 0.08 | 0.12 | **0.06** |
| urgency | 0.17 | 0.15 | 0.16 | 0.15 |
| churn_risk | 0.38 | 0.12 | 0.18 | **0.09** |
| sentiment | 0.12 | 0.17 | 0.20 | **0.04** |
| needs_human | 0.30 | 0.13 | 0.16 | **0.07** |

## Speed and cost

| | laya-base | laya-ft-head | laya-ft-full | jev |
|---|---|---|---|---|
| latency p50 / p95 (ms) | 390 / 523 | 393 / 532 | 407 / 686 | **246 / 364** |
| $ per 1M messages | $0 | $0 | $0 | $46.99 |

Laya ran on an M5 MacBook (16 GB, MPS, fp32), one request at a time. Jev's time includes the network round trip. The full model's p95 is inflated by memory pressure during its run; the three Laya checkpoints are the same architecture and should answer in the same time. Laya's $0 excludes hardware and electricity.

## Trust gate

Coverage and joint accuracy (issue_type and team both right) per confidence band. The gate uses the lower of the issue_type and team confidences: ≥ 0.8 routes automatically, ≥ 0.5 routes and has a person confirm, and anything below goes to a person.

| Band | laya-base | laya-ft-head | laya-ft-full | jev |
|---|---|---|---|---|
| auto | 6.0% · 83.3% | 24.0% · 100.0% | 83.5% · 86.8% | 58.0% · 96.6% |
| confirm | 18.5% · 97.3% | 25.0% · 78.0% | 8.5% · 35.3% | 25.5% · 68.6% |
| human | 75.5% · 42.4% | 51.0% · 20.6% | 8.0% · 12.5% | 16.5% · 30.3% |

## What it means

The deeper breakdown (clear vs ambiguous messages, tickets vs tweets, per-class F1, lessons) is in the [benchmark report](benchmark.md#findings).

- **Untrained Laya isn't usable for this task.** It's well behind on every field, and on needs_human it does worse than always answering "no".
- **Training only the final layer is cheap and fixes calibration, not routing.** needs_human jumps from 42.5% to 76% and churn_risk from 34.5% to 53.5%. issue_type and team barely move. Its confidence is honest, though: the 24% it routes automatically were all right.
- **The full fine-tune closes most of the gap to Jev.** It's within 3 points on team and churn_risk, ahead on urgency (72% vs 66%), and 6.5 points behind on issue_type. The rare classes are where it's weakest: issue_type macro-F1 is 0.71 against Jev's 0.84.
- **The full fine-tune's confidence can't be trusted as is.** Its calibration temperatures hit the trainer's upper clamp (see `docs/training.md`), so it puts 83.5% of messages in the auto band at 86.8% accuracy. Jev auto-routes 58% at 96.6%. With the same 0.8 threshold, the full fine-tune makes about five times as many unchecked routing mistakes as Jev (22 vs 4 of 200). A stricter threshold for it, chosen on val, would be needed before using it unattended.
- **Jev is still the most accurate and the best calibrated, and it's faster.** It costs about $47 per million messages. The full fine-tune is free to run, but you pay for the GPU training (1 h 22 min on a free Colab T4) and for the labeled data (about $1.35 of Claude labeling here).

## Caveats

- 200 messages is a small sample: one field's accuracy has a standard error of about 3 points, so gaps under about 5 points, such as team and churn_risk between the full fine-tune and Jev, aren't conclusive.
- The reference labels are Opus labels until the human review is done (see the note at the top).
- Training labels came from Claude Haiku and val labels from Claude Opus, so the fine-tunes learned Claude's view of the taxonomy. Jev was never trained on these labels.

## Reproduce

```bash
./scripts/serve_laya.sh
uv run scripts/compare.py --allow-unreviewed               # or without the flag once gold is reviewed
```

On a 16 GB Mac, holding all three Laya checkpoints at once can push the machine into swap. To avoid that, serve two models at a time and add the third to the same run:

```bash
LAYA_EXTRA_MODELS="{\"laya-ft-full\":\"$PWD/models/laya-ft-full\"}" ./scripts/serve_laya.sh
uv run scripts/compare.py --only laya-ft-full --add-to runs/<timestamp> --allow-unreviewed
```
