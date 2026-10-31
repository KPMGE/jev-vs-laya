# Benchmark report: Laya vs Jev on support-message triage

The one-page version: what was tested, on which data, how to rerun it, what came out and what it means. Each section links to the detailed doc.

## The question

Can Laya, an open-weight decision model you run on your own machine, replace TypeSafe's hosted Jev for triaging messages from users of a software platform, before and after fine-tuning on task data?

## The task

Every message gets six decisions, asked as one `POST /v1/systemone` call with the same questions for every model (`src/triage/schema.py`):

| Field | Type | Values |
| --- | --- | --- |
| issue_type | choice | bug, billing, cancellation, how-to, feature-request, account-access, praise |
| team | choice | engineering, billing, customer-success, support |
| urgency | score | 1–5 |
| churn_risk | score | low, medium, high |
| sentiment | choice | negative, neutral, positive |
| needs_human | noul (yes/no) | does a person need to step in beyond the normal queue |

## Contenders

| Name | What it is |
| --- | --- |
| `laya-base` | Laya `english` checkpoint (`convaiinnovations/laya`, ModernBERT-large, 421M params), untrained on this task, `laya[serve]==0.4.17` on an M5 MacBook GPU (MPS, fp32) |
| `laya-ft-head` | the same checkpoint with only the final layer fine-tuned (`--freeze-encoder`) |
| `laya-ft-full` | the same checkpoint with all parameters fine-tuned |
| `jev` | TypeSafe Jev, pinned to `jev-1.14.0`, over the network |

Details: [setup](setup.md), [training](training.md).

## Dataset

**Sources** ([data](data.md)), both non-commercial, so no data is committed:

- [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (CC BY-NC 4.0): 3,000 English tickets in formal email style, LLM-generated, sampled evenly across 7 support queues. The dataset's own `type`, `queue` and `priority` fields are passed to the labeler as hints only.
- [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (CC BY-NC-SA 4.0): 500 real customer tweets to 17 software brands (Apple, Spotify, Adobe, Microsoft, Dropbox, AWS, …), plus a boost pool of 800 keyword-matched tweets for the rare types (cancellation, praise, account access).

**Labels** ([labeling](labeling.md)): neither dataset has our six fields, so Claude labeled every message.

- **Claude Haiku 5.5** labeled all 4,300 messages. These are the training labels.
- **Claude Opus 5.5** relabeled the 500 evaluation messages (val and gold).
- A person reviews the 200 gold messages in a local web page (`scripts/review.py`).
- Total Claude cost was about $1.35, using the Batches API and prompt caching.

**Splits** (seeded, stratified on issue type; gold never touches training):

| Split | Messages | Labels | Use |
| --- | --- | --- | --- |
| train | 3,800 | Haiku | fine-tuning |
| val | 300 | Opus | fine-tune evaluation and calibration checks |
| gold | 200 (150 tickets + 50 tweets) | Opus, then human | the benchmark |

The gold mix under Opus labels:
- issue type: bug 89, how-to 42, billing 23, account-access 14, feature-request 13, praise 11, cancellation 8;
- needs_human: "yes" on 31.5%.

## How to run it

```bash
uv sync && cp .env.example .env                      # TYPESAFE_API_KEY, ANTHROPIC_API_KEY, KAGGLE_API_TOKEN

# data and labels: only needed to rebuild from scratch (about $1.35 of Claude)
uv run scripts/fetch_data.py
uv run scripts/label.py submit --model haiku && uv run scripts/label.py collect --model haiku
uv run scripts/label.py submit --model opus  && uv run scripts/label.py collect --model opus
uv run scripts/split.py
uv run scripts/review.py                             # human review of gold, http://127.0.0.1:8765
uv run scripts/build_train_data.py

# fine-tune on Colab: notebooks/finetune_colab.ipynb, then unzip into models/

# the benchmark
./scripts/serve_laya.sh                              # terminal 1
uv run scripts/compare.py                            # terminal 2; add --allow-unreviewed before the gold review
uv run scripts/compare.py --replay runs/<timestamp>  # rescore saved answers, no model calls
```

`compare.py` checks each contender before running and skips any that isn't available. It writes raw answers to `runs/<timestamp>/<contender>.jsonl` and metrics to `results.json` and `results.md`. Laya runs one request at a time; Jev runs 8 in parallel. On a 16 GB Mac, serve two Laya checkpoints at a time and add the third with `--only laya-ft-full --add-to runs/<timestamp>`. Metric definitions are in [evaluation](evaluation.md).

## Results

Gold set, 200 messages, run `runs/20261030-092746`. **These are provisional**: they're scored against the Opus labels until the human review is finished. The full tables, including calibration, are in [results](results.md).

| | laya-base | laya-ft-head | laya-ft-full | jev |
|---|---|---|---|---|
| issue_type (accuracy · macro-F1) | 66.5% · 0.54 | 68.5% · 0.55 | 81.5% · 0.71 | **88.0% · 0.84** |
| team (accuracy · macro-F1) | 62.0% · 0.51 | 65.5% · 0.57 | 83.5% · 0.80 | **86.5% · 0.85** |
| urgency (accuracy · MAE) | 29.5% · 1.06 | 32.0% · 0.90 | **72.0% · 0.36** | 66.0% · 0.39 |
| churn_risk (accuracy · MAE) | 34.5% · 0.60 | 53.5% · 0.48 | 75.5% · 0.28 | **78.5% · 0.26** |
| sentiment (accuracy) | 66.5% | 67.0% | 75.0% | **82.0%** |
| needs_human (accuracy · AUC) | 42.5% · 0.71 | 76.0% · 0.85 | 81.5% · 0.89 | **86.5% · 0.94** |
| latency p50 / p95 | 390 / 523 ms | 393 / 532 ms | 407 / 686 ms | **246 / 364 ms** |
| $ per 1M messages | $0 | $0 | $0 | $47 |
| sent through with no person checking (coverage · accuracy) | 6% · 83% | 24% · 100% | 84% · 87% | 58% · 97% |

The last row is the trust gate: messages whose issue_type and team confidence are both ≥ 0.8 are routed with no person checking them.

### Fine-tuning cost

| | laya-ft-head | laya-ft-full |
| --- | --- | --- |
| hardware | free Colab T4 | free Colab T4 |
| time | not recorded (encoder frozen, so shorter) | 1 h 22 min |
| val accuracy, all questions (untrained: 50.7%) | 66.4% | 79.7% |

## Findings

### 1. Full fine-tuning gets Laya close to Jev; head-only doesn't

- The full fine-tune gains 15 points on issue type and 21.5 on team, and beats Jev on urgency.
- Training only the final layer barely moves issue type (+2) or team (+3.5). It does learn the yes/no and score questions: needs_human goes from 42.5% to 76%, churn from 34.5% to 53.5%.
- The pretrained encoder alone doesn't separate this taxonomy well enough; the encoder has to adapt.

### 2. On clear-cut messages the full fine-tune is nearly as good as Jev; the gap is in hard cases

Opus flagged 71 of the 200 gold messages as ambiguous. Issue type accuracy:

| | clear (129) | ambiguous (71) |
|---|---|---|
| laya-base | 79.1% | 43.7% |
| laya-ft-head | 80.6% | 46.5% |
| laya-ft-full | **96.1%** | 54.9% |
| jev | **98.4%** | **69.0%** |

The full fine-tune and Jev are within about 2 points on clear messages. Most of the remaining gap comes from messages where even the labeler wasn't sure.

### 3. Real tweets are the weak spot: the training data is mostly synthetic

Issue type accuracy by source:

| | tickets (150) | tweets (50) |
|---|---|---|
| laya-base | 72.7% | 48.0% |
| laya-ft-full | 89.3% | 58.0% |
| jev | 92.0% | **76.0%** |

About 70% of the training data is formal, LLM-generated tickets. The full fine-tune gained 17 points on tickets but only 10 on tweets, where Jev is 18 points ahead. More real, informal messages in training is the most promising fix.

### 4. Rare classes lag

Per-class F1 on issue type (n in gold):

| | bug (89) | how-to (42) | billing (23) | account-access (14) | feature-request (13) | praise (11) | cancellation (8) |
|---|---|---|---|---|---|---|---|
| laya-ft-full | 0.91 | 0.83 | 0.81 | 0.56 | 0.63 | 0.62 | 0.62 |
| jev | 0.91 | 0.89 | 0.90 | 0.79 | 0.83 | 0.76 | 0.80 |

- **Classes:** the full fine-tune matches Jev on bugs and trails by 0.14–0.23 on the four rare classes. Train had only 47 praise and 59 cancellation messages, even after the boost pool.
- **Mistakes:** its most common errors are how-to read as feature-request (5), and praise or account-access read as bug (4 and 3).
- **Teams:** customer-success is its weakest team (0.67 F1 vs Jev's 0.76).

### 5. The full fine-tune is overconfident

- **The cause:** during calibration, every temperature hit the trainer's upper limit of 5. On val, its mean confidence is 93% while its accuracy is 80%.
- **The effect on routing:** at a 0.8 threshold it would send 84% of messages through with no person checking, and 22 of the 200 would be routed wrongly. Jev sends 58% through with about 4 wrong.
- **What it means:** the threshold has to be chosen per model on val before the fine-tune routes anything on its own.
- **The head-only model is the opposite.** Its confidence is honest (66% confidence at 66% accuracy on val), and the 24% it sent through were all right.

### 6. Untrained Laya isn't usable for this task

It trails Jev on every field, by 15.5 points (sentiment) to 44 (churn_risk and needs_human). On needs_human it scores 42.5%, worse than always answering "no" (68.5%), because it says "yes" far too often.

### 7. Speed and cost

- **Speed:** Jev answers faster (246 ms p50, network included) than local Laya on an M5 MacBook (about 400 ms for all six questions, one request at a time).
- **Jev's cost:** about $47 per million messages, at about 1,100 input tokens each.
- **Laya's cost:** free per call, but you pay for hardware, about 2 GB of memory per loaded checkpoint, and the one-time cost of labels and training.

### 8. The labels set a ceiling

- **Agreement:** Haiku (train labels) and Opus (eval labels) agree on 86% of issue types, 87% of teams and 76% of urgencies.
- **The ceiling:** a model trained on Haiku labels and scored against Opus can't be expected to go much beyond that, which makes the full fine-tune's 81.5% / 83.5% close to the ceiling.
- **Jev's advantage:** Jev isn't bound by this ceiling, because it wasn't trained on these labels.
- **After the review:** the human gold review will show how far Opus itself is from a person.

## Lessons from running it

- **`laya-serve` silently routes unknown model names to the default checkpoint.** A missing fine-tune would quietly score the base model twice. `compare.py` checks `routing.model` and skips mismatches.
- **On a 16 GB Mac, three fp32 checkpoints at once push the machine into swap,** and requests slow from 0.4 s to 6–10 s. Keep `LAYA_MAX_LOADED` at least as high as the number of checkpoints in use, because lower values reload a model on every switch. Serve two at a time and use `--add-to`.
- **Google Drive splits big folders into several zips,** and one can go missing. The `laya-ft-full` download arrived without `model.safetensors`. Its `checkpoint_latest/` copy only counts as the final model if it reproduces the train report's val scores; for this model it did exactly, but for the head model Drive held an older epoch. `serve_laya.sh` now skips folders without weights.
- **The full fine-tune on a T4 took 1 h 22 min,** much faster than the 3–4.5 h estimate.
- **Laya's own `confidence` uses a different formula from Jev's.** The trust gate recomputes Jev's formula from `probabilities` for every model.

## Limitations

- **Sample size:** 200 gold messages give one field's accuracy a standard error of about 3 points, so gaps under about 5 points aren't conclusive.
- **Reference labels:** until the gold review is done, the references are Opus labels, so this measures agreement with Opus, not with a person.
- **Fine-tunes:** each was a single run (seed 0) with default hyperparameters. No tuning was done.
- **Language:** English only. A Portuguese slice was ruled out of scope.

## After the gold review

```bash
uv run scripts/compare.py --replay runs/20261030-092746
```

This rescores the saved answers against the human labels without calling any model. The tables above, [results](results.md) and the "Opus vs human" line in [labeling](labeling.md) then get updated.
