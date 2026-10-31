# Laya vs Jev: support-message triage

Can a free, local, fine-tuned decision model stand in for a hosted one? This repo triages messages from users of software platforms with four contenders and scores them on the same 200 human-checked messages:

| Contender | What it is |
| --- | --- |
| `laya-base` | [Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations, Apache-2.0, ModernBERT-large, 421M params), running locally on the Mac GPU, untrained on this task |
| `laya-ft-head` | Laya with only its final layer fine-tuned on our data |
| `laya-ft-full` | Laya fully fine-tuned on our data |
| `jev` | TypeSafe's hosted [Jev](https://typesafe.ai) (`jev-1.14.0`) |

Each message gets six decisions in a single call:

| Field | Values |
| --- | --- |
| issue_type | bug, billing, cancellation, how-to, feature-request, account-access, praise |
| team | engineering, billing, customer-success, support |
| urgency | 1–5 |
| churn_risk | low, medium, high |
| sentiment | negative, neutral, positive |
| needs_human | yes/no |

The exact wording is in `src/triage/schema.py`, and the definitions are in `GLOSSARY.md`.

## Results so far

On the 200 gold messages (provisional, against Opus labels until the human review is done):

| | laya-base | laya-ft-head | laya-ft-full | jev |
|---|---|---|---|---|
| issue_type | 66.5% | 68.5% | 81.5% | **88.0%** |
| team | 62.0% | 65.5% | 83.5% | **86.5%** |
| urgency | 29.5% | 32.0% | **72.0%** | 66.0% |
| churn_risk | 34.5% | 53.5% | 75.5% | **78.5%** |
| needs_human | 42.5% | 76.0% | 81.5% | **86.5%** |
| p50 latency | 390 ms | 393 ms | 407 ms | **246 ms** |
| $ per 1M messages | $0 | $0 | $0 | $47 |

The full fine-tune closes most of the gap to Jev, but it's overconfident, so it can't be trusted to route on its own at Jev's threshold. Details are in [results](docs/results.md).

## Quick start

```bash
uv sync
cp .env.example .env              # TYPESAFE_API_KEY, ANTHROPIC_API_KEY, KAGGLE_API_TOKEN
./scripts/serve_laya.sh           # terminal 1: Laya on http://127.0.0.1:8000
uv run scripts/smoke.py           # terminal 2: one message through Laya (and Jev, if keyed)
```

## The whole pipeline

| Step | Command | Docs |
| --- | --- | --- |
| 1. Install and serve Laya | `uv sync`, `./scripts/serve_laya.sh` | [setup](docs/setup.md) |
| 2. Build the message pool (Tobi-Bueck tickets + support tweets) | `uv run scripts/fetch_data.py` | [data](docs/data.md) |
| 3. Label with Claude (Haiku for all, Opus for val + gold) | `uv run scripts/label.py submit/collect --model …` | [labeling](docs/labeling.md) |
| 4. Split into train / val / gold | `uv run scripts/split.py` | [labeling](docs/labeling.md#splits) |
| 5. Review the 200 gold messages by hand | `uv run scripts/review.py` | [labeling](docs/labeling.md#gold-review) |
| 6. Build training files | `uv run scripts/build_train_data.py` | [training](docs/training.md) |
| 7. Fine-tune both variants on Colab | `notebooks/finetune_colab.ipynb`, then unzip into `models/` | [training](docs/training.md) |
| 8. Compare all four | `uv run scripts/compare.py` | [evaluation](docs/evaluation.md) |
| 9. Read the results | `docs/results.md` | [results](docs/results.md) |

Data and models are git-ignored, because the source datasets are non-commercial (CC BY-NC / CC BY-NC-SA). Every step can be re-run from the commands above. Claude labeling cost about $1.35 in total, and Jev costs under a cent per full comparison.

## Layout

```
src/triage/
  client.py      one HTTP client for Laya and Jev (Jev's wire protocol)
  schema.py      the six questions and the message text
  labeling.py    Claude labeling prompt and output schema
  metrics.py     accuracy, F1, MAE, AUC, ECE, trust gate
scripts/         serve_laya.sh, smoke, fetch_data, label, split, review, build_train_data, compare
notebooks/       finetune_colab.ipynb
docs/            setup, data, labeling, training, evaluation, results
.scratch/        the planning map and its decision tickets
```
