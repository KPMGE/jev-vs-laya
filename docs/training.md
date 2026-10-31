# Training

How Laya is fine-tuned on the labeled support messages, and how the results come back into the local server.

## Two fine-tunes

Both start from the `english` checkpoint (`convaiinnovations/laya`, ModernBERT-large, 421M parameters), the same model the untrained contender uses. The comparison then isolates what training added.

| Model | What trains | Why |
| --- | --- | --- |
| `laya-ft-head` | final layer only (`--freeze-encoder`) | cheap; shows how much the head alone can learn |
| `laya-ft-full` | all parameters | the recipe behind upstream's published gains |

Both use hard labels only: one `expected` answer per question, with no probabilities from Claude or Jev. Using Jev's probabilities would teach Laya to imitate the model it's compared against.

## Training data

```bash
uv run scripts/build_train_data.py      # data/train.jsonl, data/val.jsonl
```

There's one row per message, in the format `laya-train` reads:

```json
{"id": "tobi-…", "state": "Subject: …\n\n…", "questions": {…the six questions from src/triage/schema.py…},
 "expected": {"issue_type": "bug", "team": "engineering", "urgency": 2, "churn_risk": 0,
              "sentiment": "negative", "needs_human": false}}
```

- `state` is `schema.state(body, subject)`, the same text the models see at evaluation.
- Score answers count from 0. Urgency 1–5 is stored as 0–4, and churn low/medium/high as 0–2.
- **train** has 3,800 messages with Haiku labels; **val** has 300 with Opus labels. The gold set never touches training. See `docs/labeling.md`.

A dry run validates the file without loading weights:

```bash
uv run laya-train --data data/train.jsonl --eval data/val.jsonl --base english --epochs 3 --dry-run
```

It reports 22,800 valid items (3,800 × 6) with none skipped. 400 of those are held out for temperature calibration, and 3 epochs at an effective batch of 64 is 1,050 optimizer updates.

## Running it on Colab

Training runs on a free Colab T4 rather than the M5. An estimated 1.5–4 hours per epoch for the full model, with only 16 GB of unified memory, made the Mac the slower and riskier option.

1. Open `notebooks/finetune_colab.ipynb` in Colab (File → Upload notebook) and set Runtime → Change runtime type → **T4 GPU**.
2. Upload `data/train.jsonl` and `data/val.jsonl` to `MyDrive/laya-vs-jev/`, or let cell 3 prompt for them.
3. Run all. The notebook:
   - installs `laya==0.4.17`;
   - dry-runs the data;
   - trains `laya-ft-head`, then `laya-ft-full`, each for 3 epochs with seed 0 and the upstream defaults (`micro_batch 8 × grad_accum 8`, AMP on CUDA);
   - writes both models to Drive;
   - prints each `train_report.json`;
   - downloads both as zips.
4. Unpack the zips into the repo:

   ```bash
   unzip laya-ft-head.zip -d models/ && unzip laya-ft-full.zip -d models/
   ```

`models/` is git-ignored. Each model folder holds the checkpoint, `questions.json` and `train_report.json`, which has the before/after scores on val.

The training data is derived from non-commercially licensed datasets, so the Drive copy is for your own use only.

Free Colab disconnects idle tabs. Because everything is written to Drive, a finished run survives a disconnect; a run that was cut off has to be restarted.

## Serving the fine-tuned models

```bash
./scripts/serve_laya.sh
```

The script registers every `models/laya-ft-*` folder that contains `rl_agent_config.json` and `model.safetensors` through `LAYA_EXTRA_MODELS`, under its folder name, and allows 3 loaded checkpoints (`LAYA_MAX_LOADED=3`, about 6 GB). A request picks the model through the `model` field:

```python
from triage.client import DecisionClient, laya
DecisionClient(laya("laya-ft-full", name="laya-ft-full"))
```

## Results

Both runs used a Colab T4 with `laya==0.4.17`, seed 0 and 3 epochs. The full fine-tune took **1 h 22 min** of wall-clock time; the head-only run is much shorter, because the encoder stays frozen. Scores are from each `train_report.json`: 1,800 val items (300 messages × 6 questions), Opus labels, never seen in training.

| | untrained | `laya-ft-head` | `laya-ft-full` |
| --- | --- | --- | --- |
| accuracy, all questions | 50.7% | 66.4% | 79.7% |
| choice (issue_type, team, sentiment) | 65.9% | 68.7% | 83.3% |
| score (urgency, churn_risk) | 33.5% | 55.7% | 70.3% |
| noul (needs_human) | 39.3% | 81.0% | 87.3% |
| mean confidence | 64.7% | 66.3% | 93.2% |
| ECE | 0.143 | 0.060 | 0.136 |
| fitted temperature (choice / score / noul) | – | 1.12 / 1.09 / 0.89 | 5.0 / 5.0 / 5.0 |

- **The head alone learns needs_human and the score questions**, but barely moves issue_type and team. Its confidence stays honest: 66% confidence, 66% accuracy.
- **The full fine-tune is far more accurate but overconfident.** Its calibration temperatures all hit the trainer's upper clamp of 5 (`the fitted temperature 5 is on the [0.5, 5] clamp`), so even after calibration its confidence runs about 13 points above its accuracy. A trust gate tuned on Jev or on the head model would let too much of its output through unchecked.
- The training log's `lr_scheduler.step() before optimizer.step()` warning is harmless: PyTorch skips the first value of the learning-rate schedule. Negative per-step losses are expected, because the `rlcd` loss includes a reward term.

### Missing weights after a Drive download

Drive zips a large folder in parts, and the `laya-ft-full` download arrived without its top-level `model.safetensors`. `checkpoint_latest/` holds the weights written at the end of the last epoch, and nothing changes the weights after that, so it should equal the final model. Drive can keep an older epoch's upload, though: for `laya-ft-head`, the two copies differed. Before using a `checkpoint_latest/` copy, check that it reproduces the `after` scores in `train_report.json` on val. For `laya-ft-full` it matched exactly (choice 83.33%, needs_human 87.33%), so it was copied up. `serve_laya.sh` skips any folder without `model.safetensors`.
