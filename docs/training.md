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

The script registers every `models/laya-ft-*` folder that contains `rl_agent_config.json` through `LAYA_EXTRA_MODELS`, under its folder name, and allows 3 loaded checkpoints (`LAYA_MAX_LOADED=3`, about 6 GB). A request picks the model through the `model` field:

```python
from triage.client import DecisionClient, laya
DecisionClient(laya("laya-ft-full", name="laya-ft-full"))
```

## Results

_Filled in after the Colab runs: wall-clock time per run and the val before/after scores from each `train_report.json`._
