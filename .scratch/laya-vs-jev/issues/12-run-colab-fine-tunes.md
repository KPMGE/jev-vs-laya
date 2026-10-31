# Run the two fine-tunes on Colab

Type: task
Status: resolved
Blocked by: 08

## Question

Done when: the user has run `notebooks/finetune_colab.ipynb` on a Colab T4 (HITL: their Google account and Drive), and `models/laya-ft-head` and `models/laya-ft-full` are unpacked in the repo and answer through `./scripts/serve_laya.sh`. Record: wall-clock time per run, each `train_report.json`'s val before/after scores, any warnings (prior collapse, thin calibration), and fill the Results section of `docs/training.md`.

## Answer

Both fine-tunes ran on a Colab T4 (`laya==0.4.17`, seed 0, 3 epochs) and are unpacked in `models/`. The full run took 1 h 22 min. Val accuracy over all six questions: untrained 50.7%, `laya-ft-head` 66.4%, `laya-ft-full` 79.7%. Warning: every calibration temperature of the full model hit the clamp at 5, so it's overconfident (mean confidence 93% at 80% accuracy). No prior-collapse warnings.

The Drive zip of `laya-ft-full` arrived without `model.safetensors`. Its `checkpoint_latest/` copy reproduced the train report's val scores exactly, so it was copied up. `serve_laya.sh` now skips folders without weights. Details are in [docs/training.md](../../../docs/training.md#results).
