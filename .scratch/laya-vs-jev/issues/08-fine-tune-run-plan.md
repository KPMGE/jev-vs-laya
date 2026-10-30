# Fine-tune run plan

Type: grilling
Status: resolved
Blocked by: 05, 06, 07

## Question

How exactly do we fine-tune Laya on our data (train = 3,800 Haiku-labeled messages, val = 300 Opus-labeled)? Hard labels only (decided in ticket 06). Decide: which base checkpoint, full fine-tune vs `--freeze-encoder`, epochs and the MPS settings (`micro_batch=1, grad_accum=64, gradient_checkpointing=True`), how long a timed pilot run is and what result switches us to the Colab T4 fallback, and the name the tuned model is served under (`LAYA_EXTRA_MODELS`). See [Laya: fine-tuning recipe](02-laya-fine-tuning-recipe.md).

## Answer

Decided with the user 2026-10-30; details in `docs/training.md`.

- Base: `english` (`convaiinnovations/laya`), same as the untrained contender.
- **Both** variants, benchmarked against each other: `laya-ft-head` (`--freeze-encoder`) and `laya-ft-full` (all parameters). The compare script gets four contenders. (User picked the name `laya-ft`; split into `-head`/`-full` because there are two.)
- Hard labels only (`expected`); urgency 0–4, churn 0–2. `scripts/build_train_data.py` → `data/train.jsonl` (3,800, Haiku) and `data/val.jsonl` (300, Opus). Dry run: 22,800 valid items, 0 skipped, 1,050 updates over 3 epochs.
- Runs on **Colab T4** straight away (user's call, no Mac pilot): `notebooks/finetune_colab.ipynb`, `laya==0.4.17`, 3 epochs, seed 0, upstream defaults (micro_batch 8 × grad_accum 8, AMP), head first then full, outputs to Drive.
- Served locally via `scripts/serve_laya.sh`, which auto-registers `models/laya-ft-*` in `LAYA_EXTRA_MODELS` with `LAYA_MAX_LOADED=3`.
