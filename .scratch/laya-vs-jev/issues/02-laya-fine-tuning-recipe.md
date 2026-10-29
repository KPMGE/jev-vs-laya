# Laya: fine-tuning recipe

Type: research
Status: resolved
Blocked by:

## Question

What is the supported way to fine-tune Laya for task-specific typed decisions? Pin down: the official training script or library, the training data format (one example per question? per message with multiple questions?), full fine-tune vs LoRA/adapters, memory needs and whether ≈3k examples fit on a 16 GB Apple M5 with MPS, expected time per epoch, how the fine-tuned checkpoint is loaded back into the Jev-compatible server, and the reported gains (the 0.36 → 0.766 figure and what task it was measured on).

## Answer

Findings: branch `research/laya-fine-tuning`, file `research/laya-fine-tuning.md` (commit 6f159bf). Estimates are arithmetic, not measured on the M5.

- Entrypoint: `laya-train` / `laya.train.finetune` from `NandhaKishorM/laya` (may need main/0.4.18; PyPI is 0.4.17). Apple Silicon script there is a settings reference only.
- Data: JSONL, one line per message: `state`, `questions` (choice/score/noul), `expected` (hard labels) or `gold` (teacher soft probabilities). Score is 0-based → urgency 1–5 stored as 0–4. 3k messages × ~6 fields ≈ 18k items.
- No LoRA: full fine-tune or `--freeze-encoder` (head only, "gains much less").
- MPS: use the Python API with `micro_batch=1, grad_accum=64, gradient_checkpointing=True` (CLI disables checkpointing on MPS). Est. 8–10 GB fp32 — tight but should fit 16 GB.
- Time est. 1.5–4 h/epoch on M5; time a short run first. Colab T4 `--device cuda` ≈ 1–1.5 h/epoch.
- Serve tuned model: `LAYA_DEVICE=mps LAYA_EXTRA_MODELS='{"ours":"/path"}' laya-serve`, clients send `"model":"ours"` to `/v1/systemone`.
- The 0.766 is on Laya's own typed-decisions benchmark with soft labels; base 0.362 is below that benchmark's majority baseline (0.461). Vendor number, unverified.
- Open: hard vs soft labels (Claude could provide soft "gold" probabilities), base checkpoint choice, real M5 memory/speed.
