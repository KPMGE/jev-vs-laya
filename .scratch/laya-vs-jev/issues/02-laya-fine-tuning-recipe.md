# Laya: fine-tuning recipe

Type: research
Status: open
Blocked by:

## Question

What is the supported way to fine-tune Laya for task-specific typed decisions? Pin down: the official training script or library, the training data format (one example per question? per message with multiple questions?), full fine-tune vs LoRA/adapters, memory needs and whether ≈3k examples fit on a 16 GB Apple M5 with MPS, expected time per epoch, how the fine-tuned checkpoint is loaded back into the Jev-compatible server, and the reported gains (the 0.36 → 0.766 figure and what task it was measured on).
