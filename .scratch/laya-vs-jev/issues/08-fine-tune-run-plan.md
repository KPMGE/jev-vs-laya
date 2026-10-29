# Fine-tune run plan

Type: grilling
Status: open
Blocked by: 05, 07

## Question

How exactly do we fine-tune Laya on our data? Decide: hard labels (`expected`) vs Claude-provided soft labels (`gold` probabilities), which base checkpoint, full fine-tune vs `--freeze-encoder`, epochs and the MPS settings (`micro_batch=1, grad_accum=64, gradient_checkpointing=True`), how long a timed pilot run is and what result switches us to the Colab T4 fallback, and the name the tuned model is served under (`LAYA_EXTRA_MODELS`). See [Laya: fine-tuning recipe](02-laya-fine-tuning-recipe.md).
