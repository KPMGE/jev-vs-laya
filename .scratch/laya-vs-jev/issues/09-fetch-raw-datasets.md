# Fetch and filter the raw datasets

Type: task
Status: claimed
Blocked by: 05

## Question

Done when: the English Tobi-Bueck tickets (HF) and the TWCS inbound tweets (Kaggle; HITL: the user creates a Kaggle API token and puts it in `~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN`) are downloaded into `data/raw/` (git-ignored). Tobi is filtered to the kept queues (Technical/Product/IT Support, Customer Service, Billing and Payments, Service Outages, General Inquiry) with bodies ≥ 40 chars, and TWCS to first inbound tweets addressed to software/SaaS brand handles. Record: download commands, the license of each copy used, row counts after each filter, and the TWCS brand handles chosen with their counts. The ~3.5k candidate pool sits in `data/candidates.jsonl` with `source`, `id`, `subject`, `body` and the dataset's hint fields.
