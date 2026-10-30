# Fetch and filter the raw datasets

Type: task
Status: resolved
Blocked by: 05

## Question

Done when: the English Tobi-Bueck tickets (HF) and the TWCS inbound tweets (Kaggle; HITL: the user creates a Kaggle API token and puts it in `~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN`) are downloaded into `data/raw/` (git-ignored). Tobi is filtered to the kept queues (Technical/Product/IT Support, Customer Service, Billing and Payments, Service Outages, General Inquiry) with bodies ≥ 40 chars, and TWCS to first inbound tweets addressed to software/SaaS brand handles. Record: download commands, the license of each copy used, row counts after each filter, and the TWCS brand handles chosen with their counts. The ~3.5k candidate pool sits in `data/candidates.jsonl` with `source`, `id`, `subject`, `body` and the dataset's hint fields.

## Answer

Done. `uv run scripts/fetch_data.py` builds `data/candidates.jsonl` (3,500 messages, seed 7); details and counts in `docs/data.md` and `data/candidates.stats.json`.

- Tobi-Bueck: HF revision `ddf1c81a`, CC BY-NC 4.0, two CSVs. 48,587 rows → 28,261 English → 21,089 kept (7 queues, body ≥ 40, deduped) → 3,000 sampled, ~443 per queue (General Inquiry 339, all it has). Research's "Service Outages" is really "Service Outages and Maintenance".
- TWCS: Kaggle `thoughtvector/customer-support-on-twitter`, CC BY-NC-SA 4.0, downloaded with `KAGGLE_API_TOKEN` (168 MB). 787,346 first customer tweets → 129,372 answered by 17 software brands, cleaned and English → 500 sampled, ~29–30 per brand.
- Brands: AppleSupport, SpotifyCares, hulu_support, AdobeCare, MicrosoftHelps, AzureSupport, DropboxSupport, GoDaddyHelp, TwitterSupport, NortonSupport, AWSSupport, PandoraSupport, GooglePlayMusic, YahooCare, asksalesforce, mediatemplehelp, OfficeSupport.
- `data/raw/` and `data/*.jsonl` are git-ignored (non-commercial licenses). Account-access is thin in Tobi (31 Login / 71 Account tags), which matters for stratifying the splits in ticket 06.
