# Data

Where the messages come from and how the 3,500-message candidate pool is built.

```bash
uv run scripts/fetch_data.py      # downloads into data/raw/, writes data/candidates.jsonl
```

The script is seeded (`SEED = 7`) and skips downloads that are already in `data/raw/`. `data/raw/` and `data/*.jsonl` are git-ignored, because both sources forbid commercial redistribution. Counts from the latest run are in `data/candidates.stats.json`.

## Sources

| Source | What | License | How it's fetched |
| --- | --- | --- | --- |
| [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets), revision `ddf1c81a` | LLM-generated support tickets in formal email style (subject + body), with `type`, `queue`, `priority` and up to 8 tags | CC BY-NC 4.0 (HF card) | `huggingface_hub`, two CSVs: `aa_dataset-tickets-multi-lang-5-2-50-version.csv` and `dataset-tickets-multi-lang-4-20k.csv` |
| [Customer Support on Twitter (TWCS)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) | Real tweets to brand support accounts, 2017, no labels | CC BY-NC-SA 4.0 | Kaggle API with `KAGGLE_API_TOKEN` (168 MB zip, 492 MB CSV) |

Both licenses are non-commercial. This project is a local evaluation; anything derived from TWCS that gets shared must stay NC-SA.

## Tobi-Bueck: 3,000 tickets

| Step | Rows |
| --- | --- |
| Both CSVs | 48,587 |
| English | 28,261 |
| Kept queues, body ≥ 40 chars, exact-duplicate bodies removed | 21,089 |
| Sampled, balanced across queues | 3,000 |

Kept queues and how many were sampled from each: Technical Support 444, Product Support 444, Customer Service 444, IT Support 443, Billing and Payments 443, Service Outages and Maintenance 443, General Inquiry 339 (all it has). Dropped queues: Returns and Exchanges, Sales and Pre-Sales, Human Resources. They aren't platform support.

Each ticket keeps `type`, `queue`, `priority` and its tags as **hints**. The labeler sees them, but every field is labeled from scratch with our definitions (`src/triage/schema.py`), because the dataset's categories don't match ours and `priority` is noisy: one sampled ticket calls itself urgent but is labelled `low`.

Known gaps: the tickets are formal and rarely about cancellation or praise, and login/account tickets are thin (31 `Login` and 71 `Account` tags in the first three tag slots). The tweets cover the informal, cancellation and praise side.

## TWCS: 500 tweets

| Step | Rows |
| --- | --- |
| First tweets of a thread sent by a customer (`inbound`, not a reply) | 787,346 |
| Answered by one of the 17 software brands below, ≥ 30 chars after cleanup, > 95% ASCII letters | 129,372 |
| Sampled, about 29–30 per brand | 500 |

Brands (the account that replied): AppleSupport, SpotifyCares, hulu_support, AdobeCare, MicrosoftHelps, AzureSupport, DropboxSupport, GoDaddyHelp, TwitterSupport, NortonSupport, AWSSupport, PandoraSupport, GooglePlayMusic, YahooCare, asksalesforce, mediatemplehelp, OfficeSupport. Airlines, retail, telecom, banks and food were left out because their product isn't software.

Cleanup: HTML entities are unescaped, the anonymised customer handles (`@123456`) and URLs are removed, and whitespace is collapsed. The brand's own `@handle` stays, since it tells the model which platform the message is about. The hint is just `brand`.

## Candidate format

`data/candidates.jsonl`, one object per line, shuffled:

```json
{"source": "tobi", "id": "tobi-ee26879beb5c", "subject": "Enhancing Squarespace Brand Visibility",
 "body": "…", "hints": {"type": "Request", "queue": "Product Support", "priority": "low", "tags": ["…"]}}
{"source": "twcs", "id": "twcs-123", "subject": null, "body": "@mediatemplehelp My sites have been down…",
 "hints": {"brand": "mediatemplehelp"}}
```

`schema.state(body, subject)` turns a candidate into the text the models see.
