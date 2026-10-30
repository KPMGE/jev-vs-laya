# Pick a public support-message dataset

Type: research
Status: resolved
Blocked by:

## Question

Which public dataset of real or realistic English customer/user support messages is the best base for this classifier? For each candidate (e.g. Hugging Face customer-support-tickets with queue/priority/type, Bitext customer-support, Twitter customer-support corpus, others), report: license and allowed use, size, the fields it already has and how they map onto our working fields (`issue_type`, `team`, `urgency`, `churn_risk`, `sentiment`, `needs_human`), message realism and length, language mix, and download method. Recommend one (or a blend) that yields ≈3.5k usable English messages.

## Answer

Findings: branch `research/support-dataset`, file `research/support-dataset.md` (commit bdf830d): 8-dataset comparison table, 5 sample rows.

- Recommended blend: ~3,000 English rows of **Tobi-Bueck/customer-support-tickets** (HF; 61,765 rows, 28,261 English; LLM-generated, formal email style) + ~500 real inbound **TWCS** tweets to software/SaaS brands (Kaggle, 2.8M tweets, unlabeled) for informal, cancellation and praise messages. Gold set drawn from both sources.
- Tobi field mapping: `priority` (5 levels) → urgency 1–5 (noisy; Claude should re-check), `queue` → team, `type` + tags → issue_type (bug / billing / feature-request / account-access / how-to). Missing: `churn_risk`, `sentiment`, `needs_human`, and few cancellation/praise rows → Claude fills.
- Licenses: Tobi HF card CC BY-NC 4.0 vs same author's Kaggle copy CC BY 4.0 (conflict; use Kaggle copy if commercial use matters). TWCS CC BY-NC-SA 4.0 → any shared derived dataset is NC-SA. Fine for a local, non-commercial evaluation.
- Rejected: Bitext (e-commerce one-liners, no bug/feature), tasksource/help-desk-tickets (only 357 texts), simulated/templated sets (s2pidape, Console-AI, suraj520), churn datasets without message text.
- Open: which TWCS brand handles are software/SaaS and their counts — decide when building the dataset.
