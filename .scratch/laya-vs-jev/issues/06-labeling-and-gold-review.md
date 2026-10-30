# Labeling protocol and gold review tool

Type: prototype
Status: resolved
Blocked by: 05

## Question

How are labels produced and verified? Decide the Claude labeling prompt and structured-output schema, how the 3k/300/200 split is sampled (stratified on which fields), and how the user reviews the 200-message gold set. Prototype the review tool (terminal or local web page: message + proposed labels, accept/fix with keys) so the user can react to it.

## Answer

Decided with the user 2026-10-29/30; full write-up in `docs/labeling.md`.

- Budget under $5 → Haiku 5.5 labels everything (4,300 msgs, $0.21 via Batches); Opus 5.5 (effort low) relabels gold + val (500, $0.83). Total with pilots ≈ $1.35. Train = Haiku labels, val = Opus, gold = Opus + human review.
- Prompt rendered from `schema.py` (`src/triage/labeling.py`), dataset labels as hints only, JSON-schema output, `ambiguous` flag (<70% on issue_type/team/needs_human) + one-line `note`. Hard labels only, no soft labels (keeps Laya-vs-Jev clean).
- Pilots changed: tighter `ambiguous` (17→7 of 20); `needs_human` redefined as "a person must step in beyond the normal queue" (13→5 of 20); `<br>` and inline sign-offs stripped in `schema.state()`.
- Rare classes: base pool had 17 cancellation / 17 praise → added an 800-tweet keyword "boost" pool (train/val only; gold stays natural).
- Splits (`scripts/split.py`, seed 7, stratified on Haiku issue_type): gold 200 from base pool, ≥15 per type, 50 tweets; val 300 proportional; train 3,800. Praise (47) and cancellation (59) remain thin in train.
- Haiku-vs-Opus agreement on 500: issue_type 86%, team 87%, urgency 76%, churn 78%, sentiment 84%, needs_human 89%.
- Review tool: `scripts/review.py`, local web page, prototype accepted by the user as is. Opus flagged 71/200 gold as ambiguous (shown first).
