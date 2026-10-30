# Lock the label schema and question mapping

Type: grilling
Status: resolved
Blocked by: 01, 03, 04

## Question

What is the final set of fields, their allowed values and definitions, and which typed question (Choice / Score / Noul / …) asks each one of both Laya and Jev? Includes the exact question wording, how `urgency` 1–5 and `churn_risk` are expressed, and which fields come from the dataset vs. are filled by Claude.

## Answer

Locked with the user 2026-10-29. Exact wording lives in `src/triage/schema.py` (user to review); terms in `GLOSSARY.md`.

| Field | Question | Values |
| --- | --- | --- |
| `issue_type` | choice | bug, billing, cancellation, how-to, feature-request, account-access, praise |
| `team` | choice | engineering, billing, customer-success, support |
| `urgency` | score, 5 levels | 1 no time pressure … 5 business-critical now (0–4 on the wire) |
| `churn_risk` | score, 3 levels | low, medium, high |
| `sentiment` | choice | negative, neutral, positive |
| `needs_human` | noul | P(true) |

- All six asked in one `/v1/systemone` call per message.
- Mixed messages: the main ask wins. `cancellation` only on an explicit cancel/downgrade request; threats to leave go to `churn_risk=high`.
- `team` is scored independently: default mapping from issue_type, overridden by cause (outage/security/bug-caused login → engineering).
- `needs_human` true when: person requested, money to refund/correct, security/legal/data loss, angry + threatening to leave, multi-user outage, or no self-serve answer. False for how-to, praise, simple feature requests.
- Data: ~3k English Tobi-Bueck tickets (filtered queues) + ~500 real TWCS tweets to software brands (user sets up a Kaggle token).
- Labels: Claude labels all six fields from scratch with these definitions; dataset `priority`/`queue`/`type`/tags passed only as hints.
- State: `Subject: …\n\n<body>` with sign-off/`[Your Name]` boilerplate stripped (`schema.state()`); tweets body only; server truncates to 512 tokens.
- Sanity run on local Laya base: works end to end, ~200 ms warm for six questions; obvious base errors (praise → engineering, needs_human 0.87 on praise).
