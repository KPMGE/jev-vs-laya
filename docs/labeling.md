# Labeling

How every message gets its six labels, how the splits are drawn, and how the gold set is checked by a human.

```bash
uv run scripts/label.py submit  --model haiku              # all candidates, one Message Batch
uv run scripts/label.py collect --model haiku              # wait, then append to data/labels.haiku.jsonl
uv run scripts/split.py                                    # data/splits.json + data/eval_ids.txt
uv run scripts/label.py submit  --model opus --ids data/eval_ids.txt
uv run scripts/label.py collect --model opus               # data/labels.opus.jsonl
uv run scripts/review.py                                   # human review of the 200 gold messages
```

`submit` only sends candidates that aren't labeled yet, so re-running after adding candidates labels just the new ones. `pilot --model <m> --n 20` labels a small sample synchronously for trying prompt changes.

## Budget plan

The whole labeling run had to cost under $5. Opus 5.5 on all ~4,300 messages would have cost about $14 through Batches, so the work is split:

| Step | Model | Messages | Cost |
| --- | --- | --- | --- |
| Label everything | Claude Haiku 5.5 (`claude-haiku-5-5`), effort `low`, Batches | 4,300 | $0.21 |
| Relabel what we score against (gold + val) | Claude Opus 5.5 (`claude-opus-5-5`), effort `low`, Batches | 500 | $0.83 |
| Check the gold set | a human, in `scripts/review.py` | 200 | your time |

With the four 20-message pilots (about $0.30), labeling cost roughly **$1.35** in total.

So:
- **train** uses Haiku labels;
- **val** uses Opus labels;
- **gold** uses Opus labels corrected by a human.

Comparing Haiku with Opus on the 500 shared messages measures how noisy the training labels are (see [Label quality](#label-quality)).

The Batches API costs half the normal price and doesn't accept server-side refusal fallbacks. A batch item that's refused or truncated is retried synchronously. Opus retries opt into `fallbacks: "default"`; Haiku has no server-side fallback.

## The prompt

`src/triage/labeling.py`. The system prompt is **rendered from `src/triage/schema.py`**, so the labeler reads exactly the field instructions and value descriptions that Laya and Jev get. It adds three things:

- the dataset's own labels (`type`, `queue`, `priority`, tags, or the tweet's `brand`) come as **hints**: weak evidence, never copied over the message content;
- an `ambiguous` flag, set only when the labeler would put less than about 70% on issue_type, team or needs_human, or when the message lacks the information to decide;
- a one-sentence `note` explaining the hardest call, which is shown in the review page.

The user turn is `<hints>{…}</hints>` followed by `<message>` holding `schema.state(body, subject)`, the same text the models see. The output is constrained by a JSON schema (`output_config.format`), so every response is valid JSON with values from the allowed enums. The system prompt is cached (`cache_control`).

### What the pilot changed

Two 20-message pilots ran before the full batch:

1. **`ambiguous` fired on 17 of 20.** The first definition ("a careful human could pick differently for any field") was too loose, because urgency and churn are almost always arguable. Restricting it to issue_type, team and needs_human at under 70% brought it down to 7 of 20, each with a useful reason.
2. **`needs_human` was true on 13 of 20**, which covered nearly every bug, because "no self-serve answer could resolve it". The definition was tightened to "a person must step in beyond the normal queue" (a person was asked for, money must be corrected, security/legal/data loss, an angry user threatening to leave, or a multi-user outage). That brought it down to 5 of 20.
3. Tobi bodies contain `<br>` tags and inline sign-offs ("… Best regards, [Your Name]"). `schema.state()` now strips both.

On the same 20 messages Haiku agreed with Opus 85–95% per field (issue_type 19/20, team 18, urgency 18, churn 17, sentiment 17, needs_human 19).

## Rare issue types: the boost pool

Haiku found only 17 cancellations and 17 praise messages among the 3,500 base candidates. The gold set's floor of 15 per type would have left 2 of each for training. So `scripts/fetch_data.py` adds a **boost pool** of keyword-matched TWCS tweets (`pool: "boost"`), 200 per group, drawn from the software-brand tweets not already picked:

| Group | Pattern (abridged) | Matches |
| --- | --- | --- |
| cancellation | cancel, unsubscribe, refund, close/delete my account, switching to, leaving | 2,796 |
| praise | love, thanks, thank you, awesome, amazing, great job | 6,866 |
| account-access | password, log in, sign in, locked out, 2FA, hacked, verification code | 3,591 |
| praise-strict | the praise words, excluding tweets containing not, n't, ?, error, issue, problem, help, fix, why, but, still | 2,951 |

Most "thanks" and "love" tweets are sarcastic complaints, hence the strict group. The keywords only decide *which* tweets get labeled. The label itself still comes from the labeler, so a sarcastic "love it when it crashes" is labeled a bug.

**Boost tweets never go into gold.** The gold set keeps the natural mix of the base pool; boost tweets feed only train and val.

## Splits

`scripts/split.py`, seeded, stratified on Haiku's `issue_type`:

- **gold (200)**: from the base pool only. At least 15 per issue type, exactly 50 tweets, the rest random tickets.
- **val (300)**: proportional to the issue-type mix of everything left, boost included.
- **train**: the rest.

| Split | Size | Tweets (boost) | account-access | billing | bug | cancellation | feature-request | how-to | praise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gold | 200 | 50 (0) | 15 | 15 | 84 | 15 | 18 | 38 | 15 |
| val | 300 | 124 (81) | 17 | 21 | 145 | 5 | 41 | 67 | 4 |
| train | 3,800 | 1,126 (719) | 215 | 268 | 1,837 | 59 | 519 | 855 | 47 |

These counts use Haiku's issue_type. Gold and val are relabeled by Opus, so their final mix shifts slightly. **Praise and cancellation stay thin in train** (47 and 59), so expect the fine-tuned model to be weakest there.

## Gold review

```bash
uv run scripts/review.py          # opens http://127.0.0.1:8765
```

The page shows the message exactly as the models see it, Claude's note, and the dataset hints (collapsed). Next to it are six field pickers preset to Opus's proposal. Ambiguous messages come first.

| Key | Action |
| --- | --- |
| ↑ / ↓ | move between fields |
| 1–7 | pick a value for the focused field |
| Enter | accept the message as shown and go to the next |
| ← / → | back / skip |

A dashed border marks Opus's proposal whenever the current choice differs, and changed fields are highlighted. Every decision is appended to `data/gold.jsonl` as `{id, labels, status: accepted|corrected, changed: [fields]}`, and the last line per id wins, so you can stop and resume at any time. `--items <file>` reviews any labels file, for example `--items data/labels.opus.pilot.jsonl`, which writes to `data/gold.pilot.jsonl`.

## Label quality

**Haiku vs Opus** on the 500 gold + val messages. This is a proxy for how noisy the Haiku training labels are:

| Field | Agreement |
| --- | --- |
| issue_type | 86% |
| team | 87% |
| urgency | 76% (mean absolute difference 0.24 levels) |
| churn_risk | 78% |
| sentiment | 84% |
| needs_human | 89% |

So roughly one training label in six to eight differs from what Opus would say. That caps how close a fine-tuned Laya can get to the gold set, and the ceiling is lowest for urgency and churn.

Opus doesn't always agree with Haiku's issue_type, so the gold set's final mix differs from the stratification target. Under Opus labels it is: bug 89, how-to 42, billing 23, account-access 14, feature-request 13, praise 11, cancellation 8. Opus flagged 71 of the 200 as ambiguous; the review page shows those first.

**Opus vs human** on the 200: _filled in after the review._
