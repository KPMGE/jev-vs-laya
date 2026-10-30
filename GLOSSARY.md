# Glossary

**Message**: one inbound text from a user of the platform to its support: a ticket (subject + body) or a tweet. The unit the classifier triages.

**Triage**: the six decisions made about one message: issue type, team, urgency, churn risk, sentiment, needs human.

**Issue type**: the single main thing the user needs resolved. When a message mixes concerns, the main ask wins. *Cancellation* means the user actually asks to cancel or downgrade; a threat to leave over another problem is not a cancellation.

**Team**: who handles the message. It usually follows the issue type, but the underlying cause can override it (an outage or security issue always goes to engineering).

**Urgency**: how soon the message needs a response, on five levels from "no time pressure" (1) to "business-critical right now" (5).

**Churn risk**: how likely the user is to stop using or paying for the product: low, medium or high. A threat to leave makes it high even when the issue type is something else.

**Needs human**: the message needs a person rather than an automated reply or help article.

**Contender**: a model being evaluated: Laya base, Laya fine-tuned, or Jev.

**Gold set**: the 200 held-out messages whose labels a human has checked. Every accuracy figure is measured against it.
