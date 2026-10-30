"""The triage label schema: six fields, each asked as one typed question.

Laya and Jev receive exactly these questions, all six in one `/v1/systemone` call
per message. The Claude labeling prompt reuses the same definitions, so the gold
labels and the models are judged against identical wording.

Score answers are 0-based: urgency level 1 is index 0, level 5 is index 4.
"""

from __future__ import annotations

import re

MAX_STATE_TOKENS = 512  # Laya's English checkpoint reads at most 512 tokens

ISSUE_TYPES = {
    "bug": "Something in the product is broken, erroring, slow or behaving wrongly",
    "billing": "Charges, invoices, refunds, payment methods, pricing of the current plan",
    "cancellation": "The user explicitly asks to cancel, downgrade, close the account or not renew",
    "how-to": "The user asks how to do something the product already supports",
    "feature-request": "The user asks for something the product does not do today",
    "account-access": "Login, password, 2FA, permissions, user seats, account settings",
    "praise": "Compliments, thanks or positive feedback with no problem to solve",
}

TEAMS = {
    "engineering": "Bugs, outages, performance, security incidents, data loss, integrations failing",
    "billing": "Charges, refunds, invoices, payment methods",
    "customer-success": "Cancellations, renewals, plan changes, retention, product feedback and praise",
    "support": "How-to questions, account access, general enquiries",
}

URGENCY = [
    "1: no time pressure; a question, idea or compliment",
    "2: minor inconvenience; a workaround exists or it can wait days",
    "3: a real problem for one user that should be handled within a day",
    "4: blocks the user's or team's work, money is wrong, or the user demands action today",
    "5: business-critical; outage, security breach or data loss affecting many users right now",
]

CHURN_RISK = [
    "low: no sign the user is unhappy with the product or considering leaving",
    "medium: frustration, repeated problems, price complaints or comparing alternatives",
    "high: the user threatens to leave, asks to cancel or says they are moving to a competitor",
]

SENTIMENTS = {
    "negative": "Frustrated, angry, disappointed or worried",
    "neutral": "Matter-of-fact, no strong emotion",
    "positive": "Happy, grateful or enthusiastic",
}

QUESTIONS: dict[str, dict] = {
    "issue_type": {
        "type": "choice",
        "instructions": (
            "What is the main thing this user needs? Pick the single issue they want resolved. "
            "Choose cancellation only if they actually ask to cancel or downgrade; a threat to "
            "leave over another problem is not a cancellation."
        ),
        "criteria": ISSUE_TYPES,
    },
    "team": {
        "type": "choice",
        "instructions": (
            "Which team should handle this message? Usually bugs go to engineering, billing to "
            "billing, cancellations, praise and feature requests to customer-success, how-to and "
            "account access to support. Override that when the cause points elsewhere, e.g. a "
            "login failure caused by a bug or any outage or security issue goes to engineering."
        ),
        "criteria": TEAMS,
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgently does this message need a response?",
        "criteria": URGENCY,
    },
    "churn_risk": {
        "type": "score",
        "instructions": "How likely is this user to stop using or paying for the product?",
        "criteria": CHURN_RISK,
    },
    "sentiment": {
        "type": "choice",
        "instructions": "What is the user's emotional tone?",
        "criteria": SENTIMENTS,
    },
    "needs_human": {
        "type": "noul",
        "instructions": (
            "Does this message need a person rather than an automated reply or help article? "
            "True if the user asks for a person, money must be refunded or corrected, there is a "
            "security, legal or data-loss issue, the user is angry and threatening to leave, an "
            "outage affects many users, or no self-serve answer could resolve it. False for "
            "plain how-to questions, praise and simple feature requests."
        ),
    },
}

FIELDS = tuple(QUESTIONS)

_SIGNOFF = re.compile(
    r"\n\s*(best regards|kind regards|regards|sincerely|thanks|thank you|cheers|best),?\s*\n.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_PLACEHOLDER = re.compile(r"\[(your name|name|company|your company|contact information)\]", re.IGNORECASE)


def state(body: str, subject: str | None = None) -> str:
    """The text a model sees: "Subject: …" plus the body, sign-off boilerplate removed.

    Truncation to MAX_STATE_TOKENS happens server-side; tweets have no subject.
    """
    body = _PLACEHOLDER.sub("", _SIGNOFF.sub("", body.strip())).strip()
    return f"Subject: {subject.strip()}\n\n{body}" if subject else body
