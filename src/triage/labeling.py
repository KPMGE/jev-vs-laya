"""Claude labels candidate messages with the same definitions the models are asked.

The system prompt is rendered from `schema.QUESTIONS`, so the labeler and the
contenders read identical field definitions. The dataset's own labels go in as
hints only. Output is constrained by a JSON schema (`output_config.format`).
"""

from __future__ import annotations

import json

from triage.schema import CHURN_RISK, ISSUE_TYPES, QUESTIONS, SENTIMENTS, TEAMS, URGENCY, state

# alias → (model id, $/MTok input, cache read, cache write, output)
MODELS = {
    "opus": ("claude-opus-5-5", 4.0, 0.20, 5.0, 20.0),
    "haiku": ("claude-haiku-5-5", 0.10, 0.01, 0.125, 0.50),
}
EFFORT = "low"
MAX_TOKENS = 4096
CHURN_LEVELS = [level.split(":")[0] for level in CHURN_RISK]  # low, medium, high


def _render_field(name: str, q: dict) -> str:
    lines = [f"### {name} ({q['type']})", q["instructions"]]
    criteria = q.get("criteria")
    if isinstance(criteria, dict):
        lines += [f"- {k}: {v}" for k, v in criteria.items()]
    elif isinstance(criteria, list):
        lines += [f"- {c}" for c in criteria]
    return "\n".join(lines)


SYSTEM = "\n\n".join([
    "You label inbound support messages sent by users of software platforms. Each message gets six "
    "labels. Apply the definitions below exactly; they are the same definitions the classifiers "
    "being evaluated receive, and your labels become the ground truth they are scored against.",
    "The message may come with hints from its source dataset (its original type, queue, priority and "
    "tags, or the brand it was sent to). Hints are often wrong or coarser than our categories: use "
    "them as weak evidence, never copy them over the message's actual content.",
    "Set `ambiguous` to true only for the hard cases: you would put less than about 70% on your "
    "answer for issue_type, team or needs_human, or the message lacks the information to decide. "
    "A borderline urgency or churn level alone is not ambiguous; most messages are not. Keep `note` "
    "to one short sentence explaining the hardest call.",
    "## Fields",
    *(_render_field(name, q) for name, q in QUESTIONS.items()),
    "For `urgency` answer the level number 1-5. For `churn_risk` answer low, medium or high. "
    "For `needs_human` answer true or false.",
])

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "issue_type": {"type": "string", "enum": list(ISSUE_TYPES)},
        "team": {"type": "string", "enum": list(TEAMS)},
        "urgency": {"type": "integer", "enum": list(range(1, len(URGENCY) + 1))},
        "churn_risk": {"type": "string", "enum": CHURN_LEVELS},
        "sentiment": {"type": "string", "enum": list(SENTIMENTS)},
        "needs_human": {"type": "boolean"},
        "ambiguous": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["issue_type", "team", "urgency", "churn_risk", "sentiment", "needs_human", "ambiguous", "note"],
    "additionalProperties": False,
}


def user_content(candidate: dict) -> str:
    hints = json.dumps(candidate["hints"], ensure_ascii=False)
    message = state(candidate["body"], candidate.get("subject"))
    return f"<hints>{hints}</hints>\n\n<message>\n{message}\n</message>"


def request_params(candidate: dict, alias: str = "opus") -> dict:
    """Messages API params for one candidate; the same dict works sync and in a batch."""
    return {
        "model": MODELS[alias][0],
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        "messages": [{"role": "user", "content": user_content(candidate)}],
    }


def cost_usd(usage: dict, alias: str) -> float:
    _, inp, read, write, out = MODELS[alias]
    return (usage["input"] * inp + usage["cache_read"] * read + usage["cache_write"] * write + usage["output"] * out) / 1e6


def parse(message) -> dict | None:
    """Labels from a Messages response, or None if the model refused or ran out of tokens."""
    if message.stop_reason in ("refusal", "max_tokens"):
        return None
    text = next((b.text for b in message.content if b.type == "text"), None)
    return json.loads(text) if text else None
