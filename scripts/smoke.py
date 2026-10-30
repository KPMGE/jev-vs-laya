"""Smoke test: one support message, one choice question, against local Laya (and Jev if keyed).

    ./scripts/serve_laya.sh          # in another terminal
    uv run scripts/smoke.py
"""

import sys

from dotenv import load_dotenv

from triage.client import DecisionClient, jev, laya
from triage.schema import QUESTIONS as SCHEMA

MESSAGE = "I was charged twice on my card this month and nobody answers. Fix it TODAY or I'm cancelling."

QUESTIONS = {"issue_type": SCHEMA["issue_type"]}


def main() -> int:
    load_dotenv()
    backends = [laya()]
    if (j := jev()) is not None:
        backends.append(j)
    else:
        print("TYPESAFE_API_KEY not set: skipping Jev")

    for backend in backends:
        with DecisionClient(backend) as client:
            client.decide(MESSAGE, QUESTIONS)  # warm-up: first call loads weights / opens TLS
            d = client.decide(MESSAGE, QUESTIONS)
        a = d.answers["issue_type"]
        print(
            f"{d.backend:10} model={d.model:14} answer={a['choice']:14} "
            f"confidence={d.confidence('issue_type'):.3f} latency={d.latency_s * 1000:.0f} ms "
            f"tokens={d.input_tokens}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
