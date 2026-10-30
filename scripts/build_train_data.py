"""Turn the labeled splits into Laya training files.

    uv run scripts/build_train_data.py

Writes data/train.jsonl (Haiku labels) and data/val.jsonl (Opus labels), one row per
message in the format `laya-train` reads: `{state, questions, expected}`, with the
exact questions from `schema.QUESTIONS`. Score answers are 0-based level indices, so
urgency 1–5 becomes 0–4 and churn low/medium/high becomes 0–2.
"""

import json
import sys
from pathlib import Path

from triage.labeling import CHURN_LEVELS
from triage.schema import QUESTIONS, state

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def expected(labels: dict) -> dict:
    return {
        "issue_type": labels["issue_type"],
        "team": labels["team"],
        "urgency": labels["urgency"] - 1,
        "churn_risk": CHURN_LEVELS.index(labels["churn_risk"]),
        "sentiment": labels["sentiment"],
        "needs_human": labels["needs_human"],
    }


def read_labels(path: Path) -> dict:
    return {json.loads(l)["id"]: json.loads(l)["labels"] for l in path.open(encoding="utf-8")}


def main() -> int:
    candidates = {json.loads(l)["id"]: json.loads(l) for l in (DATA / "candidates.jsonl").open(encoding="utf-8")}
    splits = json.loads((DATA / "splits.json").read_text())
    sources = {"train": read_labels(DATA / "labels.haiku.jsonl"), "val": read_labels(DATA / "labels.opus.jsonl")}

    for split, labels in sources.items():
        out = DATA / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for cid in splits[split]:
                c = candidates[cid]
                row = {"id": cid, "state": state(c["body"], c["subject"]), "questions": QUESTIONS,
                       "expected": expected(labels[cid])}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(splits[split])} rows → {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
