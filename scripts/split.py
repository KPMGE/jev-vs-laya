"""Split the labeled candidates into gold (200), val (300) and train (the rest).

    uv run scripts/split.py

Stratifies on Haiku's `issue_type` (the only labels that exist for every candidate):
- gold: at least 15 per issue type where available, about 50 tweets, the rest random
- val: proportional to the issue-type mix of what's left
- train: everything else
Gold is drawn from the base pool only; the keyword-matched boost tweets go to val/train.

Writes data/splits.json and data/eval_ids.txt (gold + val, for the Opus relabel).
Seeded, so re-running gives the same split.
"""

import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 7
GOLD_N, VAL_N = 200, 300
GOLD_FLOOR = 15
GOLD_TWEETS = 50


def main() -> int:
    rng = random.Random(SEED)
    cands = [json.loads(l) for l in open(DATA / "candidates.jsonl")]
    source = {c["id"]: c["source"] for c in cands}
    pool = {c["id"]: c.get("pool", "base") for c in cands}
    labels = {json.loads(l)["id"]: json.loads(l)["labels"] for l in open(DATA / "labels.haiku.jsonl")}
    ids = sorted(labels)
    rng.shuffle(ids)
    by_type = collections.defaultdict(list)
    for i in ids:
        if pool[i] == "base":  # gold keeps the natural mix; boost tweets only feed train/val
            by_type[labels[i]["issue_type"]].append(i)

    gold = []
    for t in sorted(by_type):
        gold += by_type[t][:GOLD_FLOOR]
    rest = [i for i in ids if i not in set(gold) and pool[i] == "base"]
    tweets_needed = max(0, GOLD_TWEETS - sum(source[i] == "twcs" for i in gold))
    tweets = [i for i in rest if source[i] == "twcs"][:tweets_needed]
    gold += tweets
    rest = [i for i in rest if i not in set(tweets)]
    gold += [i for i in rest if source[i] == "tobi"][: GOLD_N - len(gold)]
    gold_set = set(gold)
    rest = [i for i in ids if i not in gold_set]

    rest_by_type = collections.defaultdict(list)
    for i in rest:
        rest_by_type[labels[i]["issue_type"]].append(i)
    val = []
    for t, members in sorted(rest_by_type.items()):
        val += members[: round(VAL_N * len(members) / len(rest))]
    val = (val + [i for i in rest if i not in set(val)])[:VAL_N]
    train = [i for i in rest if i not in set(val)]

    splits = {"gold": sorted(gold), "val": sorted(val), "train": sorted(train)}
    (DATA / "splits.json").write_text(json.dumps(splits, indent=1) + "\n")
    (DATA / "eval_ids.txt").write_text("\n".join(splits["gold"] + splits["val"]) + "\n")

    for name, members in splits.items():
        types = collections.Counter(labels[i]["issue_type"] for i in members)
        tw = sum(source[i] == "twcs" for i in members)
        boost = sum(pool[i] == "boost" for i in members)
        print(f"{name:5} {len(members):5}  tweets {tw:4} (boost {boost:3})  {dict(sorted(types.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
