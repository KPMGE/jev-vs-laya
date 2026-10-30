"""Download the raw datasets and build the candidate pool of messages to label.

    uv run scripts/fetch_data.py

Writes data/candidates.jsonl: the base pool of 3,000 English Tobi-Bueck tickets + 500
TWCS tweets, then up to 800 keyword-matched "boost" tweets for rare issue types.
One JSON object per line with `source`, `id`, `subject`, `body`, `hints` (the
dataset's own labels, shown to the labeler as hints only) and `pool` (base/boost).

Downloads are skipped when the files are already in data/raw/. Sampling is seeded.
See docs/data.md.
"""

import collections
import csv
import hashlib
import html
import json
import os
import random
import re
import sys
import zipfile
from pathlib import Path

import httpx
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "candidates.jsonl"
SEED = 7

TOBI_REPO = "Tobi-Bueck/customer-support-tickets"
TOBI_REV = "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb"  # CC BY-NC 4.0
TOBI_FILES = ["aa_dataset-tickets-multi-lang-5-2-50-version.csv", "dataset-tickets-multi-lang-4-20k.csv"]
TOBI_QUEUES = [
    "Technical Support",
    "Product Support",
    "IT Support",
    "Customer Service",
    "Billing and Payments",
    "Service Outages and Maintenance",
    "General Inquiry",
]
TOBI_N = 3000

TWCS_DATASET = "thoughtvector/customer-support-on-twitter"  # CC BY-NC-SA 4.0
# Brands whose product is software or a software subscription, not airlines, retail, telecom or food.
TWCS_BRANDS = [
    "AppleSupport", "SpotifyCares", "hulu_support", "AdobeCare", "MicrosoftHelps", "AzureSupport",
    "DropboxSupport", "GoDaddyHelp", "TwitterSupport", "NortonSupport", "AWSSupport", "PandoraSupport",
    "GooglePlayMusic", "YahooCare", "asksalesforce", "mediatemplehelp", "OfficeSupport",
]
TWCS_N = 500
# Extra keyword-matched tweets for issue types the base pool barely has. They only
# feed train/val (`pool: "boost"`); the gold set is drawn from the base pool.
BOOST = {
    "cancellation": r"\b(cancel\w*|unsubscrib\w*|refund\w*|close my account|delete my account|switching to|leaving)\b",
    "praise": r"\b(love|loving|thank you|thanks|awesome|amazing|great job|you rock)\b",
    "account-access": r"\b(password|log ?in|sign ?in|locked out|2fa|two.factor|hacked|verification code)\b",
    # most "thanks"/"love" tweets are complaints; this one excludes problem words
    "praise-strict": r"^(?!.*(\bnot\b|n't|\?|\berror|\bissue|\bproblem|\bhelp|\bfix|\bwhy\b|\bbut\b|\bstill\b))"
                     r".*\b(love|awesome|amazing|great|best|thank you|thanks|kudos|brilliant)\b",
}
BOOST_N = 200  # per keyword group

MIN_BODY_CHARS = 40
MIN_TWEET_CHARS = 30


def fetch_tobi() -> list[Path]:
    return [
        Path(hf_hub_download(TOBI_REPO, fn, repo_type="dataset", revision=TOBI_REV, local_dir=RAW / "tobi"))
        for fn in TOBI_FILES
    ]


def fetch_twcs() -> Path | None:
    csv_path = RAW / "twcs" / "twcs" / "twcs.csv"
    if csv_path.exists():
        return csv_path
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        print("KAGGLE_API_TOKEN not set: skipping TWCS", file=sys.stderr)
        return None
    zip_path = RAW / "twcs" / "twcs.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.kaggle.com/api/v1/datasets/download/{TWCS_DATASET}"
    with httpx.stream("GET", url, headers={"Authorization": f"Bearer {token}"}, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(zip_path.parent)
    return csv_path


def tobi_candidates(paths: list[Path], rng: random.Random, counts: dict) -> list[dict]:
    seen, pool = set(), collections.defaultdict(list)
    total = english = 0
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                total += 1
                if r["language"] != "en":
                    continue
                english += 1
                body = (r["body"] or "").strip()
                if r["queue"] not in TOBI_QUEUES or len(body) < MIN_BODY_CHARS:
                    continue
                key = hashlib.sha1(body.lower().encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                tags = [r[f"tag_{i}"] for i in range(1, 9) if r.get(f"tag_{i}")]
                pool[r["queue"]].append({
                    "source": "tobi",
                    "id": f"tobi-{key[:12]}",
                    "subject": (r["subject"] or "").strip() or None,
                    "body": body,
                    "hints": {"type": r["type"], "queue": r["queue"], "priority": r["priority"], "tags": tags},
                })
    counts["tobi_rows"], counts["tobi_english"] = total, english
    counts["tobi_kept_by_queue"] = {q: len(pool[q]) for q in TOBI_QUEUES}

    # Balance queues: equal share per queue, small queues give their leftover to the rest.
    picked, remaining, queues = [], TOBI_N, sorted(TOBI_QUEUES, key=lambda q: len(pool[q]))
    for i, q in enumerate(queues):
        share = remaining // (len(queues) - i)
        take = rng.sample(pool[q], min(share, len(pool[q])))
        picked += take
        remaining -= len(take)
    counts["tobi_picked_by_queue"] = dict(collections.Counter(c["hints"]["queue"] for c in picked))
    return picked


_USER_ID = re.compile(r"@\d+\b")  # TWCS anonymises customers as @123456
_URL = re.compile(r"https?://\S+")


def clean_tweet(text: str) -> str:
    return re.sub(r"\s+", " ", _URL.sub("", _USER_ID.sub("", html.unescape(text)))).strip()


def looks_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.95


def twcs_candidates(path: Path, rng: random.Random, counts: dict) -> list[dict]:
    authors, firsts = {}, []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            authors[r["tweet_id"]] = r["author_id"] if r["inbound"] == "False" else None
            if r["inbound"] == "True" and not r["in_response_to_tweet_id"]:
                firsts.append(r)
    counts["twcs_first_inbound"] = len(firsts)

    by_brand = collections.defaultdict(list)
    for r in firsts:
        brand = next((authors.get(t) for t in r["response_tweet_id"].split(",") if authors.get(t)), None)
        if brand not in TWCS_BRANDS:
            continue
        text = clean_tweet(r["text"])
        if len(text) < MIN_TWEET_CHARS or not looks_english(text):
            continue
        by_brand[brand].append({
            "source": "twcs",
            "id": f"twcs-{r['tweet_id']}",
            "subject": None,
            "body": text,
            "hints": {"brand": brand},
        })
    counts["twcs_kept_by_brand"] = {b: len(by_brand[b]) for b in TWCS_BRANDS}

    picked, remaining, brands = [], TWCS_N, sorted(TWCS_BRANDS, key=lambda b: len(by_brand[b]))
    for i, b in enumerate(brands):
        share = remaining // (len(brands) - i)
        take = rng.sample(by_brand[b], min(share, len(by_brand[b])))
        picked += take
        remaining -= len(take)
    counts["twcs_picked_by_brand"] = dict(collections.Counter(c["hints"]["brand"] for c in picked))

    # Boost: keyword-matched tweets not already picked, drawn with their own seed so the
    # base sample above stays identical.
    boost_rng, taken = random.Random(SEED + 1), {c["id"] for c in picked}
    rest = [c for b in TWCS_BRANDS for c in by_brand[b] if c["id"] not in taken]
    boost = []
    for group, pattern in BOOST.items():
        rx = re.compile(pattern, re.IGNORECASE)
        matches = [c for c in rest if rx.search(c["body"]) and c["id"] not in taken]
        take = boost_rng.sample(matches, min(BOOST_N, len(matches)))
        taken |= {c["id"] for c in take}
        boost += [{**c, "pool": "boost"} for c in take]
        counts.setdefault("twcs_boost_matches", {})[group] = len(matches)
    counts["twcs_boost"] = len(boost)
    return picked, boost


def main() -> int:
    load_dotenv(ROOT / ".env")
    rng = random.Random(SEED)
    counts: dict = {}
    candidates = tobi_candidates(fetch_tobi(), rng, counts)
    boost = []
    if (twcs := fetch_twcs()) is not None:
        picked, boost = twcs_candidates(twcs, rng, counts)
        candidates += picked
    rng.shuffle(candidates)
    candidates = [{**c, "pool": "base"} for c in candidates] + boost

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    counts["candidates"] = len(candidates)
    (ROOT / "data" / "candidates.stats.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
