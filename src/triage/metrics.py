"""Scoring a contender's answers against reference labels.

Per field type:
- choice (issue_type, team, sentiment): accuracy, macro-F1, ECE
- score (urgency, churn_risk): accuracy after rounding to the nearest level, MAE of the raw
  fractional score, ECE
- noul (needs_human): accuracy and F1 for "yes" at P ≥ 0.5, AUC, ECE

ECE uses the top-option probability (the model's probability that its own answer is right),
in 10 equal-width bins. The trust gate uses Jev's confidence formula instead, recomputed from
`probabilities`, because that is the number a production gate would threshold.
"""

from __future__ import annotations

import statistics

from triage.labeling import CHURN_LEVELS
from triage.schema import QUESTIONS

ECE_BINS = 10
GATE_AUTO, GATE_CONFIRM = 0.8, 0.5


def reference_value(field: str, labels: dict):
    """A gold label in the answer's terms: score fields become 0-based level indices."""
    v = labels[field]
    if field == "urgency":
        return v - 1
    if field == "churn_risk":
        return CHURN_LEVELS.index(v)
    return v


def _probs(answer: dict) -> list[float]:
    p = answer.get("probabilities")
    if isinstance(p, dict):
        return list(p.values())
    return list(p or [])


def top_probability(field: str, answer: dict) -> float:
    if QUESTIONS[field]["type"] == "noul":
        p = answer["noul"]
        return max(p, 1 - p)
    return max(_probs(answer))


def jev_confidence(answer: dict) -> float | None:
    probs = _probs(answer)
    if not probs:
        return None
    n = len(probs)
    return (n * max(probs) - 1) / (n - 1) if n > 1 else 1.0


def predicted(field: str, answer: dict):
    kind = QUESTIONS[field]["type"]
    if kind == "choice":
        return answer["choice"]
    if kind == "score":
        return round(answer["score"])
    return answer["noul"] >= 0.5


def ece(confidences: list[float], correct: list[bool]) -> float:
    total, n = 0.0, len(confidences)
    for b in range(ECE_BINS):
        lo, hi = b / ECE_BINS, (b + 1) / ECE_BINS
        idx = [i for i, c in enumerate(confidences) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confidences[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return total


def macro_f1(gold: list, pred: list) -> float:
    scores = []
    for c in sorted(set(gold) | set(pred), key=str):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(scores) / len(scores)


def auc(gold: list[bool], scores: list[float]) -> float | None:
    pos = [s for g, s in zip(gold, scores) if g]
    neg = [s for g, s in zip(gold, scores) if not g]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def score_field(field: str, refs: list, answers: list[dict]) -> dict:
    kind = QUESTIONS[field]["type"]
    preds = [predicted(field, a) for a in answers]
    correct = [p == r for p, r in zip(preds, refs)]
    out = {
        "accuracy": sum(correct) / len(correct),
        "ece": ece([top_probability(field, a) for a in answers], correct),
    }
    if kind == "choice":
        out["macro_f1"] = macro_f1(refs, preds)
    elif kind == "score":
        out["mae"] = sum(abs(a["score"] - r) for a, r in zip(answers, refs)) / len(refs)
    else:
        tp = sum(p and r for p, r in zip(preds, refs))
        fp = sum(p and not r for p, r in zip(preds, refs))
        fn = sum(r and not p for p, r in zip(preds, refs))
        out["f1_yes"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        out["auc"] = auc(refs, [a["noul"] for a in answers])
    return out


def trust_gate(refs: list[dict], answers: list[dict]) -> dict:
    """Coverage and joint issue_type+team accuracy per confidence band (min of the two)."""
    bands = {"auto": [], "confirm": [], "human": []}
    for labels, ans in zip(refs, answers):
        conf = min(jev_confidence(ans["issue_type"]), jev_confidence(ans["team"]))
        ok = ans["issue_type"]["choice"] == labels["issue_type"] and ans["team"]["choice"] == labels["team"]
        band = "auto" if conf >= GATE_AUTO else "confirm" if conf >= GATE_CONFIRM else "human"
        bands[band].append(ok)
    n = len(refs)
    return {b: {"coverage": len(v) / n, "accuracy": (sum(v) / len(v)) if v else None} for b, v in bands.items()}


def latency(seconds: list[float]) -> dict:
    s = sorted(seconds)
    return {"p50_ms": statistics.median(s) * 1000, "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))] * 1000}
