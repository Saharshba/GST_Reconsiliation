"""Local diagnoser: sentence-transformers all-MiniLM-L6-v2 (no API, no credits).

MiniLM is an embedding model, not a chat model, so it cannot call tools or write prose. Two modes
(MINILM_MODE env):

fewshot (default): k-nearest-neighbour vote over a labelled bank of past open items (agent/minilm_bank.json,
  built from seeds 1-3 by agent/build_minilm_bank.py; the demo data is seed 42, so it is held out).
  Confidence = similarity-weighted share of the k neighbours that agree.

zeroshot: classifies each evidence line ("fact") by cosine similarity to short descriptions of each cause:
  - one cause supported            -> that cause, confidence from a softmax over cause scores
  - facts point to different causes -> "unclear" (conflicting evidence)
  - no fact clears the similarity floor -> "unclear" with low confidence
The tool sequence is driven by the deterministic ScriptedLlm policy; the explanation is the evidence line that
best supports the chosen cause, so it never contains numbers that are not in the evidence.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import numpy as np

from .models import ScriptedLlm

MODEL_NAME = os.environ.get("MINILM_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODE = os.environ.get("MINILM_MODE", "fewshot")
BANK = Path(__file__).with_name("minilm_bank.json")
K = 5
SIM_FLOOR = 0.30       # a fact must be at least this similar to a cause description to count as support
TEMPERATURE = 0.08     # softmax temperature over per-cause scores

# Cause descriptions = the "Signal" column of the deck's Diagnosis slide plus the CLAUDE.md §5 definitions.
# Deliberately NOT copies of the evidence wording in agent/tools.py, so classification rests on semantic
# similarity rather than string overlap.
PROTOTYPES: dict[str, list[str]] = {
    "wrong_gstin": [
        "Same invoice and amount under a different GSTIN",
        "the identical bill and value were filed against another supplier's tax registration number",
    ],
    "vendor_not_filed": [
        "No 2B rows for the supplier in the period",
        "the supplier has not uploaded any invoices to the return for this month",
    ],
    "amount_typo": [
        "Same GSTIN and number, small amount gap",
        "same supplier and invoice number but the amount in 2B differs from the invoice",
    ],
    "period_mismatch": [
        "Appears in an adjacent month",
        "the invoice is present in the next month's 2B instead of this one",
    ],
    "unreadable": [
        "the scanned document has no readable text",
        "the file could not be read or the text is corrupted",
    ],
    "unclear": [
        "Conflicting signals or low confidence",
        "a close but not exact invoice number match that a person should confirm",
        "the bill number exists under another registration but with a different value",
    ],
}


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _prototypes():
    labels = [c for c, ps in PROTOTYPES.items() for _ in ps]
    emb = _model().encode([p for ps in PROTOTYPES.values() for p in ps], normalize_embeddings=True)
    return labels, emb


def classify_facts(facts: list[str]) -> list[tuple[str, str, float]]:
    """For each fact: (fact, best cause, similarity)."""
    if not facts:
        return []
    labels, proto = _prototypes()
    sims = _model().encode(facts, normalize_embeddings=True) @ proto.T
    out = []
    for f, row in zip(facts, sims):
        j = int(np.argmax(row))
        out.append((f, labels[j], float(row[j])))
    return out


@lru_cache(maxsize=1)
def _bank():
    ex = json.loads(BANK.read_text())["examples"]
    return [e["cause"] for e in ex], _model().encode([e["text"] for e in ex], normalize_embeddings=True)


def diagnose_fewshot(item: dict) -> tuple[str, float, str]:
    facts = item.get("facts", [])
    if not facts:
        return "unclear", 0.3, "No evidence lines to compare."
    labels, emb = _bank()
    sims = emb @ _model().encode(["; ".join(facts)], normalize_embeddings=True)[0]
    top = np.argsort(-sims)[:K]
    votes: dict[str, float] = {}
    for j in top:
        votes[labels[j]] = votes.get(labels[j], 0.0) + float(sims[j])
    cause = max(votes, key=votes.get)
    conf = votes[cause] / sum(votes.values())
    fact = "; ".join(facts) if cause in ("unclear", "unreadable") else facts[0]
    return cause, conf, f"Evidence: {fact}"[:240]


def diagnose(item: dict) -> tuple[str, float, str]:
    return diagnose_fewshot(item) if MODE == "fewshot" else diagnose_zeroshot(item)


def diagnose_zeroshot(item: dict) -> tuple[str, float, str]:
    per_fact = classify_facts(item.get("facts", []))
    support = [(f, c, s) for f, c, s in per_fact if s >= SIM_FLOOR]
    if not support:
        return "unclear", 0.3, "No evidence line clearly matches a known mismatch cause."
    causes = {c for _, c, _ in support}
    if len(causes) > 1:
        conflict = "; ".join(f for f, _, _ in support)
        return "unclear", 0.5, f"Conflicting evidence ({', '.join(sorted(causes))}): {conflict}"[:240]
    cause = causes.pop()
    best = {c: max((s for _, cc, s in per_fact if cc == c), default=0.0) for c in PROTOTYPES}
    scores = np.array(list(best.values())) / TEMPERATURE
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    conf = float(probs[list(best).index(cause)])
    fact = max(support, key=lambda x: x[2])[0]
    return cause, conf, f"Evidence: {fact}"[:240]


class MiniLmLlm(ScriptedLlm):
    """ADK model backend: deterministic tool policy + MiniLM diagnosis. Model id: local/minilm."""

    LABEL: ClassVar[str] = "local MiniLM"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"local/minilm.*"]

    def diagnose(self, item: dict) -> tuple[str, float, str]:
        return diagnose(item)
