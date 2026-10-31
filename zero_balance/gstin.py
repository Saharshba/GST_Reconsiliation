"""GSTIN format and checksum validation."""
from __future__ import annotations

import re

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
GSTIN_FIND_RE = re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")

STATES = {
    "07": "Delhi", "09": "Uttar Pradesh", "24": "Gujarat", "27": "Maharashtra",
    "29": "Karnataka", "32": "Kerala", "33": "Tamil Nadu", "36": "Telangana", "37": "Andhra Pradesh",
}


def checksum_char(first14: str) -> str:
    total = 0
    for i, c in enumerate(first14[:14]):
        p = CHARS.index(c) * (1 if i % 2 == 0 else 2)
        total += p // 36 + p % 36
    return CHARS[(36 - total % 36) % 36]


def make_gstin(state_code: str, pan: str, entity: str = "1") -> str:
    body = f"{state_code}{pan}{entity}Z"
    return body + checksum_char(body)


def validate(gstin: str) -> dict:
    g = (gstin or "").strip().upper()
    if len(g) != 15:
        return {"valid": False, "state_code": None, "reason": f"length {len(g)}, expected 15"}
    if not GSTIN_RE.match(g):
        return {"valid": False, "state_code": g[:2], "reason": "format does not match SS-PAN-E-Z-C pattern"}
    expected = checksum_char(g[:14])
    if g[14] != expected:
        return {"valid": False, "state_code": g[:2], "reason": f"checksum mismatch (expected {expected})"}
    return {"valid": True, "state_code": g[:2], "state": STATES.get(g[:2], "unknown"), "reason": "ok"}


def is_valid(gstin: str) -> bool:
    return validate(gstin)["valid"]
