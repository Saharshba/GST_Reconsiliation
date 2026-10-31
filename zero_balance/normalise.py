"""Normalisation of invoice numbers, dates and amounts."""
from __future__ import annotations

import re
from datetime import date, datetime

_SEP_RE = re.compile(r"[-/_]")
_NUM_SEG_RE = re.compile(r"\d+")

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def invoice_no(raw: str | None) -> str:
    """INV / 2026-27 / 0042 -> INV-2026-27-42 (comparison key only)."""
    if not raw:
        return ""
    s = re.sub(r"\s+", "", raw.upper())
    s = _SEP_RE.sub("-", s)
    return _NUM_SEG_RE.sub(lambda m: str(int(m.group())), s)


def parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    s = raw.strip().replace(",", " ")
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d", "%d-%m-%y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    m = re.match(r"^(\d{1,2})[\s-]*([A-Za-z]{3})[A-Za-z]*[\s-]*(\d{4})$", s)
    if m and m.group(2).lower() in MONTHS:
        return date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.match(r"^([A-Za-z]{3})[A-Za-z]*\s+(\d{1,2})\s+(\d{4})$", s)
    if m and m.group(1).lower() in MONTHS:
        return date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
    return None


def amount(raw) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return round(float(raw), 2)
    s = re.sub(r"(?i)rs\.?|inr|₹|\s", "", str(raw)).replace(",", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def tolerance(total: float) -> float:
    return max(1.0, 0.001 * abs(total))


def within_tolerance(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance(max(abs(a), abs(b))) + 1e-9


def period_bounds(period: str) -> tuple[date, date]:
    """'092026' -> (2026-09-01, 2026-09-30)."""
    m, y = int(period[:2]), int(period[2:])
    start = date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return start, date.fromordinal(nxt.toordinal() - 1)


def next_period(period: str) -> str:
    m, y = int(period[:2]), int(period[2:])
    return f"{m % 12 + 1:02d}{y + (m == 12)}"


def fmt_inr(x: float | None) -> str:
    """12345678.5 -> '1,23,45,678.50' (Indian grouping)."""
    if x is None:
        return "-"
    neg = x < 0
    s = f"{abs(x):.2f}"
    whole, frac = s.split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = ",".join([head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1])
        whole = f"{head},{tail}"
    return ("-" if neg else "") + whole + "." + frac
