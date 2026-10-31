"""Deterministic matching in 4 passes, strongest key first. No LLM."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta

from rapidfuzz import fuzz

from . import normalise as N
from .schema import Invoice, Match, TwoBRow

DATE_WINDOW_DAYS = 5
FUZZY_MIN = 85


@dataclass
class MatchResult:
    matches: list[Match] = field(default_factory=list)
    unmatched_invoices: list[Invoice] = field(default_factory=list)
    unmatched_rows: list[TwoBRow] = field(default_factory=list)

    def by_pass(self) -> dict[int, int]:
        c = Counter(m.pass_no for m in self.matches)
        return {p: c.get(p, 0) for p in (1, 2, 3, 4)}


def _amount_key(x: float) -> float:
    return round(x, 0)


def match(invoices: list[Invoice], rows: list[TwoBRow], period: str,
          passes: tuple[int, ...] = (1, 2, 3, 4), prior: list[Match] | None = None) -> MatchResult:
    """Run the given passes. `prior` matches (from earlier passes) are kept and their rows/invoices skipped,
    so the passes can be run one at a time; the result always holds prior + new matches."""
    p_start, p_end = N.period_bounds(period)
    win_lo, win_hi = p_start - timedelta(days=DATE_WINDOW_DAYS), p_end + timedelta(days=DATE_WINDOW_DAYS)

    usable = [i for i in invoices if i.parse_status != "unreadable" and i.supplier_gstin and i.total is not None]
    open_inv = {i.invoice_id: i for i in usable}
    open_rows = {r.row_id: r for r in rows}
    res = MatchResult(matches=list(prior or []))
    for m in prior or []:
        open_inv.pop(m.invoice_id, None)
        open_rows.pop(m.row_id, None)

    def take(inv: Invoice, row: TwoBRow, pass_no: int, conf: float, notes: str):
        res.matches.append(Match(invoice_id=inv.invoice_id, row_id=row.row_id, pass_no=pass_no,
                                 confidence=conf, notes=notes))
        open_inv.pop(inv.invoice_id)
        open_rows.pop(row.row_id)

    def rows_of(gstin: str) -> list[TwoBRow]:
        return [r for r in open_rows.values() if r.supplier_gstin == gstin]

    # Pass 1: exact GSTIN + raw invoice number + total
    for inv in list(open_inv.values()) if 1 in passes else []:
        for r in rows_of(inv.supplier_gstin):
            if r.invoice_no.strip() == (inv.invoice_no or "").strip() and abs(r.total - inv.total) < 0.005:
                take(inv, r, 1, 1.0, "exact")
                break

    # Pass 2: normalised number + total within tolerance (closest amount wins)
    for inv in list(open_inv.values()) if 2 in passes else []:
        cands = [r for r in rows_of(inv.supplier_gstin)
                 if r.invoice_no_norm == inv.invoice_no_norm and N.within_tolerance(r.total, inv.total)]
        if cands:
            r = min(cands, key=lambda r: abs(r.total - inv.total))
            diffs = []
            if r.invoice_no != inv.invoice_no:
                diffs.append(f"number '{inv.invoice_no}' vs '{r.invoice_no}'")
            if abs(r.total - inv.total) >= 0.005:
                diffs.append(f"amount gap {r.total - inv.total:+.2f}")
            take(inv, r, 2, 0.95, "normalised: " + ", ".join(diffs))

    # Pass 3: same GSTIN + amount + date window, exactly one candidate.
    # An amount is only a usable key if this vendor did not bill that amount more than once.
    inv_amounts = Counter((i.supplier_gstin, _amount_key(i.total)) for i in usable)
    row_amounts = Counter((r.supplier_gstin, _amount_key(r.total)) for r in rows)
    for inv in list(open_inv.values()) if 3 in passes else []:
        if not inv.invoice_date or not (win_lo <= inv.invoice_date <= win_hi):
            continue
        if inv_amounts[(inv.supplier_gstin, _amount_key(inv.total))] > 1:
            continue
        cands = [r for r in rows_of(inv.supplier_gstin)
                 if N.within_tolerance(r.total, inv.total)
                 and abs((r.invoice_date - inv.invoice_date).days) <= DATE_WINDOW_DAYS]
        if len(cands) == 1 and row_amounts[(inv.supplier_gstin, _amount_key(cands[0].total))] == 1:
            r = cands[0]
            take(inv, r, 3, 0.85,
                 f"date window: number '{inv.invoice_no}' vs '{r.invoice_no}', "
                 f"date {inv.invoice_date} vs {r.invoice_date}")

    # Pass 4: fuzzy invoice number + total within tolerance (low confidence, sent to the agent)
    for inv in list(open_inv.values()) if 4 in passes else []:
        scored = [(fuzz.ratio(inv.invoice_no_norm, r.invoice_no_norm), r) for r in rows_of(inv.supplier_gstin)
                  if N.within_tolerance(r.total, inv.total)]
        scored = [(s, r) for s, r in scored if s >= FUZZY_MIN]
        if scored:
            s, r = max(scored, key=lambda x: x[0])
            take(inv, r, 4, 0.6, f"fuzzy: '{inv.invoice_no}' vs '{r.invoice_no}' (score {s:.0f})")

    res.unmatched_invoices = [i for i in invoices if i.invoice_id in open_inv or i not in usable]
    res.unmatched_rows = list(open_rows.values())
    return res
