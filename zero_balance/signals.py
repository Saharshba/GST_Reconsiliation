"""Evidence for open items, computed in Python. States facts; the only conclusion is a rule-based suggestion."""
from __future__ import annotations

from rapidfuzz import fuzz

from . import normalise as N
from .gstin import is_valid
from .matching import MatchResult
from .schema import Invoice, OpenItem, TwoBRow


def _row_brief(r: TwoBRow) -> dict:
    return {"row_id": r.row_id, "gstin": r.supplier_gstin, "invoice_no": r.invoice_no,
            "date": r.invoice_date.isoformat(), "total": r.total}


def compute_signals(inv: Invoice, all_rows: list[TwoBRow], open_rows: list[TwoBRow],
                    next_rows: list[TwoBRow]) -> dict:
    s: dict = {"parse_status": inv.parse_status}
    if inv.parse_notes:
        s["parse_notes"] = inv.parse_notes
    if inv.parse_status == "unreadable":
        return s
    g, no, total = inv.supplier_gstin, inv.invoice_no_norm, inv.total
    s["gstin_valid"] = bool(g) and is_valid(g)
    s["vendor_has_rows_this_period"] = any(r.supplier_gstin == g for r in all_rows)

    other = [r for r in open_rows if r.supplier_gstin != g and r.invoice_no_norm == no]
    same_amt = [r for r in other if N.within_tolerance(r.total, total)]
    if same_amt:
        s["same_no_and_amount_other_gstin"] = [_row_brief(r) for r in same_amt]
    diff_amt = [r for r in other if not N.within_tolerance(r.total, total)]
    if diff_amt:
        s["same_no_other_gstin_amount_differs"] = [_row_brief(r) | {"gap": round(r.total - total, 2)}
                                                   for r in diff_amt]

    gap = [r for r in open_rows if r.supplier_gstin == g and r.invoice_no_norm == no
           and not N.within_tolerance(r.total, total)]
    if gap:
        s["same_gstin_and_no_amount_gap"] = [
            {"row_id": r.row_id, "invoice_total": total, "row_total": r.total, "gap": round(r.total - total, 2)}
            for r in gap]

    nxt = [r for r in next_rows if r.supplier_gstin == g and r.invoice_no_norm == no]
    if nxt:
        s["found_in_next_period"] = [_row_brief(r) | {"period": r.period,
                                                      "amount_matches": N.within_tolerance(r.total, total)}
                                     for r in nxt]

    fuzzy = sorted(((fuzz.ratio(no, r.invoice_no_norm), r) for r in open_rows if r.supplier_gstin == g
                    and r.invoice_no_norm != no), key=lambda x: -x[0])
    fuzzy = [(sc, r) for sc, r in fuzzy if sc >= 70][:3]
    if fuzzy:
        s["fuzzy_candidates"] = [_row_brief(r) | {"score": round(sc)} for sc, r in fuzzy]
    return s


RULES = [
    ("wrong_gstin", lambda s: bool(s.get("same_no_and_amount_other_gstin"))),
    ("vendor_not_filed", lambda s: s.get("vendor_has_rows_this_period") is False),
    ("amount_typo", lambda s: bool(s.get("same_gstin_and_no_amount_gap"))),
    ("period_mismatch", lambda s: bool(s.get("found_in_next_period"))),
]


def suggest(signals: dict, kind: str) -> tuple[str, list[str]]:
    if signals.get("parse_status") == "unreadable":
        return "unreadable", ["unreadable"]
    fired = [name for name, rule in RULES if rule(signals)]
    if kind == "fuzzy_match_candidate":
        return "unclear", fired + ["fuzzy_match_candidate"]
    if len(fired) == 1:
        return fired[0], fired
    return "unclear", fired


def build_open_items(invoices: list[Invoice], rows: list[TwoBRow], next_rows: list[TwoBRow],
                     mr: MatchResult) -> list[OpenItem]:
    by_id = {i.invoice_id: i for i in invoices}
    row_by_id = {r.row_id: r for r in rows}
    items: list[OpenItem] = []
    work = [(i, "unmatched", None) for i in mr.unmatched_invoices]
    work += [(by_id[m.invoice_id], "fuzzy_match_candidate", m) for m in mr.matches if m.pass_no == 4]
    for inv, kind, m in work:
        open_rows = mr.unmatched_rows + ([row_by_id[m.row_id]] if m else [])
        sig = compute_signals(inv, rows, open_rows, next_rows)
        if m:
            sig["fuzzy_match_candidate"] = {"row_id": m.row_id, "notes": m.notes, "confidence": m.confidence}
        cause, fired = suggest(sig, kind)
        sig["rules_fired"] = fired
        cands = sorted({c["row_id"] for k in ("same_no_and_amount_other_gstin", "same_no_other_gstin_amount_differs",
                                               "same_gstin_and_no_amount_gap", "found_in_next_period",
                                               "fuzzy_candidates") for c in sig.get(k, [])})
        items.append(OpenItem(invoice_id=inv.invoice_id, kind=kind, signals=sig, suggested_cause=cause,
                              candidates=cands))
    items.sort(key=lambda o: o.invoice_id)
    return items
