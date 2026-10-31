"""The 6 agent tools. Thin wrappers over zero_balance/; full records live in session state["run"],
the LLM only sees short summaries. The safety gate (CLAUDE.md §7.3) is enforced here, not in the prompt."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from google.adk.tools import ToolContext

from zero_balance import gstin as G
from zero_balance import gstr2b, matching, signals
from zero_balance import normalise as N
from zero_balance.parsing import parse_client
from zero_balance.schema import (CAUSES, Diagnosis, EmailDraft, Escalation, Flag, Invoice, OpenItem, TwoBRow)

ROOT = Path(__file__).resolve().parents[1]
EMAIL_CAUSES = ("wrong_gstin", "vendor_not_filed", "amount_typo")


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", ROOT / "data" / "sample"))


def threshold() -> float:
    return float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))


def load_clients() -> dict[str, dict]:
    return {c["client_id"]: c for c in json.loads((data_dir() / "clients.json").read_text())}


# --------------------------------------------------------------------------- ops on a plain dict
# `run` is a JSON-serialisable dict; the ADK wrappers copy it in and out of tool_context.state.

def _client(run: dict, client_id: str) -> dict:
    clients = load_clients()
    if client_id not in clients:
        raise KeyError(f"unknown client_id {client_id!r}; known: {sorted(clients)}")
    if run.get("client_id") not in (None, client_id):
        raise ValueError(f"this session is reconciling {run['client_id']}, not {client_id}")
    run["client_id"] = client_id
    run["client"] = {k: clients[client_id][k] for k in ("client_id", "name", "gstin", "period")}
    run["vendors"] = {v["gstin"]: v["name"] for v in clients[client_id].get("vendors", [])}
    run.setdefault("period", clients[client_id]["period"])
    return clients[client_id]


def op_fetch_2b_rows(run: dict, client_id: str, period: str) -> dict:
    _client(run, client_id)
    path = gstr2b.path_for(data_dir() / client_id, period)
    if not path.exists():
        return {"error": f"no GSTR-2B export for period {period}"}
    rows = gstr2b.load(path)
    key = "rows" if period == run["period"] else "next_rows" if period == N.next_period(run["period"]) else None
    if key is None:
        return {"error": f"period must be {run['period']} (filing) or {N.next_period(run['period'])} (next)"}
    run[key] = [r.model_dump(mode="json") for r in rows]
    return {"period": period, "role": "filing period" if key == "rows" else "next period",
            "rows": len(rows), "suppliers": len({r.supplier_gstin for r in rows})}


def op_parse_invoice(run: dict, client_id: str) -> dict:
    client = _client(run, client_id)
    invs = parse_client(data_dir() / client_id, client_id=client_id, client_gstin=client["gstin"])
    run["invoices"] = [i.model_dump(mode="json") for i in invs]
    counts: dict[str, int] = {}
    for i in invs:
        counts[i.parse_status] = counts.get(i.parse_status, 0) + 1
    return {"files": len(invs), "by_status": counts,
            "unreadable": [i.source_file for i in invs if i.parse_status == "unreadable"],
            "partial": [{"file": i.source_file, "notes": i.parse_notes} for i in invs if i.parse_status == "partial"]}


def _fmt_fact(sig: dict) -> list[str]:
    """Compact, human-readable facts for the LLM (keeps the payload small)."""
    f: list[str] = []
    if sig.get("parse_status") == "unreadable":
        return [f"PDF unreadable: {sig.get('parse_notes', '')}"]
    if sig.get("parse_status") == "partial":
        f.append(f"parsed partially: {sig.get('parse_notes')}")
    if sig.get("gstin_valid") is False:
        f.append("supplier GSTIN on invoice fails checksum")
    if sig.get("vendor_has_rows_this_period") is False:
        f.append("vendor has NO rows in 2B this period")
    for r in sig.get("same_no_and_amount_other_gstin", []):
        f.append(f"same no.+amount in 2B under OTHER GSTIN {r['gstin']} (row {r['row_id']}, Rs {r['total']})")
    for r in sig.get("same_no_other_gstin_amount_differs", []):
        f.append(f"same no. under OTHER GSTIN {r['gstin']} but amount Rs {r['total']} (gap {r['gap']:+}) row {r['row_id']}")
    for r in sig.get("same_gstin_and_no_amount_gap", []):
        f.append(f"same GSTIN+no. in 2B row {r['row_id']} but amount Rs {r['row_total']} vs invoice "
                 f"Rs {r['invoice_total']} (gap {r['gap']:+})")
    for r in sig.get("found_in_next_period", []):
        f.append(f"found in NEXT period 2B row {r['row_id']} (amount {'matches' if r['amount_matches'] else 'differs'})")
    if sig.get("fuzzy_match_candidate"):
        fm = sig["fuzzy_match_candidate"]
        f.append(f"pass-4 fuzzy match to row {fm['row_id']}: {fm['notes']} - needs human confirmation")
    elif sig.get("fuzzy_candidates"):
        f.append("similar numbers: " + ", ".join(f"{c['invoice_no']} (row {c['row_id']}, score {c['score']})"
                                                 for c in sig["fuzzy_candidates"]))
    return f


def op_match_records(run: dict, client_id: str) -> dict:
    _client(run, client_id)
    notes = []
    if not run.get("rows"):
        op_fetch_2b_rows(run, client_id, run["period"])
        notes.append("auto-loaded filing-period 2B")
    if "next_rows" not in run:
        op_fetch_2b_rows(run, client_id, N.next_period(run["period"]))
        notes.append("auto-loaded next-period 2B")
    if not run.get("invoices"):
        op_parse_invoice(run, client_id)
        notes.append("auto-parsed invoices")
    invs = [Invoice(**i) for i in run["invoices"]]
    rows = [TwoBRow(**r) for r in run["rows"]]
    nxt = [TwoBRow(**r) for r in run.get("next_rows", [])]
    mr = matching.match(invs, rows, run["period"])
    items = signals.build_open_items(invs, rows, nxt, mr)
    run["matches"] = [m.model_dump(mode="json") for m in mr.matches]
    run["open_items"] = [o.model_dump(mode="json") for o in items]
    run["unmatched_rows"] = [r.row_id for r in mr.unmatched_rows]
    by_id = {i.invoice_id: i for i in invs}
    out_items = []
    for o in items:
        inv = by_id[o.invoice_id]
        out_items.append({
            "invoice_id": o.invoice_id,
            "supplier": run["vendors"].get(inv.supplier_gstin or "", inv.supplier_name) if inv.supplier_gstin else None,
            "gstin": inv.supplier_gstin, "invoice_no": inv.invoice_no, "amount": inv.total,
            "suggested_cause": o.suggested_cause, "facts": _fmt_fact(o.signals),
        })
    out = {"matched_by_pass": {f"pass{k}": v for k, v in mr.by_pass().items()},
           "invoices": len(invs), "open_items": out_items,
           "unmatched_2b_rows": len(mr.unmatched_rows)}
    if notes:
        out["notes"] = notes
    return out


def op_validate_gstin(gstin: str) -> dict:
    return {"gstin": gstin} | G.validate(gstin)


def _open_item(run: dict, invoice_id: str) -> OpenItem | None:
    for o in run.get("open_items", []):
        if o["invoice_id"] == invoice_id:
            return OpenItem(**o)
    return None


def _decided(run: dict) -> dict[str, str]:
    return run.setdefault("decided", {})


def _record_diagnosis(run: dict, invoice_id: str, cause: str, confidence: float, explanation: str):
    if cause in CAUSES:
        run.setdefault("diagnoses", []).append(Diagnosis(
            invoice_id=invoice_id, cause=cause, confidence=max(0.0, min(1.0, confidence)),
            explanation=explanation).model_dump(mode="json"))


def gate(item: OpenItem, cause: str, confidence: float) -> str | None:
    """Return a refusal reason, or None when an automatic email / flag is allowed."""
    if item.suggested_cause == "unreadable" or cause == "unreadable":
        return "unreadable PDF: must be escalated"
    if cause == "unclear":
        return "cause is unclear: must be escalated"
    if item.kind == "fuzzy_match_candidate":
        return "fuzzy match needs human confirmation: escalate"
    if cause != item.suggested_cause:
        return f"your cause '{cause}' disagrees with the rule-based evidence ('{item.suggested_cause}'): escalate"
    if confidence < threshold():
        return f"confidence {confidence:.2f} is below threshold {threshold():.2f}: escalate"
    return None


_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM_RE.findall(text or "") if len(n.replace(",", "")) >= 3}


def safe_explanation(explanation: str, allowed_text: str) -> tuple[str, bool]:
    """Drop the LLM sentence if it cites a figure that is not in the evidence."""
    allowed = _numbers(allowed_text)
    allowed |= {a.split(".")[0] for a in allowed} | {f"{float(a):.2f}" for a in allowed if a.replace(".", "").isdigit()}
    bad = [n for n in _numbers(explanation) if n not in allowed and n.split(".")[0] not in allowed]
    return (explanation, True) if not bad else ("", False)


def _compose_email(run: dict, item: OpenItem, cause: str, explanation: str) -> EmailDraft:
    inv = next(Invoice(**i) for i in run["invoices"] if i["invoice_id"] == item.invoice_id)
    client = run["client"]
    vendor = run["vendors"].get(inv.supplier_gstin, inv.supplier_name or "Vendor")
    sig = item.signals
    no, dt, amt = inv.invoice_no, inv.invoice_date.strftime("%d-%m-%Y") if inv.invoice_date else "-", N.fmt_inr(inv.total)
    period = run["period"]
    month = f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(period[:2]) - 1]} {period[2:]}"
    hl = [no, f"Rs {amt}"]
    if cause == "wrong_gstin":
        other = sig["same_no_and_amount_other_gstin"][0]["gstin"]
        subject = f"GSTIN correction needed: invoice {no} dated {dt}"
        problem = (f"In our GSTR-2B for {month}, invoice {no} dated {dt} for Rs {amt} is reported under "
                   f"GSTIN {other}, not under your GSTIN {inv.supplier_gstin} shown on the invoice.")
        fix = (f"Please amend this invoice in your GSTR-1 (Table 9A) so that it is reported under GSTIN "
               f"{inv.supplier_gstin}, with recipient GSTIN {client['gstin']}, or confirm in writing which "
               f"GSTIN is correct and issue a revised invoice.")
        hl += [other, inv.supplier_gstin]
    elif cause == "vendor_not_filed":
        subject = f"GSTR-1 for {month} not reflecting invoice {no}"
        problem = (f"Invoice {no} dated {dt} for Rs {amt} does not appear in our GSTR-2B for {month}; "
                   f"no invoices from GSTIN {inv.supplier_gstin} appear for that period.")
        fix = (f"Please file your GSTR-1 / IFF for {month} including this invoice against recipient GSTIN "
               f"{client['gstin']}, and share the filing acknowledgement. We cannot claim input tax credit "
               f"until it is reported.")
    elif cause == "amount_typo":
        g = sig["same_gstin_and_no_amount_gap"][0]
        row_amt = N.fmt_inr(g["row_total"])
        subject = f"Amount mismatch on invoice {no} dated {dt}"
        problem = (f"Invoice {no} dated {dt} shows a total of Rs {amt} in our books, but your GSTR-1 reports "
                   f"Rs {row_amt} for the same invoice (difference Rs {N.fmt_inr(abs(g['gap']))}).")
        fix = ("Please confirm which figure is correct. If the GSTR-1 figure is wrong, amend it in Table 9A of "
               "your next GSTR-1; if the invoice is wrong, issue a credit/debit note for the difference.")
        hl += [f"Rs {row_amt}"]
    else:
        raise ValueError(cause)
    hl.append(fix)
    lines = [f"Dear {vendor} team,", "",
             f"We are writing on behalf of our client {client['name']} (GSTIN {client['gstin']}) about their "
             f"purchase reconciliation for {month}.", "", problem]
    if explanation:
        lines += ["", explanation]
    lines += ["", f"Requested fix: {fix}", "",
              "Invoice details:",
              f"  Invoice no.: {no}", f"  Invoice date: {dt}", f"  Your GSTIN: {inv.supplier_gstin}",
              f"  Taxable value: Rs {N.fmt_inr(inv.taxable_value)}", f"  Invoice total: Rs {amt}", "",
              "Thank you for your help.", "", "Regards,", "Accounts team (on behalf of the client)",
              "[DRAFT - review before sending]"]
    return EmailDraft(invoice_id=inv.invoice_id, to_vendor=vendor, to_gstin=inv.supplier_gstin or "",
                      subject=subject, body="\n".join(lines), cause=cause, highlights=hl)


def op_draft_email(run: dict, invoice_id: str, cause: str, confidence: float, explanation: str) -> dict:
    confidence = float(confidence)
    item = _open_item(run, invoice_id)
    if item is None:
        return {"invoice_id": invoice_id, "status": "refused", "reason": "not an open item; run match_records first"}
    if invoice_id in _decided(run):
        return {"invoice_id": invoice_id, "status": "refused", "reason": f"already {_decided(run)[invoice_id]}"}
    _record_diagnosis(run, invoice_id, cause, confidence, explanation)
    refusal = gate(item, cause, confidence)
    if refusal:
        run.setdefault("refusals", []).append({"invoice_id": invoice_id, "cause": cause,
                                               "confidence": confidence, "reason": refusal})
        return {"invoice_id": invoice_id, "status": "refused", "reason": refusal, "next": "call escalate_case"}
    evidence_text = json.dumps(item.signals) + json.dumps(
        next(i for i in run["invoices"] if i["invoice_id"] == invoice_id))
    explanation, ok = safe_explanation(explanation, evidence_text)
    if cause == "period_mismatch":
        r = item.signals["found_in_next_period"][0]
        note = (f"Reported by the vendor in the {r['period']} GSTR-2B (row {r['row_id']}). Claim the ITC in the "
                f"{r['period']} return; no vendor action needed.")
        run.setdefault("flags", []).append(Flag(invoice_id=invoice_id, note=note).model_dump(mode="json"))
        _decided(run)[invoice_id] = "flagged"
        return {"invoice_id": invoice_id, "status": "flagged", "action": "flag for next return"}
    if cause not in EMAIL_CAUSES:
        return {"invoice_id": invoice_id, "status": "refused", "reason": f"no email template for {cause}"}
    email = _compose_email(run, item, cause, explanation)
    run.setdefault("emails", []).append(email.model_dump(mode="json"))
    _decided(run)[invoice_id] = "drafted"
    out = {"invoice_id": invoice_id, "status": "drafted", "to": email.to_vendor, "subject": email.subject}
    if not ok:
        out["note"] = "your explanation cited a figure not in the evidence, so it was left out of the email"
    return out


def op_escalate_case(run: dict, invoice_id: str, reason: str, confidence: float, explanation: str,
                     auto: bool = False, cause: str | None = None) -> dict:
    item = _open_item(run, invoice_id)
    if item is None:
        return {"invoice_id": invoice_id, "status": "refused", "reason": "not an open item"}
    if invoice_id in _decided(run):
        return {"invoice_id": invoice_id, "status": "refused", "reason": f"already {_decided(run)[invoice_id]}"}
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    llm_cause = next((d["cause"] for d in reversed(run.get("diagnoses", [])) if d["invoice_id"] == invoice_id), None)
    if not auto and llm_cause is None:  # escalated directly: use the cause given, else read it from the text
        m = re.search("|".join(CAUSES), f"{reason} {explanation}")
        llm_cause = cause if cause in CAUSES else (m.group(0) if m else "unclear")
        _record_diagnosis(run, invoice_id, llm_cause, confidence, explanation)
    run.setdefault("escalations", []).append(Escalation(
        invoice_id=invoice_id, reason=reason, evidence=item.signals | {"suggested_cause": item.suggested_cause},
        agent_note=explanation, llm_cause=llm_cause, llm_confidence=None if auto else confidence,
    ).model_dump(mode="json"))
    _decided(run)[invoice_id] = "escalated"
    return {"invoice_id": invoice_id, "status": "escalated"}


# --------------------------------------------------------------------------- ADK tool wrappers

def _run_state(tc: ToolContext) -> dict:
    return json.loads(json.dumps(tc.state.get("run") or {}))


def _call(tc: ToolContext, fn, *args) -> dict:
    run = _run_state(tc)
    try:
        out = fn(run, *args)
    except (KeyError, ValueError) as e:
        return {"error": str(e).strip("'\"")}
    tc.state["run"] = run
    return out


def fetch_2b_rows(client_id: str, period: str, tool_context: ToolContext) -> dict:
    """Load the client's GSTR-2B (B2B section) for a period given as MMYYYY, e.g. "092026".

    Call it for the filing period and for the next period. Returns row and supplier counts."""
    return _call(tool_context, op_fetch_2b_rows, client_id, period)


def parse_invoice(client_id: str, tool_context: ToolContext) -> dict:
    """Parse ALL purchase-invoice PDFs of the client (rule-based). Returns counts by parse status
    and the files that could not be read."""
    return _call(tool_context, op_parse_invoice, client_id)


def match_records(client_id: str, tool_context: ToolContext) -> dict:
    """Match the parsed invoices to the 2B rows in 4 deterministic passes and compute evidence for the
    invoices that stay open. Returns matches per pass and the open items with facts and a suggested cause."""
    return _call(tool_context, op_match_records, client_id)


def validate_gstin(gstin: str) -> dict:
    """Check a GSTIN's format and checksum. Returns valid, state_code and reason."""
    return op_validate_gstin(gstin)


def draft_email(invoice_id: str, cause: str, confidence: float, explanation: str,
                tool_context: ToolContext) -> dict:
    """Draft a vendor follow-up email for an open item (never sends it). cause is one of wrong_gstin,
    vendor_not_filed, amount_typo, period_mismatch (period_mismatch records a flag for the next return
    instead of an email). confidence is 0-1. explanation is one sentence citing the evidence.
    A safety gate may refuse; then call escalate_case."""
    return _call(tool_context, op_draft_email, invoice_id, cause, confidence, explanation)


def escalate_case(invoice_id: str, reason: str, confidence: float, explanation: str,
                  tool_context: ToolContext) -> dict:
    """Send an open item to the human reviewer with its evidence. Use when the cause is unclear,
    unreadable, low-confidence, or draft_email was refused."""
    return _call(tool_context, op_escalate_case, invoice_id, reason, confidence, explanation)


TOOLS = [parse_invoice, fetch_2b_rows, match_records, validate_gstin, draft_email, escalate_case]
