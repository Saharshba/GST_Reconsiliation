"""Zero Balance MCP server: the reconciliation tools, exposed over the Model Context Protocol (stdio).

The server owns ALL data and ALL side effects (drafts, flags, escalations). Agents are clients: they get short
summaries back and never hold the records themselves. The safety gate lives here, so no agent (and no other
MCP client, e.g. Claude Desktop or MCP Inspector) can draft an email the evidence doesn't support.

Run state is kept per run_id and persisted to runs/mcp/<run_id>.json after every write, so the server can be
restarted mid-run and any client can resume or audit a run.

    python -m mcp_server.server          # stdio transport (what the agents use)
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import tools as T  # noqa: E402  (pure ops on a run dict; no model calls)
from zero_balance import gstr2b, matching, signals  # noqa: E402
from zero_balance import normalise as N  # noqa: E402
from zero_balance.parsing import STRATEGIES, parse_pdf  # noqa: E402
from zero_balance.schema import Invoice, Match, OpenItem, TwoBRow  # noqa: E402

STORE = ROOT / "runs" / "mcp"
mcp = MCPServer("zero-balance")
READ_ONLY = ToolAnnotations(readOnlyHint=True)
_runs: dict[str, dict] = {}


def _load(run_id: str) -> dict:
    if run_id not in _runs:
        f = STORE / f"{run_id}.json"
        if not f.exists():
            raise ValueError(f"unknown run_id {run_id!r}; call start_run first")
        _runs[run_id] = json.loads(f.read_text())
    return _runs[run_id]


def _save(run_id: str, run: dict) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    _runs[run_id] = run
    (STORE / f"{run_id}.json").write_text(json.dumps(run, default=str))


# --------------------------------------------------------------------------------------------- ingestion

@mcp.tool()
def start_run(client_id: str) -> dict:
    """Open a reconciliation run for one client. Returns run_id, client details, filing and next period."""
    run: dict = {}
    client = T._client(run, client_id)
    run_id = f"{client_id}-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
    run["run_id"] = run_id
    _save(run_id, run)
    return {"run_id": run_id, "client": run["client"], "period": client["period"],
            "next_period": N.next_period(client["period"])}


@mcp.tool()
def fetch_2b_rows(run_id: str, period: str) -> dict:
    """Load the client's GSTR-2B for a period (MMYYYY): the filing period or the next one."""
    run = _load(run_id)
    out = T.op_fetch_2b_rows(run, run["client_id"], period)
    _save(run_id, run)
    return out


@mcp.tool()
def parse_invoice(run_id: str, strategy: str = "default", only_failed: bool = False) -> dict:
    """Parse the client's invoice PDFs with an extraction strategy (default | layout | words).
    only_failed=True re-parses just the files that are still partial/unreadable and keeps any improvement."""
    if strategy not in STRATEGIES:
        return {"error": f"strategy must be one of {STRATEGIES}"}
    run = _load(run_id)
    client = T.load_clients()[run["client_id"]]
    files = sorted((T.data_dir() / run["client_id"] / "invoices").glob("*.pdf"))
    prev = {i["invoice_id"]: i for i in run.get("invoices", [])}
    rank = {"ok": 2, "partial": 1, "unreadable": 0}
    improved = []
    for f in files:
        if only_failed and prev.get(f.stem, {}).get("parse_status") == "ok":
            continue
        inv = parse_pdf(f, client_id=run["client_id"], client_gstin=client["gstin"], strategy=strategy)
        old = prev.get(f.stem)
        if old is None or rank[inv.parse_status] > rank[old["parse_status"]]:
            if old is not None:
                improved.append(f.name)
            prev[f.stem] = inv.model_dump(mode="json")
    run["invoices"] = [prev[k] for k in sorted(prev)]
    run.setdefault("parse_attempts", []).append(strategy)
    _save(run_id, run)
    counts: dict[str, int] = {}
    for i in run["invoices"]:
        counts[i["parse_status"]] = counts.get(i["parse_status"], 0) + 1
    return {"strategy": strategy, "files": len(run["invoices"]), "by_status": counts, "improved": improved,
            "failed": [i["source_file"] for i in run["invoices"] if i["parse_status"] != "ok"]}


# --------------------------------------------------------------------------------------------- matching

@mcp.tool()
def match_records(run_id: str, pass_no: int) -> dict:
    """Run ONE deterministic matching pass (1 exact, 2 normalised, 3 date window, 4 fuzzy) on what is still
    unmatched. Returns the matches this pass found and how many invoices remain."""
    run = _load(run_id)
    invs = [Invoice(**i) for i in run["invoices"]]
    rows = [TwoBRow(**r) for r in run["rows"]]
    prior = [Match(**m) for m in run.get("matches", [])]
    mr = matching.match(invs, rows, run["period"], passes=(pass_no,), prior=prior)
    new = mr.matches[len(prior):]
    run["matches"] = [m.model_dump(mode="json") for m in mr.matches]
    run["unmatched_rows"] = [r.row_id for r in mr.unmatched_rows]
    run.setdefault("passes_run", []).append(pass_no)
    _save(run_id, run)
    return {"pass": pass_no, "new_matches": len(new),
            "examples": [f"{m.invoice_id}->{m.row_id} ({m.notes})" for m in new if m.pass_no > 1][:3],
            "matched_total": len(mr.matches), "unmatched_invoices": len(mr.unmatched_invoices),
            "unmatched_2b_rows": len(mr.unmatched_rows)}


# --------------------------------------------------------------------------------------------- evidence

@mcp.tool()
def build_evidence(run_id: str) -> dict:
    """Compute the facts for every open item (unmatched invoices + low-confidence fuzzy matches) and the
    rule-based suggested cause. Facts only: no conclusions beyond the rule suggestion."""
    run = _load(run_id)
    missing = [k for k, what in (("rows", "filing-period 2B"), ("next_rows", "next-period 2B"),
                                 ("invoices", "parsed invoices"), ("matches", "matching passes")) if k not in run]
    if missing:  # incomplete evidence would silently weaken the diagnosis, so refuse instead
        return {"error": "missing: " + ", ".join(missing)}
    invs = [Invoice(**i) for i in run["invoices"]]
    rows = [TwoBRow(**r) for r in run["rows"]]
    nxt = [TwoBRow(**r) for r in run["next_rows"]]
    matches = [Match(**m) for m in run["matches"]]
    matched_rows = {m.row_id for m in matches}
    matched_inv = {m.invoice_id for m in matches}
    mr = matching.MatchResult(matches=matches,
                              unmatched_invoices=[i for i in invs if i.invoice_id not in matched_inv],
                              unmatched_rows=[r for r in rows if r.row_id not in matched_rows])
    items = signals.build_open_items(invs, rows, nxt, mr)
    run["open_items"] = [o.model_dump(mode="json") for o in items]
    _save(run_id, run)
    by_id = {i.invoice_id: i for i in invs}
    return {"open_items": [{
        "invoice_id": o.invoice_id, "kind": o.kind,
        "supplier": run["vendors"].get(by_id[o.invoice_id].supplier_gstin or "", by_id[o.invoice_id].supplier_name),
        "amount": by_id[o.invoice_id].total, "suggested_cause": o.suggested_cause, "facts": T._fmt_fact(o.signals),
        "gstins_to_verify": [r["gstin"] for k in ("same_no_and_amount_other_gstin", "same_no_other_gstin_amount_differs")
                             for r in o.signals.get(k, [])],
    } for o in items]}


@mcp.tool(annotations=READ_ONLY)
def validate_gstin(gstin: str) -> dict:
    """Check a GSTIN's format and checksum. Returns valid, state_code and reason."""
    return T.op_validate_gstin(gstin)


@mcp.tool()
def record_gstin_check(run_id: str, invoice_id: str, gstin: str, valid: bool, reason: str) -> dict:
    """Attach a GSTIN validation result to an open item's evidence."""
    run = _load(run_id)
    for o in run.get("open_items", []):
        if o["invoice_id"] == invoice_id:
            o["signals"].setdefault("gstin_checks", {})[gstin] = {"valid": valid, "reason": reason}
            _save(run_id, run)
            return {"invoice_id": invoice_id, "recorded": gstin}
    return {"error": f"{invoice_id} is not an open item"}


# --------------------------------------------------------------------------------------------- verification

@mcp.tool(annotations=READ_ONLY)
def check_diagnosis(run_id: str, invoice_id: str, cause: str, confidence: float) -> dict:
    """Read-only safety check of a proposed diagnosis against the deterministic evidence.
    verdict: accept_action (email/flag allowed) | accept_escalate (agreed, but must go to a human) | reject.
    retryable tells the critic whether another diagnosis attempt could change the outcome."""
    run = _load(run_id)
    item = T._open_item(run, invoice_id)
    if item is None:
        return {"verdict": "reject", "retryable": False, "reason": "not an open item"}
    if invoice_id in run.get("decided", {}):
        return {"verdict": "reject", "retryable": False, "reason": f"already {run['decided'][invoice_id]}"}
    agrees = cause == item.suggested_cause
    if not agrees:
        if item.suggested_cause == "unreadable":
            return {"verdict": "reject", "retryable": False, "reason": "PDF unreadable: only a human can resolve"}
        return {"verdict": "reject", "retryable": True,
                "reason": f"'{cause}' disagrees with the evidence (rules suggest '{item.suggested_cause}')"}
    if cause in ("unclear", "unreadable") or item.kind == "fuzzy_match_candidate":
        return {"verdict": "accept_escalate", "retryable": False, "reason": f"{cause}: needs a human"}
    if float(confidence) < T.threshold():
        return {"verdict": "reject", "retryable": True,
                "reason": f"confidence {float(confidence):.2f} below threshold {T.threshold():.2f}"}
    return {"verdict": "accept_action", "retryable": False, "reason": "cause matches evidence, confidence ok"}


# --------------------------------------------------------------------------------------------- actions

@mcp.tool()
def draft_email(run_id: str, invoice_id: str, cause: str, confidence: float, explanation: str) -> dict:
    """Draft a vendor follow-up (never sent), or for period_mismatch record a flag for the next return.
    The safety gate runs again here; a refused draft must be escalated."""
    run = _load(run_id)
    out = T.op_draft_email(run, invoice_id, cause, confidence, explanation)
    _save(run_id, run)
    return out


@mcp.tool()
def escalate_case(run_id: str, invoice_id: str, reason: str, confidence: float, explanation: str,
                  cause: str = "") -> dict:
    """Send an open item to the human reviewer with its evidence and the agent's note."""
    run = _load(run_id)
    out = T.op_escalate_case(run, invoice_id, reason, confidence, explanation, cause=cause or None)
    _save(run_id, run)
    return out


# --------------------------------------------------------------------------------------------- audit

@mcp.tool(annotations=READ_ONLY)
def get_status(run_id: str) -> dict:
    """Progress of a run: matches, open items, decisions, and any open item still undecided."""
    run = _load(run_id)
    decided = run.get("decided", {})
    open_ids = [o["invoice_id"] for o in run.get("open_items", [])]
    counts: dict[str, int] = {}
    for v in decided.values():
        counts[v] = counts.get(v, 0) + 1
    return {"matched": len(run.get("matches", [])), "open_items": len(open_ids), "decisions": counts,
            "undecided": [i for i in open_ids if i not in decided]}


@mcp.tool(annotations=READ_ONLY)
def export_run(run_id: str) -> dict:
    """Full run record (invoices, rows, matches, open items, diagnoses, emails, flags, escalations)."""
    return _load(run_id)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
