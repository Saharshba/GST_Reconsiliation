"""Multi-agent system tests: MCP server contract, least privilege, iteration, and safety. Offline (MiniLM must
be cached locally; the MCP server runs as a local stdio subprocess)."""
import asyncio
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="module")
def minilm():
    try:
        from agent.minilm import _model
        _model()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"MiniLM not cached locally: {e}")


def test_mcp_server_exposes_deck_tools_and_read_only_hints():
    from mcp_server.server import mcp

    async def go():
        return await mcp.list_tools()
    tools = {t.name: t for t in asyncio.run(go())}
    for name in ("parse_invoice", "fetch_2b_rows", "match_records", "validate_gstin", "draft_email", "escalate_case"):
        assert name in tools
    for name in ("validate_gstin", "check_diagnosis", "get_status", "export_run"):
        assert tools[name].annotations and tools[name].annotations.read_only_hint


def test_server_gate_rejects_and_flags_retryable():
    from mcp_server import server as S
    rid = S.start_run("c1")["run_id"]
    for p in ("092026", "102026"):
        S.fetch_2b_rows(rid, p)
    S.parse_invoice(rid)
    assert "error" in S.build_evidence(rid)  # refuses before matching has run
    for n in (1, 2, 3, 4):
        S.match_records(rid, n)
    items = {o["invoice_id"]: o for o in S.build_evidence(rid)["open_items"]}
    typo = next(i for i, o in items.items() if o["suggested_cause"] == "amount_typo")
    unreadable = next(i for i, o in items.items() if o["suggested_cause"] == "unreadable")
    disagree = S.check_diagnosis(rid, typo, "wrong_gstin", 0.99)
    assert disagree["verdict"] == "reject" and disagree["retryable"] is True
    assert S.check_diagnosis(rid, typo, "amount_typo", 0.4)["retryable"] is True
    assert S.check_diagnosis(rid, typo, "amount_typo", 0.9)["verdict"] == "accept_action"
    assert S.check_diagnosis(rid, unreadable, "amount_typo", 0.9)["retryable"] is False
    assert S.check_diagnosis(rid, unreadable, "unreadable", 0.9)["verdict"] == "accept_escalate"


def test_agents_have_least_privilege():
    from multiagent.agents import (AuditAgent, CriticAgent, DiagnosisAgent, DraftingAgent, EscalationAgent,
                                   EvidenceAgent, IngestionAgent, MatchingAgent)
    assert DiagnosisAgent.allowed_tools == ()
    assert CriticAgent.allowed_tools == ("check_diagnosis",)
    assert "draft_email" not in EscalationAgent.allowed_tools and "escalate_case" not in DraftingAgent.allowed_tools
    every = set().union(*(a.allowed_tools for a in (IngestionAgent, MatchingAgent, EvidenceAgent, CriticAgent,
                                                    DraftingAgent, EscalationAgent, AuditAgent)))
    assert "export_run" not in every  # only the runner reads the full record


def test_team_run_iterates_and_is_safe(minilm):
    from eval.evaluate import load_key, prediction_from_run, score
    from multiagent.runner import run_client
    r = run_client("c1", save=False)
    s = score(load_key("c1"), prediction_from_run(r))
    assert s["unsafe_auto_clears"] == 0 and s["match"]["precision"] == 1.0 and s["match"]["recall"] == 1.0
    decided = {e.invoice_id for e in r.emails} | {f.invoice_id for f in r.flags} | {e.invoice_id for e in r.escalations}
    assert decided == {o.invoice_id for o in r.open_items}
    agents = {e.agent for e in r.trace if e.agent}
    assert agents == {"ingestion", "matching", "evidence", "diagnosis", "critic", "drafting", "escalation",
                      "audit", "reporter"}
    # the critique loop ran more than once (zero-shot proposals were rejected and retried with few-shot)
    assert any(isinstance(e.data, str) and e.data.startswith("round 2") for e in r.trace if e.agent == "diagnosis")
    # each critic check happens once per proposal: no duplicated (phantom) tool events across rounds
    checks = [(e.data["invoice_id"], e.t) for e in r.trace if e.kind == "tool_call" and e.name == "check_diagnosis"]
    assert len(checks) == len(r.open_items) + sum(
        1 for e in r.trace if e.agent == "critic" and isinstance(e.data, str) and "→ retry" in e.data)
