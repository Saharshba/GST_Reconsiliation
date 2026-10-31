"""The agent graph (ADK Workflow).

    START → ingestion → matching → evidence → diagnosis → critic ─┬─ "retry" → diagnosis   (critique loop)
                                                                  └─ "done"  → drafting → escalation → audit
                                                         audit ─┬─ "again" → escalation                 (audit loop)
                                                                └─ "done"  → reporter
"""
from __future__ import annotations

from pathlib import Path

from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.adk.workflow import START, Workflow
from mcp import StdioServerParameters

from .agents import (AuditAgent, CriticAgent, DiagnosisAgent, DraftingAgent, EscalationAgent, EvidenceAgent,
                     IngestionAgent, MatchingAgent, ReporterAgent)

ROOT = Path(__file__).resolve().parents[1]


def mcp_toolset() -> McpToolset:
    """One stdio connection to the Zero Balance MCP server, shared by all agents of a run."""
    import sys
    return McpToolset(connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT)),
        timeout=120))


def build_workflow(toolset: McpToolset) -> Workflow:
    a = dict(toolset=toolset)
    ingest, match, evidence = IngestionAgent(name="ingestion", **a), MatchingAgent(name="matching", **a), \
        EvidenceAgent(name="evidence", **a)
    diagnose, critic = DiagnosisAgent(name="diagnosis", **a), CriticAgent(name="critic", **a)
    draft, escalate, audit = DraftingAgent(name="drafting", **a), EscalationAgent(name="escalation", **a), \
        AuditAgent(name="audit", **a)
    report = ReporterAgent(name="reporter", **a)
    return Workflow(name="zero_balance_team", edges=[
        (START, ingest, match, evidence, diagnose, critic),
        (critic, {"retry": diagnose, "done": draft}),
        (draft, escalate, audit),
        (audit, {"again": escalate, "done": report}),
    ])
