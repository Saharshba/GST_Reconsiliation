"""Single-responsibility agents for the Zero Balance multi-agent system.

Every agent:
  - owns ONE job and touches only the MCP tools that job needs (least privilege),
  - is iterative: act -> observe -> decide whether to iterate, with an explicit stop condition and a cap,
  - talks to the data only through the MCP server, and to other agents only through small keys in ADK
    session state (a blackboard): no agent calls another agent directly.

Two kinds of iteration:
  - inside an agent (IterativeAgent.max_iterations): ingestion retries extraction strategies, matching runs
    one pass per iteration, evidence keeps verifying GSTINs, drafting/escalation drain their queues;
  - between agents (Workflow routes in workflow.py): Diagnosis <-> Critic until every proposal is accepted
    or out of attempts, and Escalation <-> Audit until no open item is undecided.
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, ClassVar

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.tools import ToolContext
from google.adk.tools.mcp_tool import McpToolset
from google.genai import types
from pydantic import ConfigDict, PrivateAttr

from zero_balance.parsing import STRATEGIES

EMAIL_CAUSES = ("wrong_gstin", "vendor_not_filed", "amount_typo", "period_mismatch")
# Diagnosis strategy per attempt. Zero-shot needs no labelled history (only cause definitions); when the critic
# rejects it, the next attempt consults precedent (few-shot nearest neighbours over past labelled cases).
DIAG_STRATEGIES = tuple(os.environ.get("DIAG_STRATEGIES", "zeroshot,fewshot").split(","))


class McpAgent(BaseAgent):
    """Base: an ADK agent whose only way to act on data is the MCP toolset it is given."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    toolset: McpToolset
    allowed_tools: ClassVar[tuple[str, ...]] = ()
    _tools: dict = PrivateAttr(default_factory=dict)
    _buf: list = PrivateAttr(default_factory=list)

    async def call(self, ctx, name: str, **args) -> dict:
        if name not in self.allowed_tools:  # least privilege: an agent can't reach outside its job
            raise PermissionError(f"{self.name} may not call {name}")
        if not self._tools:
            self._tools = {t.name: t for t in await self.toolset.get_tools()}
        self._buf.append(self._fc(name, args))
        raw = await self._tools[name].run_async(args=args, tool_context=ToolContext(ctx))
        text = "".join(c.get("text", "") for c in raw.get("content", []) if isinstance(c, dict))
        try:
            out = json.loads(text) if text else (raw.get("structuredContent") or {})
        except json.JSONDecodeError:
            out = {"error": text}
        if raw.get("isError"):
            out = {"error": text}
        self._buf.append(self._fr(name, out))
        return out

    def _fc(self, name, args) -> Event:
        return Event(author=self.name, content=types.Content(
            role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]))

    def _fr(self, name, out) -> Event:
        return Event(author=self.name, content=types.Content(
            role="user", parts=[types.Part(function_response=types.FunctionResponse(name=name, response=out))]))

    def say(self, text: str, **state) -> Event:
        return Event(author=self.name, message=text, state=state or None)

    def flush(self) -> list[Event]:
        out, self._buf = self._buf, []
        return out

    async def run_async(self, *args, **kwargs):
        # Workflow nodes may be shallow copies of one agent object; a fresh buffer per invocation stops
        # one round's buffered tool events leaking into the next round's trace.
        self._buf = []
        async for ev in super().run_async(*args, **kwargs):
            yield ev


class IterativeAgent(McpAgent):
    """act -> observe -> decide. Subclasses implement iteration(); it sets self._done when converged."""

    max_iterations: int = 4
    _done: bool = PrivateAttr(default=False)
    _summary: str = PrivateAttr(default="")

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        self._done, self._summary = False, ""
        for i in range(self.max_iterations):
            async for ev in self.iteration(ctx, i):
                yield ev
            for ev in self.flush():
                yield ev
            if self._done:
                break
        else:
            yield self.say(f"stopped at the iteration cap ({self.max_iterations})")
        if self._summary:
            yield self.say(self._summary)

    async def iteration(self, ctx, i: int) -> AsyncGenerator[Event, None]:
        raise NotImplementedError
        yield  # pragma: no cover


# ----------------------------------------------------------------------------------------------------------------

class IngestionAgent(IterativeAgent):
    """Job: get clean, structured inputs. Iterates extraction strategies on files that failed to parse."""

    allowed_tools: ClassVar = ("start_run", "fetch_2b_rows", "parse_invoice")
    max_iterations: int = len(STRATEGIES)

    async def iteration(self, ctx, i):
        st = ctx.session.state
        if i == 0:
            start = await self.call(ctx, "start_run", client_id=st["client_id"])
            for p in (start["period"], start["next_period"]):
                await self.call(ctx, "fetch_2b_rows", run_id=start["run_id"], period=p)
            res = await self.call(ctx, "parse_invoice", run_id=start["run_id"], strategy=STRATEGIES[0])
            for ev in self.flush():
                yield ev
            yield self.say(f"iteration 1: parsed {res['files']} PDFs with '{STRATEGIES[0]}', "
                           f"{len(res['failed'])} failed", run_id=start["run_id"], client=start["client"],
                           failed_files=res["failed"])
        else:
            failed = st.get("failed_files", [])
            res = await self.call(ctx, "parse_invoice", run_id=st["run_id"], strategy=STRATEGIES[i], only_failed=True)
            for ev in self.flush():
                yield ev
            yield self.say(f"iteration {i + 1}: retried {len(failed)} failed file(s) with '{STRATEGIES[i]}': "
                           f"{len(res['improved'])} recovered", failed_files=res["failed"])
            failed = res["failed"]
            if i == self.max_iterations - 1 and failed:
                self._summary = (f"{len(failed)} file(s) still unreadable after {len(STRATEGIES)} strategies "
                                 f"({', '.join(failed)}): flagged, never guessed")
            self._done = not failed
            return
        self._done = not res["failed"]


class MatchingAgent(IterativeAgent):
    """Job: pair invoices with 2B rows. One deterministic pass per iteration, strongest key first."""

    allowed_tools: ClassVar = ("match_records",)
    max_iterations: int = 4

    async def iteration(self, ctx, i):
        res = await self.call(ctx, "match_records", run_id=ctx.session.state["run_id"], pass_no=i + 1)
        for ev in self.flush():
            yield ev
        yield self.say(f"pass {i + 1}: +{res['new_matches']} matches, {res['unmatched_invoices']} invoices left")
        self._done = res["unmatched_invoices"] == 0
        if i == self.max_iterations - 1 or self._done:
            self._summary = f"{res['matched_total']} matched; {res['unmatched_invoices']} invoices go to evidence"


class EvidenceAgent(IterativeAgent):
    """Job: assemble facts for each open item. Iterates until every GSTIN in the evidence is verified."""

    allowed_tools: ClassVar = ("build_evidence", "validate_gstin", "record_gstin_check")
    max_iterations: int = 3

    async def iteration(self, ctx, i):
        st = ctx.session.state
        if i == 0:
            ev = await self.call(ctx, "build_evidence", run_id=st["run_id"])
            if "error" in ev:
                raise RuntimeError(f"evidence incomplete: {ev['error']}")
            items = ev["open_items"]
            pending = sorted({(o["invoice_id"], g) for o in items for g in o["gstins_to_verify"]})
            for e in self.flush():
                yield e
            yield self.say(f"{len(items)} open items; {len(pending)} GSTIN(s) to verify",
                           open_items=items, gstins_pending=[list(p) for p in pending])
            self._done = not pending
            return
        pending = st.get("gstins_pending", [])
        checked = []
        for iid, g in pending:
            v = await self.call(ctx, "validate_gstin", gstin=g)
            await self.call(ctx, "record_gstin_check", run_id=st["run_id"], invoice_id=iid, gstin=g,
                            valid=v["valid"], reason=v["reason"])
            checked.append(f"{g} {'valid' if v['valid'] else 'INVALID'}")
        for e in self.flush():
            yield e
        items = [dict(o) for o in st["open_items"]]
        for o in items:  # kept beside the facts, not inside them, so the diagnoser's input is unchanged
            o["gstin_checks"] = [c for c in checked if c.split()[0] in o["gstins_to_verify"]]
        yield self.say("verified: " + "; ".join(checked), open_items=items, gstins_pending=[])
        self._done = True
        self._summary = f"evidence ready for {len(items)} open items"

    async def _run_async_impl(self, ctx):
        async for ev in super()._run_async_impl(ctx):
            yield ev
        items = ctx.session.state.get("open_items", [])
        yield self.say("handing open items to diagnosis", worklist=[o["invoice_id"] for o in items],
                       attempts={o["invoice_id"]: 0 for o in items}, proposals={}, accepted_action=[],
                       accepted_escalate=[], to_escalate={}, diag_round=0)


class DiagnosisAgent(McpAgent):
    """Job: propose cause + confidence for each item on the worklist. The only agent that uses a learned
    model (MiniLM). Strategy escalates with each attempt the critic rejects."""

    allowed_tools: ClassVar = ()  # reads evidence from state; needs no data tools at all

    async def _run_async_impl(self, ctx):
        from agent import minilm
        st = ctx.session.state
        proposals, rnd = {}, st.get("diag_round", 0) + 1
        by_id = {o["invoice_id"]: o for o in st["open_items"]}
        for iid in st["worklist"]:
            n = st["attempts"][iid]
            strategy = DIAG_STRATEGIES[min(n, len(DIAG_STRATEGIES) - 1)]
            fn = minilm.diagnose_fewshot if strategy == "fewshot" else minilm.diagnose_zeroshot
            cause, conf, expl = fn(by_id[iid])
            proposals[iid] = {"cause": cause, "confidence": round(conf, 3), "explanation": expl, "strategy": strategy}
            yield self.say(f"round {rnd} · {iid} [{strategy}]: {cause} ({conf:.2f})")
        yield self.say(f"round {rnd}: {len(proposals)} proposal(s) for the critic", proposals=proposals,
                       diag_round=rnd, model_calls=st.get("model_calls", 0) + len(proposals))


class CriticAgent(McpAgent):
    """Job: verify each proposal against the deterministic evidence (via the server's read-only check) and
    decide: accept, send back for another attempt, or give up and escalate. Routes the loop."""

    allowed_tools: ClassVar = ("check_diagnosis",)

    async def _run_async_impl(self, ctx):
        st = ctx.session.state
        attempts = dict(st["attempts"])
        acc_a, acc_e = list(st["accepted_action"]), list(st["accepted_escalate"])
        to_esc, retry, final = dict(st["to_escalate"]), [], dict(st.get("final_proposals", {}))
        for iid, p in st["proposals"].items():
            v = await self.call(ctx, "check_diagnosis", run_id=st["run_id"], invoice_id=iid,
                                cause=p["cause"], confidence=p["confidence"])
            for ev in self.flush():
                yield ev
            attempts[iid] += 1
            final[iid] = p
            if v["verdict"] == "accept_action":
                acc_a.append(iid)
                note = "accepted → drafting"
            elif v["verdict"] == "accept_escalate":
                acc_e.append(iid)
                note = "accepted → human review"
            elif v["retryable"] and attempts[iid] < len(DIAG_STRATEGIES):
                retry.append(iid)
                note = f"rejected ({v['reason']}) → retry with {DIAG_STRATEGIES[attempts[iid]]}"
            else:
                to_esc[iid] = f"diagnosis rejected after {attempts[iid]} attempt(s): {v['reason']}"
                note = "rejected, no attempts left → escalate"
            yield self.say(f"{iid}: {p['cause']} ({p['confidence']:.2f}) {note}")
        route = "retry" if retry else "done"
        yield Event(author=self.name, message=f"{len(retry)} to retry" if retry else "all proposals settled",
                    state={"attempts": attempts, "accepted_action": acc_a, "accepted_escalate": acc_e,
                           "to_escalate": to_esc, "worklist": retry, "proposals": {}, "final_proposals": final},
                    route=route)


class DraftingAgent(IterativeAgent):
    """Job: turn accepted diagnoses into drafts (or next-return flags). Drains its queue; anything the
    server's gate refuses is handed to escalation instead of being retried here."""

    allowed_tools: ClassVar = ("draft_email",)
    max_iterations: int = 2

    async def iteration(self, ctx, i):
        st = ctx.session.state
        queue = [x for x in st["accepted_action"] if x not in st.get("drafted", [])]
        drafted, refused = list(st.get("drafted", [])), dict(st["to_escalate"])
        for iid in queue:
            p = st["final_proposals"][iid]
            r = await self.call(ctx, "draft_email", run_id=st["run_id"], invoice_id=iid, cause=p["cause"],
                                confidence=p["confidence"], explanation=p["explanation"])
            for ev in self.flush():
                yield ev
            if r.get("status") in ("drafted", "flagged"):
                drafted.append(iid)
            else:
                refused[iid] = f"draft refused by gate: {r.get('reason')}"
        yield self.say(f"queue {len(queue)}: {len(drafted)} drafted/flagged, "
                       f"{len(refused) - len(st['to_escalate'])} refused", drafted=drafted, to_escalate=refused)
        self._done = True


class EscalationAgent(IterativeAgent):
    """Job: hand every item that needs a human to the reviewer queue, with evidence and the agent's note."""

    allowed_tools: ClassVar = ("escalate_case",)
    max_iterations: int = 2

    async def iteration(self, ctx, i):
        st = ctx.session.state
        done = set(st.get("escalated", []))
        queue = {iid: "agreed: needs a human" for iid in st["accepted_escalate"]} | dict(st["to_escalate"])
        queue = {k: v for k, v in queue.items() if k not in done}
        for iid, reason in queue.items():
            p = st.get("final_proposals", {}).get(iid, {})
            await self.call(ctx, "escalate_case", run_id=st["run_id"], invoice_id=iid, reason=reason,
                            confidence=p.get("confidence", 0.0), explanation=p.get("explanation", ""),
                            cause=p.get("cause", ""))
            for ev in self.flush():
                yield ev
            done.add(iid)
        yield self.say(f"escalated {len(queue)} item(s)", escalated=sorted(done))
        self._done = True


class AuditAgent(McpAgent):
    """Job: completeness. Every open item must be drafted, flagged or escalated. Anything missing is sent
    back to escalation (loop); when nothing is missing, the run is closed."""

    allowed_tools: ClassVar = ("get_status",)

    async def _run_async_impl(self, ctx):
        st = ctx.session.state
        s = await self.call(ctx, "get_status", run_id=st["run_id"])
        for ev in self.flush():
            yield ev
        missing = s["undecided"]
        n = st.get("audit_rounds", 0) + 1
        if missing and n < 3:
            to_esc = dict(st["to_escalate"]) | {i: "agent_did_not_decide" for i in missing}
            yield Event(author=self.name, message=f"audit {n}: {len(missing)} undecided → escalation",
                        state={"to_escalate": to_esc, "audit_rounds": n}, route="again")
        else:
            yield Event(author=self.name, message=f"audit {n}: complete · {s['matched']} matched · "
                        f"decisions {s['decisions']}", state={"audit_rounds": n, "status": s}, route="done")


class ReporterAgent(McpAgent):
    """Job: close the run with a one-paragraph summary for the junior accountant."""

    allowed_tools: ClassVar = ()

    async def _run_async_impl(self, ctx):
        st = ctx.session.state
        s, d = st["status"], st["status"]["decisions"]
        yield self.say(f"Matched {s['matched']} invoices. {len(st['open_items'])} open items: "
                       f"{d.get('drafted', 0)} vendor emails drafted, {d.get('flagged', 0)} flagged for the next "
                       f"return, {d.get('escalated', 0)} escalated to the reviewer. Diagnosis took "
                       f"{st.get('diag_round', 0)} round(s) of propose → critique.")


AGENT_ROLES: dict[str, str] = {
    "ingestion": "Load both 2B periods and parse PDFs; retry failed files with other extraction strategies",
    "matching": "Run the 4 deterministic passes, one per iteration, until nothing is left or passes run out",
    "evidence": "Compute facts per open item; verify every other-GSTIN in the evidence",
    "diagnosis": "Propose cause + confidence per item with MiniLM; stronger strategy on each retry",
    "critic": "Check proposals against the evidence; accept, retry, or give up; routes the loop",
    "drafting": "Draft vendor emails / next-return flags for accepted items",
    "escalation": "Send items that need a human to the reviewer queue with evidence",
    "audit": "Verify every open item is decided; loop back to escalation if not",
    "reporter": "Write the run summary",
}


def dump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)
