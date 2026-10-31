"""Fallback A (CLAUDE.md §3): Python drives parse → fetch → match, then for each open item an ADK LlmAgent
(no tools) returns a JSON diagnosis, and Python routes it through the same gated draft_email / escalate_case.
Use when tool calling does not survive NVIDIA NIM → LiteLLM → ADK.  Select with AGENT_MODE=driven.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import ValidationError

from zero_balance import normalise as N
from zero_balance.schema import Diagnosis

from . import tools as T
from .agent import build_model
from .prompts import DIAGNOSE_PROMPT

APP = "zero_balance_driven"


def _parse(text: str, invoice_id: str) -> Diagnosis | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        d["invoice_id"] = invoice_id
        return Diagnosis(**d)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


async def run_driven(run: dict, client_id: str, mid: str, emit: Callable, count_call) -> None:
    period = T.load_clients()[client_id]["period"]
    steps = [("fetch_2b_rows", T.op_fetch_2b_rows, (client_id, period), {"client_id": client_id, "period": period}),
             ("fetch_2b_rows", T.op_fetch_2b_rows, (client_id, N.next_period(period)),
              {"client_id": client_id, "period": N.next_period(period)}),
             ("parse_invoice", T.op_parse_invoice, (client_id,), {"client_id": client_id}),
             ("match_records", T.op_match_records, (client_id,), {"client_id": client_id})]
    match_out = None
    for name, fn, args, shown in steps:
        emit("tool_call", name, shown)
        out = fn(run, *args)
        emit("tool_result", name, out)
        match_out = out
    diagnoser = LlmAgent(name="diagnoser", model=build_model(mid), instruction=DIAGNOSE_PROMPT,
                         before_model_callback=count_call)
    svc = InMemorySessionService()
    runner = Runner(app_name=APP, agent=diagnoser, session_service=svc)
    for item in match_out["open_items"]:
        iid = item["invoice_id"]
        session = await svc.create_session(app_name=APP, user_id="junior")
        msg = "Evidence for one open item:\n" + json.dumps(item, ensure_ascii=False)
        text = ""
        async for ev in runner.run_async(user_id="junior", session_id=session.id,
                                         new_message=types.Content(role="user", parts=[types.Part(text=msg)])):
            if ev.content and not ev.partial:
                text += "".join(p.text or "" for p in ev.content.parts or [])
        d = _parse(text, iid)
        if d is None:
            emit("warning", "diagnoser", f"{iid}: model output was not valid JSON; escalating")
            args = {"invoice_id": iid, "reason": "unparseable model output", "confidence": 0.0, "explanation": text[:200]}
            emit("tool_call", "escalate_case", args)
            emit("tool_result", "escalate_case", T.op_escalate_case(run, **args))
            continue
        emit("llm_text", "", f"{iid}: {d.cause} ({d.confidence:.2f}) - {d.explanation}")
        if d.cause in ("wrong_gstin", "vendor_not_filed", "amount_typo", "period_mismatch"):
            args = {"invoice_id": iid, "cause": d.cause, "confidence": d.confidence, "explanation": d.explanation}
            emit("tool_call", "draft_email", args)
            out = T.op_draft_email(run, **args)
            emit("tool_result", "draft_email", out)
            if out["status"] != "refused":
                continue
            reason = f"draft refused: {out['reason']}"
        else:
            reason = f"{d.cause}: needs a human"
        args = {"invoice_id": iid, "reason": reason, "confidence": d.confidence, "explanation": d.explanation}
        if not any(x["invoice_id"] == iid for x in run.get("diagnoses", [])):
            T._record_diagnosis(run, iid, d.cause, d.confidence, d.explanation)
        emit("tool_call", "escalate_case", args)
        emit("tool_result", "escalate_case", T.op_escalate_case(run, **args))
