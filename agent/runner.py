"""Run the agent for one client and capture the trace.

    python -m agent.runner --client c1                       # live (MODEL_ID from .env)
    python -m agent.runner --client c1 --model offline/scripted   # no LLM, no network
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from zero_balance import normalise as N
from zero_balance.schema import RunResult, TraceEvent

from . import tools as T
from .agent import build_agent, model_id
from .driven import run_driven
from .models import TRACE_HOOK
from .prompts import user_message

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
APP = "zero_balance"


def _short(obj, limit: int = 2500):
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return obj if len(s) <= limit else s[:limit] + " …"


def expected_open_items(client_id: str) -> int:
    scratch: dict = {}
    client = T.load_clients()[client_id]
    T.op_fetch_2b_rows(scratch, client_id, client["period"])
    T.op_fetch_2b_rows(scratch, client_id, N.next_period(client["period"]))
    T.op_parse_invoice(scratch, client_id)
    return len(T.op_match_records(scratch, client_id)["open_items"])


async def run_client_async(client_id: str, mid: str | None = None,
                           on_event: Optional[Callable[[TraceEvent], None]] = None,
                           save: bool = True, agent_mode: str | None = None) -> RunResult:
    mid = mid or model_id()
    client = T.load_clients()[client_id]
    period = client["period"]
    t0 = time.time()
    trace: list[TraceEvent] = []
    llm_calls = 0

    def emit(kind: str, name: str = "", data=None):
        ev = TraceEvent(t=round(time.time() - t0, 2), kind=kind, name=name, data=data)
        trace.append(ev)
        if on_event:
            on_event(ev)

    def count_call(callback_context, llm_request):
        nonlocal llm_calls
        llm_calls += 1
        return None

    agent_mode = agent_mode or os.environ.get("AGENT_MODE", "tools")
    msg = user_message(client, period, N.next_period(period))
    emit("user", "task", msg)
    token = TRACE_HOOK.set(lambda kind, text: emit(kind, "llm", text))
    summary = ""
    if agent_mode == "driven":  # Fallback A: Python drives, the LLM only diagnoses
        emit("info", "config", {"model": mid, "agent_mode": "driven", "threshold": T.threshold()})
        run: dict = {}
        try:
            await run_driven(run, client_id, mid, emit, count_call)
        except Exception as e:  # noqa: BLE001
            emit("error", type(e).__name__, str(e)[:500])
        finally:
            TRACE_HOOK.reset(token)
    else:
        n_open = expected_open_items(client_id)
        cap = 6 + 2 * n_open
        agent = build_agent(mid, before_model_callback=count_call)
        svc = InMemorySessionService()
        runner = Runner(app_name=APP, agent=agent, session_service=svc)
        session = await svc.create_session(app_name=APP, user_id="junior", state={"run": {}})
        emit("info", "config", {"model": mid, "agent_mode": "tools", "max_llm_calls": cap,
                                "threshold": T.threshold()})
        try:
            async for ev in runner.run_async(
                user_id="junior", session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=msg)]),
                run_config=RunConfig(max_llm_calls=cap),
            ):
                if ev.partial or not ev.content:
                    continue
                for p in ev.content.parts or []:
                    if p.function_call:
                        emit("tool_call", p.function_call.name, dict(p.function_call.args or {}))
                    elif p.function_response:
                        emit("tool_result", p.function_response.name,
                             _short(dict(p.function_response.response or {})))
                    elif p.text and ev.author == agent.name and not getattr(p, "thought", False):
                        emit("llm_text", "", p.text.strip())
                        summary = p.text.strip()
        except Exception as e:  # noqa: BLE001  (rate limits after retries, call cap, bad model output)
            emit("error", type(e).__name__, str(e)[:500])
        finally:
            TRACE_HOOK.reset(token)
        session = await svc.get_session(app_name=APP, user_id="junior", session_id=session.id)
        run = json.loads(json.dumps(session.state.get("run") or {}))

    # If the agent died before matching, finish the deterministic steps in Python so nothing is lost.
    if "open_items" not in run:
        emit("warning", "fallback", "agent stopped before match_records; running parse/fetch/match in Python")
        T.op_fetch_2b_rows(run, client_id, period)
        T.op_fetch_2b_rows(run, client_id, N.next_period(period))
        T.op_parse_invoice(run, client_id)
        T.op_match_records(run, client_id)

    # Done check: every open item must be drafted, flagged or escalated.
    for o in run["open_items"]:
        if o["invoice_id"] not in run.get("decided", {}):
            T.op_escalate_case(run, o["invoice_id"], "agent_did_not_decide", 0.0,
                               "The agent stopped without deciding this item.", auto=True)
            emit("warning", "done_check", f"{o['invoice_id']} auto-escalated: agent_did_not_decide")

    finished = time.time()
    result = RunResult(
        run_id=f"{client_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}",
        client_id=client_id,
        mode="offline" if mid.startswith("offline/") else "local" if mid.startswith("local/") else "live", model=mid,
        started_at=datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        finished_at=datetime.fromtimestamp(finished).isoformat(timespec="seconds"),
        duration_s=round(finished - t0, 2),
        invoices=run.get("invoices", []), rows=run.get("rows", []), matches=run.get("matches", []),
        open_items=run.get("open_items", []), diagnoses=run.get("diagnoses", []), emails=run.get("emails", []),
        flags=run.get("flags", []), escalations=run.get("escalations", []),
        trace=trace, llm_calls=llm_calls, summary=summary,
    )
    if save:
        RUNS_DIR.mkdir(exist_ok=True)
        (RUNS_DIR / f"{result.run_id}.json").write_text(result.model_dump_json(indent=1))
    return result


def run_client(client_id: str, mid: str | None = None, on_event=None, save: bool = True,
               agent_mode: str | None = None) -> RunResult:
    return asyncio.run(run_client_async(client_id, mid, on_event, save, agent_mode))


def latest_run(client_id: str, mode: str | None = None) -> RunResult | None:
    """Latest saved run; replay files (runs/replay_<client>.json) win when present."""
    pinned = RUNS_DIR / f"replay_{client_id}.json"
    if pinned.exists():
        r = RunResult.model_validate_json(pinned.read_text())
        if mode is None or r.mode == mode:
            return r
    files = sorted(RUNS_DIR.glob(f"{client_id}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        r = RunResult.model_validate_json(f.read_text())
        if mode is None or r.mode == mode:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True)
    ap.add_argument("--model", default=None, help="override MODEL_ID, e.g. offline/scripted")
    ap.add_argument("--agent-mode", choices=["tools", "driven"], default=None,
                    help="tools = LLM drives the tools (default); driven = Fallback A")
    ap.add_argument("--pin", action="store_true", help="also save as runs/replay_<client>.json for demo replay")
    args = ap.parse_args()

    def show(ev: TraceEvent):
        d = ev.data if isinstance(ev.data, str) else json.dumps(ev.data, default=str)
        print(f"[{ev.t:6.1f}s] {ev.kind:<11} {ev.name:<15} {d[:160]}")

    r = run_client(args.client, args.model, on_event=show, agent_mode=args.agent_mode)
    if args.pin:
        (RUNS_DIR / f"replay_{args.client}.json").write_text(r.model_dump_json(indent=1))
        print(f"pinned as runs/replay_{args.client}.json")
    print(f"\nmatches={len(r.matches)} emails={len(r.emails)} flags={len(r.flags)} "
          f"escalations={len(r.escalations)} llm_calls={r.llm_calls} {r.duration_s}s -> runs/{r.run_id}.json")


if __name__ == "__main__":
    main()
