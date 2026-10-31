"""Run the multi-agent team for one client.

    python -m multiagent.runner --client c1 [--pin]
    DIAG_STRATEGIES=fewshot python -m multiagent.runner --client c1     # single strategy, no retries
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from zero_balance.schema import RunResult, TraceEvent

from .agents import DIAG_STRATEGIES
from .workflow import build_workflow, mcp_toolset

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
APP = "zero_balance_team"


async def run_client_async(client_id: str, on_event: Optional[Callable[[TraceEvent], None]] = None,
                           save: bool = True) -> RunResult:
    t0 = time.time()
    trace: list[TraceEvent] = []

    def emit(kind, name="", data=None, agent=""):
        ev = TraceEvent(t=round(time.time() - t0, 2), kind=kind, name=name, data=data, agent=agent)
        trace.append(ev)
        if on_event:
            on_event(ev)

    toolset = mcp_toolset()
    wf = build_workflow(toolset)
    svc = InMemorySessionService()
    runner = Runner(app_name=APP, node=wf, session_service=svc)
    session = await svc.create_session(app_name=APP, user_id="junior", state={"client_id": client_id})
    msg = f"Reconcile client {client_id} (multi-agent, MCP tools, diagnosis strategies {list(DIAG_STRATEGIES)})."
    emit("user", "task", msg)
    try:
        async for ev in runner.run_async(user_id="junior", session_id=session.id,
                                         new_message=types.Content(role="user", parts=[types.Part(text=msg)])):
            if not ev.content or ev.author == "user":
                continue
            for p in ev.content.parts or []:
                if p.function_call:
                    emit("tool_call", p.function_call.name, dict(p.function_call.args or {}), ev.author)
                elif p.function_response:
                    emit("tool_result", p.function_response.name, dict(p.function_response.response or {}), ev.author)
                elif p.text:
                    emit("llm_text", ev.author, p.text, ev.author)
    except Exception as e:  # noqa: BLE001
        emit("error", type(e).__name__, str(e)[:500])
    finally:
        await toolset.close()

    session = await svc.get_session(app_name=APP, user_id="junior", session_id=session.id)
    st = session.state
    run = json.loads((RUNS_DIR / "mcp" / f"{st['run_id']}.json").read_text())  # the server's record of the run
    finished = time.time()
    result = RunResult(
        run_id=f"{client_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}",
        client_id=client_id, mode="multiagent", model="local/minilm (" + "→".join(DIAG_STRATEGIES) + ")",
        started_at=datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        finished_at=datetime.fromtimestamp(finished).isoformat(timespec="seconds"),
        duration_s=round(finished - t0, 2),
        invoices=run.get("invoices", []), rows=run.get("rows", []), matches=run.get("matches", []),
        open_items=run.get("open_items", []), diagnoses=run.get("diagnoses", []), emails=run.get("emails", []),
        flags=run.get("flags", []), escalations=run.get("escalations", []), trace=trace,
        llm_calls=st.get("model_calls", 0),
        summary=next((e.data for e in reversed(trace) if e.agent == "reporter"), ""),
    )
    if save:
        RUNS_DIR.mkdir(exist_ok=True)
        (RUNS_DIR / f"{result.run_id}.json").write_text(result.model_dump_json(indent=1))
    return result


def run_client(client_id: str, on_event=None, save: bool = True) -> RunResult:
    return asyncio.run(run_client_async(client_id, on_event, save))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True)
    ap.add_argument("--pin", action="store_true", help="also save as runs/replay_multi_<client>.json")
    args = ap.parse_args()

    def show(ev: TraceEvent):
        d = ev.data if isinstance(ev.data, str) else json.dumps(ev.data, default=str)
        who = f"[{ev.agent}]" if ev.agent else ""
        print(f"{ev.t:5.1f}s {who:<13} {ev.kind:<11} {(ev.name if ev.kind != 'llm_text' else ''):<16} {d[:150]}")

    r = run_client(args.client, on_event=show)
    if args.pin:
        (RUNS_DIR / f"replay_multi_{args.client}.json").write_text(r.model_dump_json(indent=1))
    print(f"\nmatches={len(r.matches)} emails={len(r.emails)} flags={len(r.flags)} escalations={len(r.escalations)} "
          f"model_calls={r.llm_calls} {r.duration_s}s -> runs/{r.run_id}.json")


if __name__ == "__main__":
    main()
