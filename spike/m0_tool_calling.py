"""M0 spike: does Gemma 4 tool calling survive NVIDIA NIM -> LiteLLM -> ADK?

Run: .venv/bin/python spike/m0_tool_calling.py
"""
import asyncio
import os
import time

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

load_dotenv()
MODEL_ID = os.environ.get("MODEL_ID", "nvidia_nim/google/gemma-4-31b-it")


def add(a: int, b: int) -> dict:
    """Add two integers and return the sum."""
    return {"sum": a + b}


def remember(key: str, value: str, tool_context: ToolContext) -> dict:
    """Store a value in session state under the given key."""
    tool_context.state[key] = value
    return {"stored": key}


def recall(key: str, tool_context: ToolContext) -> dict:
    """Read a value previously stored in session state."""
    return {"value": tool_context.state.get(key)}


async def turn(runner, session_id, text):
    calls, final = [], ""
    t0 = time.time()
    async for ev in runner.run_async(
        user_id="u", session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        for fc in ev.get_function_calls():
            calls.append((fc.name, dict(fc.args or {})))
        if ev.is_final_response() and ev.content and ev.content.parts:
            final = "".join(p.text or "" for p in ev.content.parts)
    return calls, final, time.time() - t0


async def main():
    agent = LlmAgent(
        name="spike", model=LiteLlm(model=MODEL_ID),
        instruction="You are a test agent. Always use the tools when asked.",
        tools=[add, remember, recall],
    )
    svc = InMemorySessionService()
    runner = Runner(app_name="spike", agent=agent, session_service=svc)
    s = await svc.create_session(app_name="spike", user_id="u")

    print("== 1. tool call")
    print(await turn(runner, s.id, "What is 2+3? Use the add tool."))
    print("== 2. state write, then read in a later turn")
    print(await turn(runner, s.id, "Use remember to store key 'client' with value 'c1'."))
    print(await turn(runner, s.id, "Use recall to read key 'client' and tell me the value."))
    s2 = await svc.get_session(app_name="spike", user_id="u", session_id=s.id)
    print("state:", s2.state)

    print("== 3. latency / rate limit (10 quick calls)")
    import litellm
    lat, errs = [], []
    for i in range(10):
        t0 = time.time()
        try:
            await litellm.acompletion(model=MODEL_ID, messages=[{"role": "user", "content": f"Say {i}"}], max_tokens=5)
            lat.append(time.time() - t0)
        except Exception as e:  # noqa: BLE001
            errs.append(type(e).__name__ + ": " + str(e)[:120])
    print("latencies:", [round(x, 2) for x in lat])
    print("errors:", errs)


if __name__ == "__main__":
    asyncio.run(main())
