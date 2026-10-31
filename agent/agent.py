from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent

from .models import RetryingLiteLlm, ScriptedLlm
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
DEFAULT_MODEL = "nvidia_nim/google/gemma-4-31b-it"


def model_id() -> str:
    return os.environ.get("MODEL_ID", DEFAULT_MODEL)


def build_model(mid: str | None = None):
    mid = mid or model_id()
    if mid.startswith("offline/"):
        return ScriptedLlm(model=mid)
    return RetryingLiteLlm(model=mid, temperature=0.1)


def build_agent(mid: str | None = None, **callbacks) -> LlmAgent:
    return LlmAgent(
        name="zero_balance",
        model=build_model(mid),
        description="Reconciles purchase invoices against GSTR-2B, drafts vendor follow-ups, escalates unclear cases.",
        instruction=SYSTEM_PROMPT,
        tools=TOOLS,
        **callbacks,
    )


root_agent = build_agent()
