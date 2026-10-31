"""Model backends.

- RetryingLiteLlm: Gemma 4 on NVIDIA NIM through LiteLLM, with backoff on 429/5xx.
- ScriptedLlm ("offline/scripted"): a NO-LLM stand-in that emits the tool calls a model would, following the
  rule-based suggested_cause. It exists so the whole ADK pipeline, UI and evaluator can be tested without
  credits. Its numbers are a rules-only baseline, never to be reported as model results.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import re
from typing import AsyncGenerator, Callable, ClassVar, Optional

from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

# The runner sets this to push "waiting on rate limit" notes into the live trace.
TRACE_HOOK: contextvars.ContextVar[Optional[Callable[[str, str], None]]] = contextvars.ContextVar(
    "TRACE_HOOK", default=None)

RETRY_DELAYS = (4, 10, 25)


def _retryable(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}"
    return any(k in s for k in ("429", "RateLimit", "rate limit", "500", "502", "503", "504",
                                "ServiceUnavailable", "InternalServerError", "Timeout", "APIConnectionError"))


class RetryingLiteLlm(LiteLlm):
    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                async for r in super().generate_content_async(llm_request, stream=False):
                    yield r
                return
            except Exception as e:  # noqa: BLE001
                if attempt == len(RETRY_DELAYS) or not _retryable(e):
                    raise
                delay = RETRY_DELAYS[attempt]
                hook = TRACE_HOOK.get()
                if hook:
                    hook("warning", f"waiting on rate limit / server error ({type(e).__name__}); "
                                    f"retry {attempt + 1}/{len(RETRY_DELAYS)} in {delay}s")
                await asyncio.sleep(delay)


class ScriptedLlm(BaseLlm):
    """Deterministic offline stand-in. Reads the conversation and emits the next tool call."""

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"offline/.*"]

    LABEL: ClassVar[str] = "offline scripted"

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        yield self._next(llm_request.contents or [])

    def diagnose(self, item: dict) -> tuple[str, float, str]:
        """(cause, confidence, explanation) for one open item. The scripted stand-in copies the rule suggestion."""
        cause = item["suggested_cause"]
        conf = 0.9 if cause not in ("unclear", "unreadable") else 0.4
        facts = item.get("facts", [])
        return cause, conf, "Evidence: " + ("; ".join(facts) if cause in ("unclear", "unreadable")
                                            else (facts[0] if facts else ""))[:200]

    # -- policy ---------------------------------------------------------------------------------
    @staticmethod
    def _reply(text: str | None = None, call: tuple[str, dict] | None = None) -> LlmResponse:
        parts = []
        if text:
            parts.append(types.Part(text=text))
        if call:
            parts.append(types.Part(function_call=types.FunctionCall(name=call[0], args=call[1])))
        return LlmResponse(content=types.Content(role="model", parts=parts),
                           usage_metadata=types.GenerateContentResponseUsageMetadata(
                               prompt_token_count=0, candidates_token_count=0, total_token_count=0))

    def _next(self, contents: list[types.Content]) -> LlmResponse:
        user_text = " ".join(p.text or "" for c in contents if c.role == "user" for p in (c.parts or []) if p.text)
        if user_text.startswith("Evidence for one open item"):  # driven mode: answer with a JSON diagnosis
            item = json.loads(user_text.split("\n", 1)[1])
            cause, conf, expl = self.diagnose(item)
            return self._reply(json.dumps({"invoice_id": item["invoice_id"], "cause": cause,
                                           "confidence": round(conf, 3), "explanation": expl}))
        m = re.search(r"client (\S+) .*?period (\d{6}).*?next period is (\d{6})", user_text, re.S)
        if not m:
            return self._reply("I need a client id and period to start.")
        cid, period, nxt = m.groups()
        responses: list[tuple[str, dict, dict]] = []
        calls: list[tuple[str, dict]] = []
        for c in contents:
            for p in c.parts or []:
                if p.function_call:
                    calls.append((p.function_call.name, dict(p.function_call.args or {})))
                if p.function_response:
                    args = calls[len(responses)][1] if len(responses) < len(calls) else {}
                    responses.append((p.function_response.name, args, dict(p.function_response.response or {})))
        done = {(n, a.get("period")) for n, a, _ in responses}
        if ("fetch_2b_rows", period) not in done:
            return self._reply(f"Plan: load 2B for {period} and {nxt}, parse invoices, match, then decide each "
                               f"open item.", ("fetch_2b_rows", {"client_id": cid, "period": period}))
        if ("fetch_2b_rows", nxt) not in done:
            return self._reply(None, ("fetch_2b_rows", {"client_id": cid, "period": nxt}))
        if not any(n == "parse_invoice" for n, _, _ in responses):
            return self._reply(None, ("parse_invoice", {"client_id": cid}))
        match = next((r for n, _, r in reversed(responses) if n == "match_records"), None)
        if match is None:
            return self._reply(None, ("match_records", {"client_id": cid}))

        decided, refused, validated = set(), {}, set()
        for n, a, r in responses:
            if n in ("draft_email", "escalate_case") and r.get("status") in ("drafted", "flagged", "escalated"):
                decided.add(r.get("invoice_id"))
            if n == "draft_email" and r.get("status") == "refused":
                refused[r.get("invoice_id")] = r.get("reason", "")
            if n == "validate_gstin":
                validated.add(a.get("gstin"))
        items = match.get("open_items", [])
        for it in items:
            iid = it["invoice_id"]
            if iid in decided:
                continue
            facts = it.get("facts", [])
            if iid in refused:
                return self._reply(None, ("escalate_case", {
                    "invoice_id": iid, "reason": f"draft refused: {refused[iid]}", "confidence": 0.5,
                    "explanation": "; ".join(facts)[:200]}))
            cause, conf, expl = self.diagnose(it)
            if cause == "wrong_gstin":
                other = next((g for f in facts for g in re.findall(r"OTHER GSTIN (\w{15})", f)), None)
                if other and other not in validated:
                    return self._reply(None, ("validate_gstin", {"gstin": other}))
            note = f"{iid}: diagnosed {cause} (confidence {conf:.2f})."
            if cause in ("unclear", "unreadable"):
                return self._reply(note, ("escalate_case", {
                    "invoice_id": iid, "reason": f"{cause}: needs a human", "confidence": round(conf, 3),
                    "explanation": expl}))
            return self._reply(note, ("draft_email", {
                "invoice_id": iid, "cause": cause, "confidence": round(conf, 3), "explanation": expl}))
        n_email = sum(1 for n, _, r in responses if n == "draft_email" and r.get("status") == "drafted")
        n_flag = sum(1 for n, _, r in responses if n == "draft_email" and r.get("status") == "flagged")
        n_esc = sum(1 for n, _, r in responses if n == "escalate_case" and r.get("status") == "escalated")
        matched = sum(match.get("matched_by_pass", {}).values())
        return self._reply(f"[{self.LABEL}] Matched {matched} invoices; drafted {n_email} emails, "
                           f"flagged {n_flag} for the next return and escalated {n_esc} to the reviewer.")


def dump_args(args: dict) -> str:
    return json.dumps(args, ensure_ascii=False)
