# Zero Balance: build notes

## M0: tool-calling spike (PENDING: needs NVIDIA_NIM_API_KEY)

Script: `spike/m0_tool_calling.py` (add tool, state write/read across turns, 10 quick calls for latency/429s).
Not run yet: no API key on this machine, and the team asked not to spend credits until the key is provided.
The agent is built so that both paths exist (see M4): `AGENT_MODE=tools` (LlmAgent drives the tools) and
`AGENT_MODE=driven` (Fallback A: Python drives parse → fetch → match, LLM only diagnoses with JSON output).

Environment verified:
- Python 3.12 venv via uv (system Python is 3.14).
- google-adk 2.13.0: `LlmAgent`, `LiteLlm`, `Runner(app_name=, agent=, session_service=)`,
  `InMemorySessionService.create_session` (async), `ToolContext` imports all match the CLAUDE.md sketch.
- litellm 1.106.6 lists `nvidia_nim` as a provider.

## Decisions / corrections to CLAUDE.md

- **Pass 3 ambiguity rule.** A date-window match on (GSTIN, amount) is only accepted when that vendor billed that
  amount once (on both the invoice side and the 2B side) and exactly one unmatched row is a candidate. Without
  this, the near-miss case (same vendor, same amount billed twice, one number mistyped) is swallowed by pass 3
  and never reaches pass 4. Also requires |invoice date − 2B date| ≤ 5 days.
- **Near-miss handling.** Pass-4 matches are recorded as matches (confidence 0.6) *and* raised as open items of
  kind `fuzzy_match_candidate` with `suggested_cause = "unclear"`, so the gate escalates them for human
  confirmation. Answer key accepts `none` or `escalate`.
- **Unreadable PDFs**: c1 and c3 are image-only scans (no text layer), c2 has a garbled text layer. The vendor
  did file these invoices, so their 2B rows stay unmatched on the 2B side.
- **Unclear variants** (one per client): c1 = same number under another GSTIN with a different amount (no rule
  fires); c2 = wrong-GSTIN row + copy in next period (two rules fire); c3 = amount gap + copy in next period.
- Supplier name accuracy is compared case-insensitively: the two-column layout prints the name in capitals.

## M1–M3 results (offline)

- 3 clients, 31 / 33 / 35 invoices, 12 vendors each, intra- and inter-state, 5% and 18% slabs, 4 PDF layouts.
- Parsing: 100% of fields correct on the 96 readable PDFs (case-insensitive name); 3/3 unreadable flagged.
- Matching: every planted exact / rounding / date-drift / near-miss case found by its expected pass and row;
  open items equal the planted mismatches; rule-based `suggested_cause` is right for all 21 open items.
  Checked on seeds 42 (sample), 7, 2027 (in pytest) and 1, 2, 3, 4, 5, 11, 99, 123 (manual sweep).
