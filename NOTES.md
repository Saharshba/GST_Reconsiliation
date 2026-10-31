# Zero Balance: build notes

## M0: tool-calling spike (BLOCKED: Gemma 4 endpoint not responding, 2026-10-31)

Attempt log (key valid; `GET /v1/models` returns 200 in ~3 s and lists `google/gemma-4-31b-it`):
- LiteLLM's `nvidia_nim` provider needs the `mcp` package at call time (`No module named 'mcp'`). Added to
  requirements.txt.
- ADK → LiteLLM → NIM, add-tool turn: `litellm.Timeout ... Error code: 504` after several minutes.
- Direct `POST /v1/chat/completions` (no tools, 20 max tokens): no response in 150 s.
- Same with `stream: true`: no response headers in 240 s.
- `google/gemma-3-12b-it` probe: 404 "Function not found for account" (not deployed for this key), so it doesn't
  tell us whether the stall is Gemma-4-specific.
Conclusion so far: the free Gemma 4 31B endpoint is queued/overloaded; tool calling is still **untested**.


Script: `spike/m0_tool_calling.py` (add tool, state write/read across turns, 10 quick calls for latency/429s).
Not run yet: no API key on this machine, and the team asked not to spend credits until the key is provided.
The agent is built so that both paths exist (see M4): `AGENT_MODE=tools` (LlmAgent drives the tools) and
`AGENT_MODE=driven` (Fallback A: Python drives parse → fetch → match, LLM only diagnoses with JSON output).

Environment verified:
- Python 3.12 venv via uv (system Python is 3.14).
- google-adk 2.13.0: `LlmAgent`, `LiteLlm`, `Runner(app_name=, agent=, session_service=)`,
  `InMemorySessionService.create_session` (async), `ToolContext` imports all match the CLAUDE.md sketch.
- litellm 1.106.6 lists `nvidia_nim` as a provider.

## Model switch: Gemma 4 → MiniLM (2026-10-31, team decision)

The Gemma 4 endpoint never answered (see M0), so the team switched to
**sentence-transformers/all-MiniLM-L6-v2**: local, free, works offline (`HF_HUB_OFFLINE=1` after the one-time
download). `MODEL_ID=local/minilm` is the default; Gemma remains selectable.

What changes, honestly:
- MiniLM is an **embedding model, not an LLM**. It cannot call tools or write text. The ADK tool sequence is
  driven by the deterministic policy in `ScriptedLlm`; MiniLM makes the **diagnosis decision** (cause +
  confidence) for each open item (`agent/minilm.py`, `MiniLmLlm`). Emails come from templates; the explanation
  is the strongest evidence line (extractive), so it never adds numbers.
- **Few-shot (default):** k-NN (k=5) over a labelled bank of 63 past open items from seeds 1–3
  (`agent/build_minilm_bank.py`); demo data is seed 42, held out. Result: 21/21 diagnoses correct. Caveat: the
  synthetic evidence lines use the same templates across seeds, so this is an easy test; real data would be
  harder.
- **Zero-shot:** cosine similarity of each evidence line to cause descriptions (the deck's slide-8 "Signal"
  column + paraphrases). Result: 13/21 standalone; through the full agent 55.6% diagnosis accuracy. It confuses
  amount typos with wrong GSTIN and period mismatches with vendor-not-filed. **Unsafe auto-clears still 0**:
  the gate escalates every disagreement with the rule evidence (escalation rate 76% vs 43%). This is a good
  slide for "the safety gate works even when the model is wrong".
- "What didn't work" is now **embedding-only matching** (match invoices to 2B rows by MiniLM similarity):
  87.1% precision, 12 wrong pairs that would clear silently (e.g. ₹82,280 matched to ₹91,730 at
  similarity 0.992). The LLM-only experiment for Gemma is kept but not run.
- Deck edits needed: slide 5 LLM → "all-MiniLM-L6-v2 (sentence-transformers, local)"; slides 2/4 "agentic AI"
  wording should say the model diagnoses and the agent loop is ADK with a deterministic tool policy; slide 10
  "what didn't work" → embedding-only matching (numbers below).

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

## Status (2026-10-31)

| Milestone | State |
|---|---|
| M0 spike | script ready, **not run** (needs key) |
| M1–M3 | done, tested offline |
| M4 agent + tools + gate + done check | done; verified end to end through ADK with the offline scripted model |
| M5 Streamlit (4 steps, live trace, replay, offline) | done; checked in the browser |
| M6 evaluator + metrics tab | done; offline numbers below are a pipeline check only |
| M7 LLM-only experiment | script ready (`--dry-run` works, ~1.6k tokens/client), **not run** (needs key) |
| Fallback A (`AGENT_MODE=driven`) | done, tested offline |
| M8 README / pinned requirements | done; replay runs still need recording with the live model |

To do once the NVIDIA key is in `.env` (all use credits):
1. `python spike/m0_tool_calling.py`: record here whether tool calls work, plus latency and 429 behaviour.
   If they don't, set `AGENT_MODE=driven`.
2. `python -m agent.runner --client c1` (then c2, c3 with `--pin`), then `python eval/evaluate.py --source saved --mode live`.
3. `python experiments/llm_only_matching.py --clients all`.
4. Deck: slide 5 LLM = "Gemma 4 31B IT (NVIDIA NIM)", framework = "Google ADK"; slide 10 "what didn't work" from step 3.

Things to watch in live runs: Gemma may skip validate_gstin, pass confidence as a string (coerced), or stop
early (the done check catches it and counts `agent_did_not_decide`). If the system prompt is rejected by the
endpoint, LiteLLM/NIM chat template handling is the first thing to check.

## M1–M3 results (offline)

- 3 clients, 31 / 33 / 35 invoices, 12 vendors each, intra- and inter-state, 5% and 18% slabs, 4 PDF layouts.
- Parsing: 100% of fields correct on the 96 readable PDFs (case-insensitive name); 3/3 unreadable flagged.
- Matching: every planted exact / rounding / date-drift / near-miss case found by its expected pass and row;
  open items equal the planted mismatches; rule-based `suggested_cause` is right for all 21 open items.
  Checked on seeds 42 (sample), 7, 2027 (in pytest) and 1, 2, 3, 4, 5, 11, 99, 123 (manual sweep).

<!-- EVAL-OFFLINE:START -->
### Agent pipeline, offline scripted model (rules-only baseline, NOT an LLM result)

The scripted stand-in always follows the rule-based suggested_cause. These numbers check the pipeline and the gate; they are not model numbers.

| Client | Match P | Match R | Diagnosis acc. | Action acc. | Unsafe auto-clears | Escalation rate | Min/client | LLM calls |
|---|---|---|---|---|---|---|---|---|
| c1 | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 43% | 0.01 | 13 |
| c2 | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 43% | 0.01 | 13 |
| c3 | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 43% | 0.01 | 13 |
| **all** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0** | **43%** | **0.01** | **13.0** |

Diagnosis accuracy per mismatch type:

| Cause | Correct | Total |
|---|---|---|
| amount_typo | 3 | 3 |
| period_mismatch | 3 | 3 |
| unclear | 3 | 3 |
| unreadable | 3 | 3 |
| vendor_not_filed | 3 | 3 |
| wrong_gstin | 3 | 3 |

Confusion (expected → predicted): amount_typo: amount_typo×3; period_mismatch: period_mismatch×3; unclear: unclear×3; unreadable: unreadable×3; vendor_not_filed: vendor_not_filed×3; wrong_gstin: wrong_gstin×3
<!-- EVAL-OFFLINE:END -->

<!-- EVAL-LOCAL-ZEROSHOT:START -->
### Agent with local MiniLM (all-MiniLM-L6-v2, zeroshot)

Diagnosis by sentence-embedding nearest neighbours over a labelled bank built from seeds 1-3; the demo data (seed 42) is held out. Caveat: synthetic evidence lines share templates across seeds, so this is an easy test. Model calls = deterministic policy steps, not LLM calls.

| Client | Match P | Match R | Diagnosis acc. | Action acc. | Unsafe auto-clears | Escalation rate | Min/client | LLM calls |
|---|---|---|---|---|---|---|---|---|
| c1 | 100.0% | 100.0% | 33.3% | 90.3% | 0 | 86% | 0.09 | 16 |
| c2 | 100.0% | 100.0% | 66.7% | 93.9% | 0 | 71% | 0.01 | 15 |
| c3 | 100.0% | 100.0% | 66.7% | 94.3% | 0 | 71% | 0.01 | 15 |
| **all** | **100.0%** | **100.0%** | **55.6%** | **92.9%** | **0** | **76%** | **0.04** | **15.3** |

Diagnosis accuracy per mismatch type:

| Cause | Correct | Total |
|---|---|---|
| amount_typo | 0 | 3 |
| period_mismatch | 0 | 3 |
| unclear | 2 | 3 |
| unreadable | 3 | 3 |
| vendor_not_filed | 3 | 3 |
| wrong_gstin | 2 | 3 |

Confusion (expected → predicted): amount_typo: wrong_gstin×3; period_mismatch: vendor_not_filed×3; unclear: amount_typo×1, unclear×2; unreadable: unreadable×3; vendor_not_filed: vendor_not_filed×3; wrong_gstin: amount_typo×1, wrong_gstin×2
<!-- EVAL-LOCAL-ZEROSHOT:END -->

<!-- EVAL-LOCAL-FEWSHOT:START -->
### Agent with local MiniLM (all-MiniLM-L6-v2, fewshot)

Diagnosis by sentence-embedding nearest neighbours over a labelled bank built from seeds 1-3; the demo data (seed 42) is held out. Caveat: synthetic evidence lines share templates across seeds, so this is an easy test. Model calls = deterministic policy steps, not LLM calls.

| Client | Match P | Match R | Diagnosis acc. | Action acc. | Unsafe auto-clears | Escalation rate | Min/client | LLM calls |
|---|---|---|---|---|---|---|---|---|
| c1 | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 43% | 0.1 | 13 |
| c2 | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 43% | 0.1 | 13 |
| c3 | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 43% | 0.1 | 13 |
| **all** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0** | **43%** | **0.1** | **13.0** |

Diagnosis accuracy per mismatch type:

| Cause | Correct | Total |
|---|---|---|
| amount_typo | 3 | 3 |
| period_mismatch | 3 | 3 |
| unclear | 3 | 3 |
| unreadable | 3 | 3 |
| vendor_not_filed | 3 | 3 |
| wrong_gstin | 3 | 3 |

Confusion (expected → predicted): amount_typo: amount_typo×3; period_mismatch: period_mismatch×3; unclear: unclear×3; unreadable: unreadable×3; vendor_not_filed: vendor_not_filed×3; wrong_gstin: wrong_gstin×3
<!-- EVAL-LOCAL-FEWSHOT:END -->

<!-- EMBEDDING-ONLY:START -->
### Embedding-only matching baseline (all-MiniLM-L6-v2, cosine ≥ 0.8, greedy one-to-one)

Match precision 87.1%, recall 100.0% (81/81 correct pairs, 12 wrong pairs that would have cleared silently). Deterministic passes on the same data: 100% / 100%.

Failure examples:
- c1-inv-028 (amount_typo): invoice `07XJTCM6618Y1ZU BC/1185/26-27 Rs 82280.1` matched to `09-004: 07XJTCM6618Y1ZU BC/1185/26-27 Rs 91730.1` (sim 0.992); expected None
- c1-inv-007 (wrong_gstin): invoice `33DBPCQ5908C1ZA 00473 Rs 66522.75` matched to `09-001: 07DBPCQ5908C3Z3 00473 Rs 66522.75` (sim 0.971); expected None
- c1-inv-010 (unclear): invoice `29QLUPA1204D1Z2 BT262701262 Rs 134558.55` matched to `09-029: 37QLUPA1204D2Z4 BT262701262 Rs 117065.94` (sim 0.958); expected None
- c1-inv-031 (period_mismatch): invoice `24DJIPD6280S1ZL INV/2026-27/0043 Rs 18490.5` matched to `09-009: 27ZIKFY3791P1ZX CD-1282 Rs 173294.8` (sim 0.864); expected None
<!-- EMBEDDING-ONLY:END -->
