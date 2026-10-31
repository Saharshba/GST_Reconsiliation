# Zero Balance: GST reconciliation agent

Team Zero Balance · Agentathon 2026 (Encode.ai × PES University)

A Google ADK agent that reconciles one client's purchase invoices against their GSTR-2B. Parsing and matching are
rule-based Python. The model only **diagnoses** each mismatch: by default **all-MiniLM-L6-v2**
(sentence-transformers), running locally with no API or credits. It classifies each open item by nearest
neighbours over labelled past cases. Vendor follow-ups are drafted from templates filled with data. Gemma 4
(NVIDIA NIM) is still supported via `MODEL_ID`. Low-confidence or unclear cases go to a human reviewer with the evidence. Emails are drafts
only; nothing is ever sent.

```
invoice PDFs ─┐                ┌─ pass 1 exact ─┐
              ├─ parse_invoice │  pass 2 normal │      open items + evidence      ┌─ draft_email (gated)
GSTR-2B JSON ─┴─ fetch_2b_rows ┴─ pass 3 window ┴─ match_records ──► MiniLM  ──┤
                                  pass 4 fuzzy                 diagnose + conf.   └─ escalate_case
```

## Setup

```bash
brew install uv            # or: pipx install uv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env       # MODEL_ID=local/minilm needs no key; Gemma 4 needs NVIDIA_NIM_API_KEY
.venv/bin/python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"  # one-time download (~90 MB)
```

After the one-time download, set `HF_HUB_OFFLINE=1` in `.env` so nothing touches the network at demo time.

Sample data is committed in `data/sample/`. To regenerate it (deterministic):

```bash
.venv/bin/python data/generate.py --seed 42
```

## Run

| What | Command | Uses credits? |
|---|---|---|
| Tests (offline) | `.venv/bin/python -m pytest -q` | no |
| Agent, one client, MiniLM (default) | `.venv/bin/python -m agent.runner --client c1` | no |
| Rebuild MiniLM's labelled bank (seeds 1-3) | `.venv/bin/python -m agent.build_minilm_bank` | no |
| Embedding-only matching baseline ("what didn't work") | `.venv/bin/python experiments/embedding_only_matching.py` | no |
| M0 tool-calling spike | `.venv/bin/python spike/m0_tool_calling.py` | yes (~15 calls) |
| Agent, one client, Gemma 4 | `MODEL_ID=nvidia_nim/google/gemma-4-31b-it .venv/bin/python -m agent.runner --client c1` | yes |
| Agent, Fallback A (Python drives, model diagnoses; Gemma) | `.venv/bin/python -m agent.runner --client c1 --agent-mode driven` | yes |
| Agent, offline scripted stand-in (no LLM) | `.venv/bin/python -m agent.runner --client c1 --model offline/scripted` | no |
| Evaluate latest saved runs | `.venv/bin/python eval/evaluate.py --clients all --source saved` | no |
| LLM-only baseline (Gemma 4, needs key) | `.venv/bin/python experiments/llm_only_matching.py --clients all` | yes (3 calls) |
| Demo UI | `.venv/bin/streamlit run app/streamlit_app.py` | only in Live mode |

`AGENT_MODE=driven` in `.env` switches the UI and the runner to Fallback A.
The offline scripted model follows the rule-based suggested cause. Use it to test the pipeline. **Never
present its numbers as model results.**

## Demo script (≈3 minutes, matches deck slide 9)

Replay runs for c1–c3 are already pinned (`runs/replay_c*.json`). To re-record them:

```bash
for c in c1 c2 c3; do .venv/bin/python -m agent.runner --client $c --pin; done
.venv/bin/python eval/evaluate.py --clients all --source saved --mode local
```

Then run `streamlit run app/streamlit_app.py`, keep **Run with MiniLM (local)** selected (it works with no
internet), and walk through:

1. **Load the data.** Pick client c1. Show an invoice PDF (4 layouts; one deliberately unreadable scan) and
   the GSTR-2B rows for 09/2026 and 10/2026.
2. **Watch the matching.** Press **Run agent**. The trace shows the plan, each tool call and its result: 2B
   loaded, invoices parsed (1 unreadable flagged), matches per pass, then one decision per open item. The table
   turns green for matches, yellow for drafted emails / flags, orange for escalations.
3. **Review the email.** Open the wrong-GSTIN draft. The invoice number, amounts and the requested fix are
   highlighted. All numbers come from data; the model's explanation is dropped if it cites a figure that isn't in the
   evidence.
4. **See the escalation.** Open the "unclear" case: conflicting evidence (e.g. another GSTIN *and* a next-period
   copy). The reviewer sees the evidence and the agent's note, and can approve or override.
5. **Metrics** tab: match precision/recall, diagnosis accuracy per type, unsafe auto-clears, minutes and model
   calls per client.

## Safety design

- The **gate runs inside `draft_email`**, so the model can't bypass it. It refuses (and the agent must escalate)
  when the PDF is unreadable, the cause is `unclear`, the item is a fuzzy (pass-4) match, the model's cause
  disagrees with the rule-based evidence, or the confidence is below `CONFIDENCE_THRESHOLD` (0.75).
- **Done check**: after the agent stops, any undecided open item is auto-escalated as `agent_did_not_decide`
  and counted in the metrics.
- Tool results sent to the model are summaries (< 2 KB). Full records stay in ADK session state.
- Free-tier handling: retries with backoff on 429/5xx (shown in the trace), a cap on model calls per run, and
  replay mode as demo insurance.

## Layout

```
data/generate.py        synthetic clients, invoice PDFs, GSTR-2B (09/2026 + 10/2026), answer keys
zero_balance/           pure Python core (no LLM): gstin, normalise, parsing, gstr2b, matching, signals
agent/                  ADK agent: tools (6 + gate), prompts, models (NIM + offline), runner, driven fallback
app/streamlit_app.py    demo UI
eval/evaluate.py        metrics vs answer key → eval/results.json + NOTES.md
experiments/            what-didn't-work baselines (embedding-only matching; LLM-only for Gemma)
NOTES.md                spike results, decisions, numbers for the deck
```

`adk web` also works from this folder (`agent/` exposes `root_agent`).
