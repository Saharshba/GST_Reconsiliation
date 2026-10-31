"""Zero Balance demo UI.  Run: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import asyncio
import html
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agent import tools as T  # noqa: E402
from agent.runner import latest_run, run_client_async  # noqa: E402
from zero_balance import gstr2b  # noqa: E402
from zero_balance.normalise import fmt_inr  # noqa: E402
from zero_balance.schema import RunResult, TraceEvent  # noqa: E402

st.set_page_config(page_title="Zero Balance", page_icon="🧾", layout="wide")
st.markdown("""
<style>
.trace-line {font-family: ui-monospace, Menlo, monospace; font-size: 12.5px; padding: 3px 6px; margin: 1px 0;
             border-left: 3px solid #ccc; white-space: pre-wrap; word-break: break-word;}
.trace-call {border-color: #3b82f6; background: rgba(59,130,246,.07);}
.trace-result {border-color: #94a3b8; opacity: .72;}
.trace-llm {border-color: #8b5cf6; background: rgba(139,92,246,.08); font-family: inherit; font-size: 14px;}
.trace-warn {border-color: #f59e0b; background: rgba(245,158,11,.12);}
.trace-err {border-color: #ef4444; background: rgba(239,68,68,.12);}
.trace-user {border-color: #10b981; background: rgba(16,185,129,.08); font-family: inherit;}
.email-box {border: 1px solid #ddd; border-radius: 8px; padding: 14px 18px;
            font-family: inherit; line-height: 1.5; background: rgba(127,127,127,.04);}
mark {background: #fde68a; color: #111; padding: 0 2px; border-radius: 3px;}
</style>
""", unsafe_allow_html=True)

CLIENTS = T.load_clients()
TOOL_ICON = {"fetch_2b_rows": "📥", "parse_invoice": "📄", "match_records": "🔗", "validate_gstin": "✅",
             "draft_email": "✉️", "escalate_case": "🚩"}

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧾 Zero Balance")
    st.caption("GSTR-2B reconciliation agent · Google ADK + MiniLM (local)")
    cid = st.selectbox("Client", list(CLIENTS), format_func=lambda c: f"{c} · {CLIENTS[c]['name']}")
    has_key = bool(os.environ.get("NVIDIA_NIM_API_KEY"))
    modes = ["Run with MiniLM (local)", "Replay saved run", "Live (Gemma 4 via NVIDIA NIM)",
             "Offline scripted (no model)"]
    default = 1 if os.environ.get("LLM_MODE") == "replay" else 0
    mode = st.radio("Mode", modes, index=default)
    if mode.startswith("Live") and not has_key:
        st.warning("NVIDIA_NIM_API_KEY is not set in .env, so live mode is unavailable.")
    st.caption(f"Model: `{os.environ.get('MODEL_ID', 'nvidia_nim/google/gemma-4-31b-it')}`  \n"
               f"Confidence threshold: {T.threshold():.2f}")

client = CLIENTS[cid]
results: dict[str, RunResult] = st.session_state.setdefault("results", {})
reviews: dict[str, dict] = st.session_state.setdefault("reviews", {})


@st.cache_data
def load_rows(client_id: str, period: str):
    return [r.model_dump() for r in gstr2b.load(T.data_dir() / client_id / f"gstr2b_{period}.json")]


@st.cache_data
def pdf_image(path: str):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return pdf.pages[0].to_image(resolution=90).original


def trace_html(ev: TraceEvent) -> str:
    t = f"{ev.t:5.1f}s"
    if ev.kind == "user":
        return f'<div class="trace-line trace-user">👤 {t}  {html.escape(str(ev.data))}</div>'
    if ev.kind == "llm_text":
        return f'<div class="trace-line trace-llm">🧠 {t}  {html.escape(str(ev.data))}</div>'
    if ev.kind == "tool_call":
        args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (ev.data or {}).items())
        return f'<div class="trace-line trace-call">{TOOL_ICON.get(ev.name, "🔧")} {t}  <b>{ev.name}</b>({html.escape(args)})</div>'
    if ev.kind == "tool_result":
        d = ev.data
        if isinstance(d, dict) and "open_items" in d:
            head = json.dumps({k: v for k, v in d.items() if k != "open_items"}, ensure_ascii=False)
            items = "".join(f"\n     • {o['invoice_id']}  {o.get('supplier') or ''}  →  suggested "
                            f"{o['suggested_cause']}: {'; '.join(o.get('facts', []))}" for o in d["open_items"])
            return f'<div class="trace-line trace-result">   ↳ {html.escape(head)}{html.escape(items)}</div>'
        s = d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)
        return f'<div class="trace-line trace-result">   ↳ {html.escape(s[:500])}</div>'
    if ev.kind == "warning":
        return f'<div class="trace-line trace-warn">⏳ {t}  {html.escape(str(ev.data))}</div>'
    if ev.kind == "error":
        return f'<div class="trace-line trace-err">❌ {t}  {ev.name}: {html.escape(str(ev.data))}</div>'
    return f'<div class="trace-line">ℹ️ {t}  {ev.name}: {html.escape(json.dumps(ev.data, default=str))}</div>'


def results_table(r: RunResult) -> pd.DataFrame:
    match = {m.invoice_id: m for m in r.matches}
    diag = {d.invoice_id: d for d in r.diagnoses}
    action = {e.invoice_id: "✉️ email drafted" for e in r.emails}
    action |= {f.invoice_id: "📌 flag next return" for f in r.flags}
    action |= {e.invoice_id: "🚩 escalated" for e in r.escalations}
    vendors = {v["gstin"]: v["name"] for v in CLIENTS[r.client_id]["vendors"]}
    rows = []
    for inv in r.invoices:
        m, d = match.get(inv.invoice_id), diag.get(inv.invoice_id)
        rows.append({
            "invoice": inv.invoice_id,
            "supplier": vendors.get(inv.supplier_gstin, inv.supplier_name) if inv.parse_status != "unreadable"
            else f"(unreadable PDF {inv.source_file})",
            "invoice no.": inv.invoice_no or "—", "amount": fmt_inr(inv.total) if inv.total else "—",
            "2B row": m.row_id if m else "—", "pass": str(m.pass_no) if m else "—",
            "cause": d.cause if d else ("—" if m else ""),
            "confidence": f"{d.confidence:.2f}" if d else (f"{m.confidence:.2f}" if m else "—"),
            "action": action.get(inv.invoice_id, "✔ matched" if m else "?"),
        })
    return pd.DataFrame(rows)


def colour(row):
    a = row["action"]
    c = "#dcfce7" if a.startswith("✔") else "#fef9c3" if a.startswith(("✉", "📌")) else "#ffedd5"
    return [f"background-color: {c}; color: #111"] * len(row)


# ----------------------------------------------------------------------------- tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 · Load the data", "2 · Watch the matching", "3 · Review the email",
                                        "4 · See the escalation", "📊 Metrics"])

with tab1:
    st.subheader(f"{client['name']}")
    st.caption(f"GSTIN {client['gstin']} · filing period {client['period'][:2]}/{client['period'][2:]} · "
               f"{client['n_invoices']} purchase invoices · {len(client['vendors'])} vendors")
    c1, c2 = st.columns([2, 3])
    with c1:
        files = sorted((T.data_dir() / cid / "invoices").glob("*.pdf"))
        f = st.selectbox("Invoice PDF", files, format_func=lambda p: p.name)
        st.image(pdf_image(str(f)), width="stretch")
    with c2:
        period = st.radio("GSTR-2B period", [client["period"], "102026"], horizontal=True,
                          format_func=lambda p: f"{p[:2]}/{p[2:]}" + (" (filing)" if p == client["period"] else " (next)"))
        rows = load_rows(cid, period)
        df = pd.DataFrame(rows)[["row_id", "supplier_gstin", "supplier_name", "invoice_no", "invoice_date",
                                 "taxable_value", "igst", "cgst", "sgst", "total"]]
        st.dataframe(df, hide_index=True, width="stretch", height=520)

with tab2:
    run_col, info_col = st.columns([1, 3])
    go = run_col.button("▶ Run agent", type="primary", width="stretch",
                        disabled=mode.startswith("Live") and not has_key)
    info_col.caption({"Replay saved run": "Replays the latest recorded run with realistic delays (demo insurance).",
                      "Run with MiniLM (local)": "Runs the ADK pipeline; all-MiniLM-L6-v2 diagnoses each open item "
                                                 "locally (no API, no credits).",
                      "Offline scripted (no model)": "Runs the real ADK pipeline with a scripted stand-in for the "
                                                   "model. No network, no credits.",
                      }.get(mode, "Calls Gemma 4 on NVIDIA NIM. Free tier: expect rate-limit pauses."))
    left, right = st.columns([5, 6])
    with left:
        st.markdown("**Agent trace**")
        trace_box = st.container(height=620)
        trace_ph = trace_box.empty()
    with right:
        st.markdown("**Results**")
        table_ph = st.empty()
        stats_ph = st.empty()

    def render_trace(events: list[TraceEvent]):
        trace_ph.markdown("".join(trace_html(e) for e in events), unsafe_allow_html=True)

    def render_results(r: RunResult):
        df = results_table(r)
        table_ph.dataframe(df.style.apply(colour, axis=1), hide_index=True, width="stretch", height=560)
        stats_ph.markdown(
            f"**{len(r.matches)}** matched · **{len(r.emails)}** emails · **{len(r.flags)}** flagged · "
            f"**{len(r.escalations)}** escalated · {r.llm_calls} model calls · {r.duration_s:.0f}s · "
            f"mode `{r.mode}` · model `{r.model}`")

    if go:
        events: list[TraceEvent] = []
        if mode.startswith("Replay"):
            saved = latest_run(cid)
            if saved is None:
                st.error("No saved run for this client yet. Do a live or offline run first.")
            else:
                prev = 0.0
                for ev in saved.trace:
                    time.sleep(min(max(ev.t - prev, 0.15), 1.5))
                    prev = ev.t
                    events.append(ev)
                    render_trace(events)
                results[cid] = saved
        else:
            mid = ("offline/scripted" if mode.startswith("Offline") else "local/minilm" if mode.startswith("Run with")
                   else "nvidia_nim/google/gemma-4-31b-it")
            q: queue.Queue = queue.Queue()
            box: dict = {}

            def work():
                try:
                    box["result"] = asyncio.run(run_client_async(cid, mid, on_event=q.put))
                except Exception as e:  # noqa: BLE001
                    box["error"] = e

            th = threading.Thread(target=work, daemon=True)
            th.start()
            while th.is_alive() or not q.empty():
                changed = False
                while not q.empty():
                    events.append(q.get())
                    changed = True
                if changed:
                    render_trace(events)
                time.sleep(0.25)
            if "error" in box:
                st.error(f"Run failed: {box['error']}")
            else:
                results[cid] = box["result"]
    r = results.get(cid)
    if r is not None:
        render_trace(r.trace)
        render_results(r)
    elif not go:
        trace_ph.info("Press **Run agent** to start.")

r = results.get(cid)

with tab3:
    if r is None:
        st.info("Run the agent first (step 2).")
    elif not r.emails and not r.flags:
        st.info("No emails drafted for this client.")
    else:
        if r.emails:
            pick = st.selectbox("Drafted email", r.emails,
                                format_func=lambda e: f"{e.invoice_id} · {e.to_vendor} · {e.cause}")
            body = html.escape(pick.body)
            for h in sorted(set(pick.highlights), key=len, reverse=True):
                body = body.replace(html.escape(h), f"<mark>{html.escape(h)}</mark>")
            st.markdown(f"**To:** {html.escape(pick.to_vendor)} ({pick.to_gstin})  \n"
                        f"**Subject:** {html.escape(pick.subject)}", unsafe_allow_html=True)
            body = body.replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")
            st.markdown(f'<div class="email-box">{body}</div>', unsafe_allow_html=True)
            d = next((d for d in r.diagnoses if d.invoice_id == pick.invoice_id), None)
            if d:
                st.caption(f"Agent diagnosis: **{d.cause}** · confidence {d.confidence:.2f} · {d.explanation}")
            st.caption("Drafts only. Zero Balance never sends email.")
        if r.flags:
            st.markdown("**Flagged for next return**")
            for fl in r.flags:
                st.markdown(f"- `{fl.invoice_id}`: {fl.note}")

with tab4:
    if r is None:
        st.info("Run the agent first (step 2).")
    elif not r.escalations:
        st.success("Nothing escalated.")
    else:
        st.caption("Reviewer queue: cases the agent would not clear on its own. Decisions stay local to this demo.")
        for e in r.escalations:
            key = f"{r.run_id}:{e.invoice_id}"
            status = reviews.get(key, {}).get("status", "pending")
            badge = {"pending": "🟠 pending", "approved": "✅ approved", "overridden": "✏️ overridden"}[status]
            with st.expander(f"{badge} · {e.invoice_id} · {e.reason}", expanded=status == "pending"):
                a, b = st.columns([3, 2])
                with a:
                    st.markdown(f"**Agent note:** {e.agent_note or '—'}")
                    st.markdown(f"**Agent cause:** `{e.llm_cause}` · confidence "
                                f"{'—' if e.llm_confidence is None else f'{e.llm_confidence:.2f}'} · rules suggested "
                                f"`{e.evidence.get('suggested_cause')}`")
                    st.markdown("**Evidence**")
                    st.json({k: v for k, v in e.evidence.items() if k != "suggested_cause"}, expanded=True)
                with b:
                    inv = next((i for i in r.invoices if i.invoice_id == e.invoice_id), None)
                    if inv is not None:
                        pdf = T.data_dir() / r.client_id / "invoices" / inv.source_file
                        if pdf.exists():
                            st.image(pdf_image(str(pdf)), width="stretch")
                    if st.button("Approve agent's call", key=key + ":ok"):
                        reviews[key] = {"status": "approved"}
                        st.rerun()
                    new = st.selectbox("Override cause", ["wrong_gstin", "vendor_not_filed", "amount_typo",
                                                          "period_mismatch", "unclear", "matched_ok"], key=key + ":sel")
                    if st.button("Override", key=key + ":ov"):
                        reviews[key] = {"status": "overridden", "cause": new}
                        st.rerun()
                    if status == "overridden":
                        st.caption(f"Overridden to `{reviews[key]['cause']}`")

with tab5:
    res_path = ROOT / "eval" / "results.json"
    if not res_path.exists():
        st.info("No evaluation yet. Run `python eval/evaluate.py --clients all`.")
    else:
        res = json.loads(res_path.read_text())
        agg = res["aggregate"]
        if res["label"] == "local":
            st.info("Diagnosis by all-MiniLM-L6-v2 nearest neighbours over labelled past cases (seeds 1-3); demo data "
                    "held out. Synthetic evidence lines share templates, so this is an easy test.")
        if res["label"] == "offline":
            st.warning("These numbers come from the offline scripted stand-in (rules only, no LLM). "
                       "They check the pipeline, not the model.")
        st.caption(f"Runs: {', '.join(res['runs'].values())} · model(s): {', '.join(res['models'])}")
        k = st.columns(6)
        k[0].metric("Match precision", f"{agg['match']['precision']:.1%}")
        k[1].metric("Match recall", f"{agg['match']['recall']:.1%}")
        k[2].metric("Diagnosis accuracy", f"{agg['diagnosis']['accuracy']:.1%}")
        k[3].metric("Unsafe auto-clears", agg["unsafe_auto_clears"])
        k[4].metric("Minutes / client", agg["minutes_per_client"])
        k[5].metric("LLM calls / client", agg["llm_calls_per_client"])
        a, b = st.columns(2)
        with a:
            st.markdown("**Per client**")
            st.dataframe(pd.DataFrame([{
                "client": c, "precision": x["match"]["precision"], "recall": x["match"]["recall"],
                "diagnosis": x["diagnosis"]["accuracy"], "actions": x["actions"]["accuracy"],
                "unsafe": x["unsafe_auto_clears"], "escalation rate": x["escalation_rate"],
                "minutes": x["minutes"], "llm calls": x["llm_calls"]} for c, x in res["per_client"].items()]),
                hide_index=True, width="stretch")
            st.markdown("**Diagnosis accuracy per mismatch type**")
            st.dataframe(pd.DataFrame([{"cause": t, **v} for t, v in agg["diagnosis"]["per_type"].items()]),
                         hide_index=True, width="stretch")
        with b:
            st.markdown("**Confusion matrix** (rows = expected, columns = predicted)")
            st.dataframe(pd.DataFrame(agg["diagnosis"]["confusion"]).T.fillna(0).astype(int),
                         width="stretch")
        emb = ROOT / "eval" / "embedding_only_results.json"
        if emb.exists():
            eb = json.loads(emb.read_text())
            st.markdown("**What didn't work: matching by embedding similarity alone (MiniLM)**")
            e1, e2, e3 = st.columns(3)
            e1.metric("Match precision", f"{eb['aggregate']['match']['precision']:.1%}")
            e2.metric("Match recall", f"{eb['aggregate']['match']['recall']:.1%}")
            e3.metric("Wrong pairs cleared silently", sum(len(v) for v in eb["failure_examples"].values()))
            st.caption("The deterministic passes score 100% precision and 100% recall on the same data.")
            st.dataframe(pd.DataFrame([e for v in eb["failure_examples"].values() for e in v]),
                         hide_index=True, width="stretch")
        base = ROOT / "eval" / "llm_only_results.json"
        if base.exists():
            bl = json.loads(base.read_text())
            st.markdown("**What didn't work: the whole job in one LLM prompt**")
            st.json(bl.get("aggregate", bl), expanded=False)
