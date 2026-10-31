"""Score agent runs against answer_key.json.

    python eval/evaluate.py --clients all --source saved            # latest saved run per client
    python eval/evaluate.py --clients all --source run --model offline/scripted
    python eval/evaluate.py --clients c1 --source run                 # live (uses credits)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zero_balance.schema import CAUSES, RunResult  # noqa: E402

DATA = ROOT / "data" / "sample"
OUT = ROOT / "eval" / "results.json"
EMAIL_ACTION = {"wrong_gstin": "email_correction", "vendor_not_filed": "email_filing_reminder",
                "amount_typo": "email_amount_query"}


def load_key(client_id: str, data_dir: Path = DATA) -> dict[str, dict]:
    return {k["invoice_id"]: k for k in json.loads((data_dir / client_id / "answer_key.json").read_text())}


def prediction_from_run(r: RunResult) -> dict:
    """Normalise a RunResult into {pairs, passes, causes, actions}."""
    causes: dict[str, str] = {}
    for d in r.diagnoses:
        causes[d.invoice_id] = d.cause
    for e in r.escalations:
        if e.invoice_id not in causes:
            causes[e.invoice_id] = "agent_did_not_decide" if e.reason == "agent_did_not_decide" else (e.llm_cause or "unclear")
    actions: dict[str, str] = {m.invoice_id: "none" for m in r.matches}
    for e in r.emails:
        actions[e.invoice_id] = EMAIL_ACTION[e.cause]
    for f in r.flags:
        actions[f.invoice_id] = "flag_next_return"
    for e in r.escalations:
        actions[e.invoice_id] = "escalate"
    return {
        "pairs": {m.invoice_id: m.row_id for m in r.matches},
        "passes": {m.invoice_id: m.pass_no for m in r.matches},
        "causes": causes, "actions": actions,
        "auto_escalated": sum(e.reason == "agent_did_not_decide" for e in r.escalations),
        "open_items": len(r.open_items), "escalations": len(r.escalations),
        "duration_s": r.duration_s, "llm_calls": r.llm_calls, "model": r.model, "mode": r.mode,
        "run_id": r.run_id,
    }


def score(key: dict[str, dict], pred: dict) -> dict:
    exp_pairs = {i: k["expected_2b_row_id"] for i, k in key.items() if k["expected_2b_row_id"]}
    got_pairs = pred["pairs"]
    tp = sum(1 for i, r in got_pairs.items() if exp_pairs.get(i) == r)
    precision = tp / len(got_pairs) if got_pairs else 0.0
    recall = tp / len(exp_pairs) if exp_pairs else 0.0
    pass_ok = sum(1 for i, k in key.items() if k["expected_pass"] and pred.get("passes", {}).get(i) == k["expected_pass"])

    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion: dict[str, Counter] = defaultdict(Counter)
    for i, k in key.items():
        if not k["expected_cause"]:
            continue
        got = pred["causes"].get(i, "none")
        per_type[k["expected_cause"]][1] += 1
        per_type[k["expected_cause"]][0] += got == k["expected_cause"]
        confusion[k["expected_cause"]][got] += 1

    action_ok, unsafe, unsafe_items = 0, 0, []
    for i, k in key.items():
        got = pred["actions"].get(i, "none")
        action_ok += got in k["acceptable_actions"]
        auto_cleared = got.startswith("email") or got == "flag_next_return"
        if auto_cleared and got not in k["acceptable_actions"]:
            unsafe += 1
            unsafe_items.append({"invoice_id": i, "case": k["case"], "expected": k["expected_action"], "got": got})
    wrong_auto_matches = [i for i, r in got_pairs.items()
                          if exp_pairs.get(i) != r and pred.get("passes", {}).get(i, 1) < 4]
    n_causes = sum(v[1] for v in per_type.values())
    return {
        "match": {"precision": round(precision, 4), "recall": round(recall, 4), "tp": tp,
                  "predicted": len(got_pairs), "expected": len(exp_pairs),
                  "pass_correct": pass_ok, "pass_expected": sum(1 for k in key.values() if k["expected_pass"])},
        "diagnosis": {"accuracy": round(sum(v[0] for v in per_type.values()) / n_causes, 4) if n_causes else None,
                      "per_type": {t: {"correct": v[0], "total": v[1], "accuracy": round(v[0] / v[1], 3)}
                                   for t, v in sorted(per_type.items())},
                      "confusion": {t: dict(c) for t, c in sorted(confusion.items())}},
        "actions": {"correct": action_ok, "total": len(key), "accuracy": round(action_ok / len(key), 4)},
        "unsafe_auto_clears": unsafe, "unsafe_items": unsafe_items,
        "wrong_auto_matches": wrong_auto_matches,
        "escalation_rate": round(pred["escalations"] / pred["open_items"], 3) if pred.get("open_items") else None,
        "agent_did_not_decide": pred.get("auto_escalated", 0),
        "minutes": round(pred.get("duration_s", 0) / 60, 2), "llm_calls": pred.get("llm_calls", 0),
    }


def aggregate(per_client: dict[str, dict]) -> dict:
    s = list(per_client.values())
    tp = sum(x["match"]["tp"] for x in s)
    pred = sum(x["match"]["predicted"] for x in s)
    exp = sum(x["match"]["expected"] for x in s)
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion: dict[str, Counter] = defaultdict(Counter)
    for x in s:
        for t, v in x["diagnosis"]["per_type"].items():
            per_type[t][0] += v["correct"]
            per_type[t][1] += v["total"]
        for t, c in x["diagnosis"]["confusion"].items():
            confusion[t].update(c)
    n = sum(v[1] for v in per_type.values())
    return {
        "match": {"precision": round(tp / pred, 4) if pred else 0, "recall": round(tp / exp, 4) if exp else 0,
                  "tp": tp, "predicted": pred, "expected": exp},
        "diagnosis": {"accuracy": round(sum(v[0] for v in per_type.values()) / n, 4) if n else None,
                      "per_type": {t: {"correct": v[0], "total": v[1], "accuracy": round(v[0] / v[1], 3)}
                                   for t, v in sorted(per_type.items())},
                      "confusion": {t: dict(c) for t, c in sorted(confusion.items())}},
        "actions_accuracy": round(sum(x["actions"]["correct"] for x in s) / sum(x["actions"]["total"] for x in s), 4),
        "unsafe_auto_clears": sum(x["unsafe_auto_clears"] for x in s),
        "wrong_auto_matches": sum(len(x["wrong_auto_matches"]) for x in s),
        "escalation_rate": round(sum(x["escalation_rate"] or 0 for x in s) / len(s), 3),
        "agent_did_not_decide": sum(x["agent_did_not_decide"] for x in s),
        "minutes_per_client": round(sum(x["minutes"] for x in s) / len(s), 2),
        "llm_calls_per_client": round(sum(x["llm_calls"] for x in s) / len(s), 1),
    }


def markdown(title: str, per_client: dict, agg: dict, note: str = "") -> str:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines += ["| Client | Match P | Match R | Diagnosis acc. | Action acc. | Unsafe auto-clears | Escalation rate | "
              "Min/client | LLM calls |", "|---|---|---|---|---|---|---|---|---|"]
    for c, x in per_client.items():
        lines.append(f"| {c} | {x['match']['precision']:.1%} | {x['match']['recall']:.1%} | "
                     f"{x['diagnosis']['accuracy']:.1%} | {x['actions']['accuracy']:.1%} | {x['unsafe_auto_clears']} | "
                     f"{x['escalation_rate']:.0%} | {x['minutes']} | {x['llm_calls']} |")
    lines.append(f"| **all** | **{agg['match']['precision']:.1%}** | **{agg['match']['recall']:.1%}** | "
                 f"**{agg['diagnosis']['accuracy']:.1%}** | **{agg['actions_accuracy']:.1%}** | "
                 f"**{agg['unsafe_auto_clears']}** | **{agg['escalation_rate']:.0%}** | "
                 f"**{agg['minutes_per_client']}** | **{agg['llm_calls_per_client']}** |")
    lines += ["", "Diagnosis accuracy per mismatch type:", "", "| Cause | Correct | Total |", "|---|---|---|"]
    for t, v in agg["diagnosis"]["per_type"].items():
        lines.append(f"| {t} | {v['correct']} | {v['total']} |")
    lines += ["", "Confusion (expected → predicted): " + "; ".join(
        f"{t}: " + ", ".join(f"{k}×{n}" for k, n in c.items()) for t, c in agg["diagnosis"]["confusion"].items())]
    return "\n".join(lines) + "\n"


def write_notes_section(marker: str, text: str):
    notes = ROOT / "NOTES.md"
    body = notes.read_text() if notes.exists() else ""
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    block = f"{start}\n{text}{end}"
    if start in body:
        body = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, body, flags=re.S)
    else:
        body = body.rstrip() + "\n\n" + block + "\n"
    notes.write_text(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", default="all")
    ap.add_argument("--source", choices=["saved", "run"], default="saved")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", default=None, help="with --source saved: only runs of this mode (live/local/offline)")
    args = ap.parse_args()
    clients = ["c1", "c2", "c3"] if args.clients == "all" else args.clients.split(",")

    from agent.runner import latest_run, run_client
    per_client, runs = {}, {}
    for c in clients:
        r = run_client(c, args.model) if args.source == "run" else latest_run(c, args.mode)
        if r is None:
            sys.exit(f"no saved run for {c}; use --source run")
        runs[c] = r
        per_client[c] = score(load_key(c), prediction_from_run(r))
    agg = aggregate(per_client)
    modes = {r.mode for r in runs.values()}
    label = modes.pop() if len(modes) == 1 else "mixed"
    results = {"label": label, "models": sorted({r.model for r in runs.values()}),
               "runs": {c: r.run_id for c, r in runs.items()}, "aggregate": agg, "per_client": per_client}
    OUT.write_text(json.dumps(results, indent=1))
    if label == "offline":
        title, note = ("Agent pipeline, offline scripted model (rules-only baseline, NOT an LLM result)",
                       "The scripted stand-in always follows the rule-based suggested_cause. These numbers check "
                       "the pipeline and the gate; they are not model numbers.")
    elif label == "local":
        import os
        title = f"Agent with local MiniLM (all-MiniLM-L6-v2, {os.environ.get('MINILM_MODE', 'fewshot')})"
        note = ("Diagnosis by sentence-embedding nearest neighbours over a labelled bank built from seeds 1-3; "
                "the demo data (seed 42) is held out. Caveat: synthetic evidence lines share templates across seeds, "
                "so this is an easy test. Model calls = deterministic policy steps, not LLM calls.")
    else:
        title, note = f"Agent with {', '.join(results['models'])} ({label} runs)", ""
    marker = f"EVAL-{label.upper()}"
    if label == "local":
        import os
        marker += "-" + os.environ.get("MINILM_MODE", "fewshot").upper()
    write_notes_section(marker, markdown(title, per_client, agg, note))
    print(markdown(title, per_client, agg, note))
    if agg["unsafe_auto_clears"]:
        print("UNSAFE:", [x["unsafe_items"] for x in per_client.values() if x["unsafe_items"]])


if __name__ == "__main__":
    main()
