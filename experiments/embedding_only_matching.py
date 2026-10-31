"""'What didn't work' (MiniLM version): match invoices to GSTR-2B rows by sentence-embedding similarity alone,
instead of the deterministic passes, and score it with the same evaluator.

    python experiments/embedding_only_matching.py --clients all

Each invoice and each 2B row becomes a sentence ("supplier GSTIN ..., invoice no ..., dated ..., total Rs ...").
Pairs are assigned greedily by cosine similarity (each row once) above a threshold; everything left is escalated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import tools as T  # noqa: E402
from agent.minilm import _model  # noqa: E402
from eval.evaluate import aggregate, load_key, score, write_notes_section  # noqa: E402
from zero_balance.normalise import fmt_inr, next_period  # noqa: E402

OUT = ROOT / "eval" / "embedding_only_results.json"
THRESHOLD = 0.80


def sentence(r: dict) -> str:
    return (f"supplier GSTIN {r['supplier_gstin']}, invoice no {r['invoice_no']}, dated {r['invoice_date']}, "
            f"total Rs {fmt_inr(r['total'])}")


def run_client(client_id: str) -> tuple[dict, list[dict]]:
    run: dict = {}
    period = T.load_clients()[client_id]["period"]
    T.op_fetch_2b_rows(run, client_id, period)
    T.op_fetch_2b_rows(run, client_id, next_period(period))
    T.op_parse_invoice(run, client_id)
    invs = [i for i in run["invoices"] if i["parse_status"] != "unreadable"]
    rows = run["rows"]
    t0 = time.time()
    m = _model()
    sims = m.encode([sentence(i) for i in invs], normalize_embeddings=True) @ \
        m.encode([sentence(r) for r in rows], normalize_embeddings=True).T
    pairs, used_i, used_r = {}, set(), set()
    for flat in np.argsort(-sims, axis=None):
        a, b = divmod(int(flat), len(rows))
        if sims[a, b] < THRESHOLD:
            break
        if a in used_i or b in used_r:
            continue
        used_i.add(a)
        used_r.add(b)
        pairs[invs[a]["invoice_id"]] = (rows[b]["row_id"], float(sims[a, b]))
    actions = {i: "none" for i in pairs}
    for i in run["invoices"]:
        actions.setdefault(i["invoice_id"], "escalate")
    pred = {"pairs": {i: r for i, (r, _) in pairs.items()}, "passes": {}, "causes": {}, "actions": actions,
            "open_items": sum(a == "escalate" for a in actions.values()),
            "escalations": sum(a == "escalate" for a in actions.values()),
            "duration_s": time.time() - t0, "llm_calls": 0}
    key = load_key(client_id)
    inv_by_id = {i["invoice_id"]: i for i in run["invoices"]}
    row_by_id = {r["row_id"]: r for r in rows}
    wrong = []
    for i, (r, s) in pairs.items():
        if key[i]["expected_2b_row_id"] != r:
            inv, row = inv_by_id[i], row_by_id[r]
            wrong.append({"invoice_id": i, "case": key[i]["case"], "similarity": round(s, 3),
                          "invoice": f"{inv['supplier_gstin']} {inv['invoice_no']} Rs {inv['total']}",
                          "matched_2b_row": f"{r}: {row['supplier_gstin']} {row['invoice_no']} Rs {row['total']}",
                          "expected_row": key[i]["expected_2b_row_id"]})
    return pred, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", default="all")
    args = ap.parse_args()
    clients = ["c1", "c2", "c3"] if args.clients == "all" else args.clients.split(",")
    per_client, examples = {}, {}
    for c in clients:
        pred, wrong = run_client(c)
        per_client[c] = score(load_key(c), pred)
        examples[c] = wrong
        mt = per_client[c]["match"]
        print(f"{c}: precision {mt['precision']:.1%} recall {mt['recall']:.1%} wrong pairs {len(wrong)} "
              f"unsafe auto-matches {len(per_client[c]['wrong_auto_matches'])}")
    agg = aggregate(per_client)
    OUT.write_text(json.dumps({"model": "sentence-transformers/all-MiniLM-L6-v2", "threshold": THRESHOLD,
                               "aggregate": agg, "per_client": per_client, "failure_examples": examples}, indent=1))
    n_wrong = sum(len(v) for v in examples.values())
    lines = ["### Embedding-only matching baseline (all-MiniLM-L6-v2, cosine ≥ "
             f"{THRESHOLD}, greedy one-to-one)", "",
             f"Match precision {agg['match']['precision']:.1%}, recall {agg['match']['recall']:.1%} "
             f"({agg['match']['tp']}/{agg['match']['expected']} correct pairs, {n_wrong} wrong pairs that would "
             "have cleared silently). Deterministic passes on the same data: 100% / 100%.", "",
             "Failure examples:"]
    shown = [e for ex in examples.values() for e in ex][:4]
    for e in shown:
        lines.append(f"- {e['invoice_id']} ({e['case']}): invoice `{e['invoice']}` matched to "
                     f"`{e['matched_2b_row']}` (sim {e['similarity']}); expected {e['expected_row']}")
    write_notes_section("EMBEDDING-ONLY", "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
