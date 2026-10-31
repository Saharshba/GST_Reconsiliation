"""'What didn't work': give the WHOLE reconciliation to the LLM in one prompt, score it with the same evaluator.

    python experiments/llm_only_matching.py --clients all --dry-run   # build prompts only, no API call
    python experiments/llm_only_matching.py --clients all             # 1 LLM call per client (uses credits)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agent import tools as T  # noqa: E402
from eval.evaluate import EMAIL_ACTION, aggregate, load_key, score, write_notes_section  # noqa: E402
from zero_balance.normalise import next_period  # noqa: E402

OUT = ROOT / "eval" / "llm_only_results.json"
PROMPT = """You are reconciling a client's purchase invoices against GSTR-2B for period {period}.

PURCHASE INVOICES (from the client's books; UNREADABLE means the PDF could not be read):
invoice_id,supplier_gstin,invoice_no,date,taxable,total
{invoices}

GSTR-2B ROWS, period {period}:
row_id,supplier_gstin,invoice_no,date,taxable,total
{rows}

GSTR-2B ROWS, next period {next_period}:
row_id,supplier_gstin,invoice_no,date,taxable,total
{next_rows}

Task:
1. Match every invoice to at most one {period} 2B row (each row at most once). Small formatting differences in
   invoice numbers and amount differences up to Rs 1 are fine.
2. For every invoice you could not match, give the cause: wrong_gstin (same number+amount under another GSTIN),
   vendor_not_filed (no 2B rows for that GSTIN), amount_typo (same GSTIN+number, amount differs),
   period_mismatch (only in the next period), unreadable, or unclear.
   Include the invoice total and the 2B amount you compared.

Return ONLY JSON:
{{"matches": [{{"invoice_id": "...", "row_id": "..."}}],
  "issues": [{{"invoice_id": "...", "cause": "...", "invoice_total": 0.0, "row_total": 0.0, "explanation": "..."}}]}}
"""


def build_prompt(client_id: str) -> tuple[str, dict]:
    run: dict = {}
    client = T.load_clients()[client_id]
    period = client["period"]
    T.op_fetch_2b_rows(run, client_id, period)
    T.op_fetch_2b_rows(run, client_id, next_period(period))
    T.op_parse_invoice(run, client_id)

    def inv_line(i):
        if i["parse_status"] == "unreadable":
            return f"{i['invoice_id']},UNREADABLE,,,,"
        return f"{i['invoice_id']},{i['supplier_gstin']},{i['invoice_no']},{i['invoice_date']},{i['taxable_value']},{i['total']}"

    def row_line(r):
        return f"{r['row_id']},{r['supplier_gstin']},{r['invoice_no']},{r['invoice_date']},{r['taxable_value']},{r['total']}"

    prompt = PROMPT.format(period=period, next_period=next_period(period),
                           invoices="\n".join(inv_line(i) for i in run["invoices"]),
                           rows="\n".join(row_line(r) for r in run["rows"]),
                           next_rows="\n".join(row_line(r) for r in run["next_rows"]))
    return prompt, run


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def to_prediction(out: dict, run: dict) -> tuple[dict, list[dict]]:
    """Map the LLM's answer to the evaluator's prediction format and collect concrete failures."""
    rows = {r["row_id"]: r for r in run["rows"]}
    invs = {i["invoice_id"]: i for i in run["invoices"]}
    pairs, failures, used = {}, [], set()
    for m in out.get("matches", []):
        iid, rid = m.get("invoice_id"), m.get("row_id")
        if rid not in rows:
            failures.append({"type": "hallucinated_row", "invoice_id": iid, "row_id": rid})
            continue
        if rid in used:
            failures.append({"type": "row_matched_twice", "invoice_id": iid, "row_id": rid})
        used.add(rid)
        pairs[iid] = rid
    causes, actions = {}, {i: "none" for i in pairs}
    for it in out.get("issues", []):
        iid, cause = it.get("invoice_id"), it.get("cause", "unclear")
        causes[iid] = cause
        actions[iid] = EMAIL_ACTION.get(cause, "flag_next_return" if cause == "period_mismatch" else "escalate")
        src = invs.get(iid)
        if src and src.get("total") is not None and it.get("invoice_total") is not None \
                and abs(float(it["invoice_total"]) - src["total"]) > 0.01:
            failures.append({"type": "copied_amount_wrong", "invoice_id": iid,
                             "llm_said": it["invoice_total"], "source": src["total"]})
    return {"pairs": pairs, "passes": {}, "causes": causes, "actions": actions,
            "open_items": len(causes), "escalations": sum(a == "escalate" for a in actions.values())}, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    clients = ["c1", "c2", "c3"] if args.clients == "all" else args.clients.split(",")
    import os
    model = os.environ.get("MODEL_ID", "nvidia_nim/google/gemma-4-31b-it")

    per_client, examples = {}, {}
    for c in clients:
        prompt, run = build_prompt(c)
        if args.dry_run:
            print(f"{c}: prompt {len(prompt)} chars (~{len(prompt) // 4} tokens)")
            continue
        import litellm
        t0 = time.time()
        resp = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}],
                                  temperature=0.1, max_tokens=4096, num_retries=3)
        text = resp.choices[0].message.content or ""
        try:
            out = parse_json(text)
        except Exception as e:  # noqa: BLE001
            out = {}
            examples[c] = [{"type": "invalid_json", "error": str(e), "head": text[:300]}]
        pred, failures = to_prediction(out, run)
        pred |= {"duration_s": time.time() - t0, "llm_calls": 1}
        key = load_key(c)
        s = score(key, pred)
        wrong = [{"type": "wrong_pair", "invoice_id": i, "llm_row": r, "expected_row": key[i]["expected_2b_row_id"],
                  "case": key[i]["case"]} for i, r in pred["pairs"].items()
                 if i in key and key[i]["expected_2b_row_id"] != r]
        examples[c] = examples.get(c, []) + failures + wrong
        per_client[c] = s
        (ROOT / "eval" / f"llm_only_raw_{c}.txt").write_text(text)
        print(c, json.dumps(s["match"]), "diag", s["diagnosis"]["accuracy"], "unsafe", s["unsafe_auto_clears"])
    if args.dry_run:
        return
    agg = aggregate(per_client)
    OUT.write_text(json.dumps({"model": model, "aggregate": agg, "per_client": per_client,
                               "failure_examples": examples}, indent=1))
    lines = [f"### LLM-only baseline ({model}, one prompt per client)", "",
             f"Match precision {agg['match']['precision']:.1%}, recall {agg['match']['recall']:.1%}; "
             f"diagnosis accuracy {agg['diagnosis']['accuracy'] or 0:.1%}; unsafe auto-clears "
             f"{agg['unsafe_auto_clears']}.", "", "Failure examples:"]
    for c, ex in examples.items():
        for e in ex[:3]:
            lines.append(f"- {c}: `{json.dumps(e)}`")
    write_notes_section("LLM-ONLY", "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
