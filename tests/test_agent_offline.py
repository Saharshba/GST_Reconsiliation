"""Agent pipeline tests with the offline scripted model (no network, no LLM)."""
import json

import pytest

from agent import tools as T
from agent.runner import run_client
from zero_balance.schema import OpenItem


@pytest.fixture(scope="module")
def results():
    return {c: run_client(c, "offline/scripted", save=False) for c in ("c1", "c2", "c3")}


def test_every_open_item_decided_exactly_once(results):
    for r in results.values():
        decided = [e.invoice_id for e in r.emails] + [f.invoice_id for f in r.flags] + \
                  [e.invoice_id for e in r.escalations]
        assert sorted(decided) == sorted(o.invoice_id for o in r.open_items)
        assert not any(e.reason == "agent_did_not_decide" for e in r.escalations)


def test_tool_payloads_are_small():
    for c in ("c1", "c2", "c3"):
        run = {}
        sizes = [len(json.dumps(T.op_fetch_2b_rows(run, c, "092026"))),
                 len(json.dumps(T.op_fetch_2b_rows(run, c, "102026"))),
                 len(json.dumps(T.op_parse_invoice(run, c))),
                 len(json.dumps(T.op_match_records(run, c)))]
        assert max(sizes) < 2048, sizes


def test_trace_has_all_steps(results):
    names = [e.name for e in results["c1"].trace if e.kind == "tool_call"]
    for tool in ("fetch_2b_rows", "parse_invoice", "match_records", "validate_gstin", "draft_email", "escalate_case"):
        assert tool in names


def _item(**kw):
    base = dict(invoice_id="x", kind="unmatched", signals={}, suggested_cause="amount_typo")
    return OpenItem(**(base | kw))


@pytest.mark.parametrize("item,cause,conf,refused", [
    (_item(), "amount_typo", 0.9, False),
    (_item(), "amount_typo", 0.5, True),                         # low confidence
    (_item(), "wrong_gstin", 0.95, True),                        # disagrees with evidence
    (_item(suggested_cause="unclear"), "unclear", 0.99, True),   # unclear
    (_item(suggested_cause="unreadable"), "amount_typo", 0.99, True),
    (_item(kind="fuzzy_match_candidate", suggested_cause="unclear"), "amount_typo", 0.99, True),
])
def test_gate(item, cause, conf, refused):
    assert (T.gate(item, cause, conf) is not None) == refused


def test_draft_email_refuses_and_escalate_works():
    run = {}
    T.op_match_records(run, "c1")
    unclear = next(o for o in run["open_items"] if o["suggested_cause"] == "unclear")
    out = T.op_draft_email(run, unclear["invoice_id"], "wrong_gstin", 0.99, "I am sure.")
    assert out["status"] == "refused" and not run.get("emails")
    assert T.op_escalate_case(run, unclear["invoice_id"], "refused", 0.5, "conflicting")["status"] == "escalated"
    assert T.op_escalate_case(run, unclear["invoice_id"], "again", 0.5, "x")["status"] == "refused"


def test_email_numbers_come_from_data_not_llm():
    run = {}
    T.op_match_records(run, "c1")
    typo = next(o for o in run["open_items"] if o["suggested_cause"] == "amount_typo")
    out = T.op_draft_email(run, typo["invoice_id"], "amount_typo", 0.9, "The vendor billed Rs 99,999.00 instead.")
    assert out["status"] == "drafted" and "note" in out
    assert "99,999" not in run["emails"][0]["body"]


def test_llm_only_scoring_catches_failures():
    """Offline check of the experiment's scoring path with a hand-written (fake) model answer."""
    from eval.evaluate import load_key, score
    from experiments.llm_only_matching import build_prompt, parse_json, to_prediction
    prompt, run = build_prompt("c1")
    key = load_key("c1")
    unclear = next(i for i, k in key.items() if k["case"] == "unclear")
    good = next(i for i, k in key.items() if k["case"] == "exact")
    fake = "```json\n" + json.dumps({
        "matches": [{"invoice_id": good, "row_id": key[good]["expected_2b_row_id"]},
                    {"invoice_id": "c1-inv-002", "row_id": "09-999"}],
        "issues": [{"invoice_id": unclear, "cause": "wrong_gstin", "invoice_total": 1.0, "row_total": 2.0}],
    }) + "\n```"
    pred, failures = to_prediction(parse_json(fake), run)
    s = score(key, pred)
    assert s["unsafe_auto_clears"] == 1
    assert {f["type"] for f in failures} == {"hallucinated_row", "copied_amount_wrong"}
    assert s["match"]["precision"] == 1.0 and s["match"]["tp"] == 1


def test_driven_fallback_decides_everything():
    r = run_client("c2", "offline/scripted", save=False, agent_mode="driven")
    decided = {e.invoice_id for e in r.emails} | {f.invoice_id for f in r.flags} | {e.invoice_id for e in r.escalations}
    assert decided == {o.invoice_id for o in r.open_items}
    assert not any(e.reason == "agent_did_not_decide" for e in r.escalations)
    assert r.llm_calls == len(r.open_items)
