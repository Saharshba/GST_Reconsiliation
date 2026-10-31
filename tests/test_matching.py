import subprocess
import sys

import pytest

from conftest import ROOT, load_dataset
from zero_balance import matching, signals


def check_dataset(datasets):
    for ds in datasets:
        key = ds["key"]
        mr = matching.match(ds["invoices"], ds["rows"], "092026")
        got = {m.invoice_id: (m.row_id, m.pass_no) for m in mr.matches}
        for iid, k in key.items():
            exp = (k["expected_2b_row_id"], k["expected_pass"]) if k["expected_pass"] else None
            assert got.get(iid) == exp, (iid, k["case"], exp, got.get(iid))
        items = signals.build_open_items(ds["invoices"], ds["rows"], ds["next_rows"], mr)
        expected_open = {i for i, k in key.items() if k["expected_cause"] or k["case"] == "near_miss"}
        assert {o.invoice_id for o in items} == expected_open
        for o in items:
            k = key[o.invoice_id]
            assert o.suggested_cause == (k["expected_cause"] or "unclear"), (o.invoice_id, k["case"], o.signals)


def test_sample_matching_and_signals(sample):
    check_dataset(sample)


def test_each_2b_row_matched_once(sample):
    for ds in sample:
        mr = matching.match(ds["invoices"], ds["rows"], "092026")
        rows = [m.row_id for m in mr.matches]
        assert len(rows) == len(set(rows))


@pytest.mark.parametrize("seed", [7, 2027])
def test_other_seeds(tmp_path, seed):
    subprocess.run([sys.executable, str(ROOT / "data" / "generate.py"), "--seed", str(seed), "--out", str(tmp_path)],
                   check=True, capture_output=True)
    check_dataset(load_dataset(tmp_path))
