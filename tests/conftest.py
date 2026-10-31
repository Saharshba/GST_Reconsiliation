import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLE = ROOT / "data" / "sample"


def load_dataset(base: Path):
    from zero_balance import gstr2b
    from zero_balance.parsing import parse_client
    out = []
    for c in json.loads((base / "clients.json").read_text()):
        d = base / c["client_id"]
        out.append({
            "client": c,
            "key": {k["invoice_id"]: k for k in json.loads((d / "answer_key.json").read_text())},
            "invoices": parse_client(d, client_id=c["client_id"], client_gstin=c["gstin"]),
            "rows": gstr2b.load(d / "gstr2b_092026.json"),
            "next_rows": gstr2b.load(d / "gstr2b_102026.json"),
        })
    return out


@pytest.fixture(scope="session")
def sample():
    return load_dataset(SAMPLE)
