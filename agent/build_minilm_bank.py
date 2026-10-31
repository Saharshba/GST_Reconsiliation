"""Build the labelled example bank for the MiniLM nearest-neighbour diagnoser.

Uses data generated with seeds that are NOT the demo seed (42), so evaluation on data/sample stays held out.
    python -m agent.build_minilm_bank
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = Path(__file__).with_name("minilm_bank.json")
TRAIN_SEEDS = (1, 2, 3)


def main():
    from agent import tools as T
    bank = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in TRAIN_SEEDS:
            out = Path(tmp) / f"seed{seed}"
            subprocess.run([sys.executable, str(ROOT / "data" / "generate.py"), "--seed", str(seed), "--out", str(out)],
                           check=True, capture_output=True)
            os.environ["DATA_DIR"] = str(out)
            for c in json.loads((out / "clients.json").read_text()):
                key = {k["invoice_id"]: k for k in json.loads((out / c["client_id"] / "answer_key.json").read_text())}
                for it in T.op_match_records({}, c["client_id"])["open_items"]:
                    bank.append({"text": "; ".join(it["facts"]),
                                 "cause": key[it["invoice_id"]]["expected_cause"] or "unclear",
                                 "source": f"seed{seed}/{it['invoice_id']}"})
    BANK.write_text(json.dumps({"train_seeds": TRAIN_SEEDS, "examples": bank}, indent=1))
    counts = {}
    for b in bank:
        counts[b["cause"]] = counts.get(b["cause"], 0) + 1
    print(f"{len(bank)} examples -> {BANK.name}: {counts}")


if __name__ == "__main__":
    main()
