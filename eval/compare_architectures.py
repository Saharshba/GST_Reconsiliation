"""Compare the single-agent and multi-agent designs (and diagnosis strategy orders) on the same data.

    python eval/compare_architectures.py

Every configuration runs all 3 clients in a fresh process and is scored with eval/evaluate.py's scorer.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.evaluate import aggregate, load_key, prediction_from_run, score, write_notes_section  # noqa: E402
from zero_balance.schema import RunResult  # noqa: E402

CONFIGS = [
    ("Single agent · MiniLM few-shot", ["-m", "agent.runner"], {"MINILM_MODE": "fewshot"}),
    ("Single agent · MiniLM zero-shot", ["-m", "agent.runner"], {"MINILM_MODE": "zeroshot"}),
    ("Multi-agent · zero-shot only (no retry)", ["-m", "multiagent.runner"], {"DIAG_STRATEGIES": "zeroshot"}),
    ("Multi-agent · few-shot only", ["-m", "multiagent.runner"], {"DIAG_STRATEGIES": "fewshot"}),
    ("Multi-agent · zero-shot → few-shot (default)", ["-m", "multiagent.runner"], {"DIAG_STRATEGIES": "zeroshot,fewshot"}),
]


def latest(client: str) -> RunResult:
    f = max(glob.glob(str(ROOT / "runs" / f"{client}_2*.json")), key=os.path.getmtime)
    return RunResult.model_validate_json(Path(f).read_text())


def rounds(r: RunResult) -> int:
    import re
    found = [int(m.group(1)) for e in r.trace if e.agent == "diagnosis" and isinstance(e.data, str)
             for m in [re.match(r"round (\d+)", e.data)] if m]
    return max(found, default=1)


def main():
    rows, out = [], {}
    for label, mod, env in CONFIGS:
        per, extra = {}, {"rounds": [], "steps": []}
        for c in ("c1", "c2", "c3"):
            subprocess.run([sys.executable, *mod, "--client", c], cwd=ROOT, check=True, capture_output=True,
                           env=os.environ | {"HF_HUB_OFFLINE": "1"} | env)
            r = latest(c)
            per[c] = score(load_key(c), prediction_from_run(r))
            extra["rounds"].append(rounds(r))
            extra["steps"].append(sum(1 for e in r.trace if e.kind == "tool_call"))
        agg = aggregate(per)
        out[label] = {"aggregate": agg, "per_client": per, **extra}
        rows.append(f"| {label} | {agg['match']['precision']:.0%} / {agg['match']['recall']:.0%} | "
                    f"{agg['diagnosis']['accuracy']:.1%} | **{agg['unsafe_auto_clears']}** | "
                    f"{agg['escalation_rate']:.0%} | {agg['llm_calls_per_client']:.1f} | "
                    f"{sum(extra['steps']) / 3:.0f} | {max(extra['rounds'])} | {agg['minutes_per_client'] * 60:.1f}s |")
        print(rows[-1], flush=True)
    (ROOT / "eval" / "architecture_comparison.json").write_text(json.dumps(out, indent=1))
    table = ["### Architecture comparison (3 clients, demo data)", "",
             "| Configuration | Match P / R | Diagnosis acc. | Unsafe auto-clears | Escalation rate | "
             "Model calls / client | Tool calls / client | Diagnosis rounds | Time / client |",
             "|---|---|---|---|---|---|---|---|---|", *rows, ""]
    write_notes_section("ARCH-COMPARE", "\n".join(table))


if __name__ == "__main__":
    main()
