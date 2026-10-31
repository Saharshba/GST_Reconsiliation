"""GSTR-2B JSON (B2B section) -> TwoBRow."""
from __future__ import annotations

import json
from pathlib import Path

from . import normalise as N
from .schema import TwoBRow


def load(path: str | Path) -> list[TwoBRow]:
    doc = json.loads(Path(path).read_text())
    data = doc.get("data", doc)
    period = data["rtnprd"]
    rows: list[TwoBRow] = []
    for sup in data.get("docdata", {}).get("b2b", []):
        for inv in sup.get("inv", []):
            rows.append(TwoBRow(
                row_id=f"{period[:2]}-{len(rows) + 1:03d}",
                period=period,
                supplier_gstin=sup["ctin"],
                supplier_name=sup.get("trdnm", ""),
                invoice_no=inv["inum"],
                invoice_no_norm=N.invoice_no(inv["inum"]),
                invoice_date=N.parse_date(inv["dt"]),
                taxable_value=N.amount(inv.get("txval", 0)),
                igst=N.amount(inv.get("igst", 0)),
                cgst=N.amount(inv.get("cgst", 0)),
                sgst=N.amount(inv.get("sgst", 0)),
                cess=N.amount(inv.get("cess", 0)),
                total=N.amount(inv["val"]),
                itc_available=inv.get("itcavl", "Y") == "Y",
            ))
    return rows


def path_for(client_dir: str | Path, period: str) -> Path:
    return Path(client_dir) / f"gstr2b_{period}.json"
