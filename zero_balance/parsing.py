"""Invoice PDF -> Invoice. Rule-based: pdfplumber text + label regexes. Never guesses."""
from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from . import normalise as N
from .gstin import GSTIN_FIND_RE, is_valid
from .schema import Invoice

AMT = r"(?:Rs\.?|INR|₹)?\s*([\d,]+\.\d{2})"
RATE = r"(?:\s*@\s*[\d.]+\s*%|\s*\(\s*[\d.]+\s*%\s*\))?"

INVOICE_NO_RE = re.compile(
    r"(?:Invoice\s*(?:No\.?|Number|#)|Inv\.?\s*(?:#|No\.?)|Bill\s*No\.?)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9/\-_.]*)", re.I)
DATE_RE = re.compile(
    r"(?:Invoice\s*Date|Date\s*of\s*Invoice|Dated|\bDate)\s*[:\-]?\s*"
    r"(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{4})", re.I)

FIELDS: dict[str, list[str]] = {
    "taxable_value": [r"Total\s+Taxable\s+Value", r"Taxable\s+Value", r"Assessable\s+Value", r"Sub\s*total",
                      r"Amount\s+before\s+tax"],
    "igst": [r"(?:Add:\s*)?IGST", r"Integrated\s+Tax"],
    "cgst": [r"(?:Add:\s*)?CGST", r"Central\s+Tax"],
    "sgst": [r"(?:Add:\s*)?SGST", r"State\s+Tax"],
    "cess": [r"(?:Add:\s*)?Cess"],
    "total": [r"Total\s+Invoice\s+Value", r"Grand\s+Total", r"Amount\s+Payable", r"Invoice\s+Total",
              r"^Total(?!\s+Taxable)", r"(?<=\s)Total(?!\s+Taxable)"],
}
HEADER_WORDS = re.compile(r"^(TAX\s+INVOICE|INVOICE|ORIGINAL FOR RECIPIENT|BILL OF SUPPLY)$", re.I)


STRATEGIES = ("default", "layout", "words")


def extract_text(path: str | Path, strategy: str = "default") -> str:
    """default = pdfplumber text; layout = keeps horizontal spacing (helps multi-column pages);
    words = rebuilds lines from individual words (helps odd character spacing)."""
    with pdfplumber.open(str(path)) as pdf:
        if strategy == "layout":
            return "\n".join((p.extract_text(layout=True) or "") for p in pdf.pages)
        if strategy == "words":
            out = []
            for p in pdf.pages:
                lines: dict[int, list] = {}
                for w in p.extract_words(keep_blank_chars=False, x_tolerance=1.5):
                    lines.setdefault(round(w["top"] / 3), []).append(w)
                out += [" ".join(w["text"] for w in sorted(ws, key=lambda w: w["x0"])) for _, ws in sorted(lines.items())]
            return "\n".join(out)
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


def _find_amount(lines: list[str], patterns: list[str]) -> float | None:
    for pat in patterns:
        rx = re.compile(pat + RATE + r"\s*[:\-]?\s*" + AMT, re.I)
        for ln in lines:
            m = rx.search(ln)
            if m:
                return N.amount(m.group(1))
    return None


def _supplier_name(lines: list[str], supplier_gstin: str | None) -> str | None:
    for ln in lines:
        s = ln.strip()
        if not s or HEADER_WORDS.match(s):
            continue
        if re.match(r"^(Sold By|Buyer|Bill To|GSTIN)", s, re.I):
            continue
        # two-column / right-aligned metadata merged onto the name line
        s = re.split(r"\s+(?:Invoice\s*(?:No|Number|Date)|Bill\s*No|Dated)\b", s, flags=re.I)[0]
        if supplier_gstin and supplier_gstin in s:
            continue
        return s.strip() or None
    return None


def parse_text(text: str, *, invoice_id: str, client_id: str, source_file: str, client_gstin: str) -> Invoice:
    notes: list[str] = []
    base = dict(invoice_id=invoice_id, client_id=client_id, source_file=source_file)
    clean = text.strip()
    gstins = [g for g in GSTIN_FIND_RE.findall(clean) if g != client_gstin]
    total_probe = _find_amount(clean.splitlines(), FIELDS["total"])
    if len(re.sub(r"\s", "", clean)) < 40 or (not gstins and total_probe is None):
        reason = "no text layer (image-only scan?)" if len(clean) < 40 else "text is unreadable / garbled"
        return Invoice(**base, parse_status="unreadable", parse_notes=reason)

    lines = [ln for ln in clean.splitlines() if ln.strip()]
    supplier_gstin = gstins[0] if gstins else None
    if supplier_gstin is None:
        notes.append("supplier GSTIN not found")
    elif not is_valid(supplier_gstin):
        notes.append(f"supplier GSTIN {supplier_gstin} fails checksum")

    m = INVOICE_NO_RE.search(clean)
    inv_no = m.group(1).rstrip(".") if m else None
    if not inv_no:
        notes.append("invoice number not found")
    m = DATE_RE.search(clean)
    inv_date = N.parse_date(m.group(1)) if m else None
    if not inv_date:
        notes.append("invoice date not found")

    vals = {k: _find_amount(lines, pats) for k, pats in FIELDS.items()}
    taxable, total = vals["taxable_value"], vals["total"]
    taxes = sum(vals[k] or 0.0 for k in ("igst", "cgst", "sgst", "cess"))
    if taxable is None:
        notes.append("taxable value not found")
    if total is None:
        notes.append("total not found")
    if taxable is not None and total is not None and abs(taxable + taxes - total) > 1.0:
        notes.append(f"taxable {taxable} + tax {taxes:.2f} != total {total}")
    if vals["igst"] and (vals["cgst"] or vals["sgst"]):
        notes.append("both IGST and CGST/SGST present")

    critical_missing = supplier_gstin is None or inv_no is None or total is None
    status = "partial" if notes else "ok"
    if critical_missing and inv_date is None and taxable is None:
        status = "unreadable"
    return Invoice(
        **base,
        supplier_gstin=supplier_gstin,
        supplier_name=_supplier_name(lines, supplier_gstin),
        invoice_no=inv_no,
        invoice_no_norm=N.invoice_no(inv_no),
        invoice_date=inv_date,
        taxable_value=taxable,
        igst=vals["igst"] or 0.0, cgst=vals["cgst"] or 0.0, sgst=vals["sgst"] or 0.0, cess=vals["cess"] or 0.0,
        total=total,
        parse_status=status,
        parse_notes="; ".join(notes),
    )


def parse_pdf(path: str | Path, *, client_id: str, client_gstin: str, strategy: str = "default") -> Invoice:
    path = Path(path)
    try:
        text = extract_text(path, strategy)
    except Exception as e:  # corrupt file: flag it, don't guess
        return Invoice(invoice_id=path.stem, client_id=client_id, source_file=path.name,
                       parse_status="unreadable", parse_notes=f"PDF could not be opened: {type(e).__name__}")
    inv = parse_text(text, invoice_id=path.stem, client_id=client_id, source_file=path.name,
                     client_gstin=client_gstin)
    if strategy != "default":
        inv.parse_notes = f"[{strategy} extraction] " + inv.parse_notes
    return inv


def parse_client(client_dir: str | Path, *, client_id: str, client_gstin: str) -> list[Invoice]:
    files = sorted((Path(client_dir) / "invoices").glob("*.pdf"))
    return [parse_pdf(f, client_id=client_id, client_gstin=client_gstin) for f in files]
