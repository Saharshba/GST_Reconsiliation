"""Synthetic data generator: invoice PDFs, GSTR-2B JSON (current + next period) and an answer key.

Usage: python data/generate.py [--seed 42] [--out data/sample]

Everything is deterministic for a given seed. Planted cases follow CLAUDE.md §5.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rapidfuzz import fuzz
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zero_balance import gstr2b  # noqa: E402
from zero_balance import normalise as N  # noqa: E402
from zero_balance.gstin import STATES, make_gstin  # noqa: E402

PERIOD = "092026"
NEXT_PERIOD = "102026"
P_START, P_END = date(2026, 9, 1), date(2026, 9, 30)
CLIENT_STATE = "29"

VENDOR_STATES = ["29", "29", "29", "29", "29", "27", "33", "07", "36", "24", "32", "37"]
SUFFIXES = ["Traders", "Enterprises", "Industries", "Supplies", "Pvt Ltd", "& Co", "Agencies", "Distributors"]
CATALOG = {
    5: [("Packaged rice 25kg", "1006"), ("Edible oil 15L tin", "1508"), ("Coir mats", "5702"),
        ("Printed books", "4901"), ("Paper cups (pack of 1000)", "4823"), ("Biomass briquettes", "4401")],
    18: [("Laptop repair service", "998713"), ("LED panel light", "9405"), ("Office chair", "9401"),
         ("Printer toner cartridge", "8443"), ("Annual maintenance contract", "998719"),
         ("Steel fasteners (box)", "7318"), ("Courier services", "996812"), ("Software subscription", "997331"),
         ("Copper wire 90m coil", "7408"), ("Lubricant oil 5L", "2710")],
}

# Invoice number styles. fmt(n) -> the invoice number printed on the PDF.
STYLES = [
    ("fy_slash", lambda p, n: f"INV/2026-27/{n:04d}"),
    ("prefix_dash", lambda p, n: f"{p}-{n}"),
    ("zero_pad", lambda p, n: f"{n:05d}"),
    ("prefix_fy", lambda p, n: f"{p}/{n:04d}/26-27"),
    ("long_prefix", lambda p, n: f"{p}2627{n:05d}"),
]


@dataclass
class Vendor:
    name: str
    gstin: str
    pan: str
    state: str
    address: str
    prefix: str
    style: str
    rate: int
    layout: int
    counter: int

    def fmt(self, n: int) -> str:
        return dict(STYLES)[self.style](self.prefix, n)


@dataclass
class Inv:
    invoice_id: str
    vendor: Vendor
    number: str
    dt: date
    items: list  # (desc, hsn, qty, rate, amount)
    taxable: float
    igst: float
    cgst: float
    sgst: float
    total: float
    n: int = 0
    case: str = "exact"
    # what goes into 2B
    tb_period: str | None = PERIOD
    tb_ctin: str | None = None
    tb_number: str | None = None
    tb_date: date | None = None
    tb_taxable: float | None = None
    tb_total: float | None = None
    unreadable_mode: str | None = None
    extra_rows: list = field(default_factory=list)  # (period, ctin, number, date, taxable, total)


def taxes(taxable: float, rate: int, inter: bool) -> tuple[float, float, float]:
    tax = round(taxable * rate / 100, 2)
    if inter:
        return tax, 0.0, 0.0
    c = round(tax / 2, 2)
    return 0.0, c, round(tax - c, 2)


def pan(rng: random.Random, kind: str) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(letters) for _ in range(3)) + kind + rng.choice(letters) + \
        f"{rng.randint(1000, 9999)}" + rng.choice(letters)


def ascii_addr(fake: Faker) -> str:
    return fake.address().replace("\n", ", ").encode("ascii", "ignore").decode()


def make_items(rng: random.Random, rate: int, target: float | None = None):
    items = []
    for desc, hsn in rng.sample(CATALOG[rate], rng.randint(1, 3)):
        qty = rng.randint(1, 40)
        unit = round(rng.uniform(150, 6000), 0)
        items.append((desc, hsn, qty, unit, round(qty * unit, 2)))
    if target is not None:  # single line with exact amount (for same-amount siblings)
        items = [(items[0][0], items[0][1], 1, target, target)]
    return items


# ---------------------------------------------------------------- planting helpers

def number_format_variant(number: str) -> str | None:
    """Same normalised key, different raw string (case / spaces / leading zeros)."""
    cands = []
    if any(c.isalpha() for c in number):
        cands.append(number.lower())
    if "/" in number or "-" in number:
        cands.append(number.replace("/", " / ").replace("-", " - "))
    import re
    m = list(re.finditer(r"\d+", number))
    if m:
        last = m[-1]
        seg = last.group()
        alt = seg.lstrip("0") or "0"
        if alt == seg:
            alt = "00" + seg
        cands.append(number[:last.start()] + alt + number[last.end():])
    for c in cands:
        if c != number and N.invoice_no(c) == N.invoice_no(number):
            return c
    return None


def drift_variant(vendor: Vendor, n: int) -> str:
    """Very different format of the same invoice number (defeats passes 1, 2 and 4)."""
    for cand in (f"{n}", f"FY2627/{vendor.prefix}{n}", f"TI{n:07d}", f"{vendor.prefix}{n}/26"):
        a, b = N.invoice_no(cand), N.invoice_no(vendor.fmt(n))
        if a != b and fuzz.ratio(a, b) < 85:
            return cand
    return f"X{n}-{vendor.prefix}-SEP"


def near_miss_variant(number: str, taken: set[str] = frozenset()) -> str | None:
    """Transpose two adjacent digits in the last numeric run, keeping fuzzy ratio >= 85."""
    import re
    runs = list(re.finditer(r"\d+", number))
    if not runs:
        return None
    run = runs[-1]
    s = run.group()
    for i in range(len(s) - 1, 0, -1):
        if s[i] != s[i - 1]:
            t = s[:i - 1] + s[i] + s[i - 1] + s[i + 1:]
            cand = number[:run.start()] + t + number[run.end():]
            if N.invoice_no(cand) != N.invoice_no(number) and N.invoice_no(cand) not in taken and \
                    fuzz.ratio(N.invoice_no(cand), N.invoice_no(number)) >= 85:
                return cand
    return None


def typo_taxable(taxable: float, rate: int) -> float | None:
    """Swap two adjacent digits of the integer part so the total moves well beyond tolerance."""
    whole, frac = f"{taxable:.2f}".split(".")
    for i in range(len(whole) - 1):
        if whole[i] != whole[i + 1] and i + 1 < len(whole):
            t = whole[:i] + whole[i + 1] + whole[i] + whole[i + 2:]
            if t[0] == "0":
                continue
            new = float(f"{t}.{frac}")
            gap = abs(new - taxable) * (1 + rate / 100)
            if gap > 3 * N.tolerance(taxable * (1 + rate / 100)) and abs(new - taxable) / taxable < 0.2:
                return new
    return None


# ---------------------------------------------------------------- client generation

def gen_client(idx: int, seed: int) -> dict:
    rng = random.Random(seed * 100 + idx)
    fake = Faker("en_IN")
    fake.seed_instance(seed * 100 + idx)
    cid = f"c{idx}"
    client = {
        "client_id": cid,
        "name": fake.company().encode("ascii", "ignore").decode() + " " + rng.choice(["Pvt Ltd", "LLP"]),
        "gstin": make_gstin(CLIENT_STATE, pan(rng, "C")),
        "state_code": CLIENT_STATE,
        "city": rng.choice(["Bengaluru", "Mysuru", "Hubballi", "Mangaluru"]),
        "period": PERIOD,
    }

    vendors: list[Vendor] = []
    used_prefixes = set()
    for v in range(12):
        st = VENDOR_STATES[v]
        name = fake.last_name().encode("ascii", "ignore").decode() + " " + rng.choice(SUFFIXES)
        prefix = "".join(w[0] for w in name.split() if w[0].isalpha()).upper()[:3]
        while prefix in used_prefixes or len(prefix) < 2:
            prefix = prefix + rng.choice("KMRTS")
        used_prefixes.add(prefix)
        p = pan(rng, rng.choice("CFP"))
        vendors.append(Vendor(
            name=name, gstin=make_gstin(st, p), pan=p, state=st, address=ascii_addr(fake),
            prefix=prefix, style=STYLES[(v + idx) % len(STYLES)][0], rate=rng.choice([5, 18, 18]),
            layout=(v + idx) % 4, counter=rng.randint(20, 1400),
        ))

    target_n = rng.randint(28, 36)
    vnf_vendor = vendors[11]
    per_vendor = {id(v): 1 for v in vendors}
    per_vendor[id(vnf_vendor)] = 1
    others = vendors[:11]
    for v in others:
        per_vendor[id(v)] = 2
    while sum(per_vendor.values()) < target_n:
        per_vendor[id(rng.choice(others))] += 1

    invoices: list[Inv] = []
    for v in vendors:
        days = sorted(rng.sample(range(1, 26), per_vendor[id(v)]))
        for d in days:
            v.counter += rng.randint(1, 12)
            items = make_items(rng, v.rate)
            taxable = round(sum(i[4] for i in items), 2)
            ig, cg, sg = taxes(taxable, v.rate, v.state != CLIENT_STATE)
            invoices.append(Inv(
                invoice_id="", vendor=v, number=v.fmt(v.counter), dt=date(2026, 9, d), items=items,
                taxable=taxable, igst=ig, cgst=cg, sgst=sg, total=round(taxable + ig + cg + sg, 2), n=v.counter,
            ))

    def pick(pred, exclude_vendors=()):
        pool = [i for i in invoices if i.case == "exact" and pred(i) and i.vendor not in exclude_vendors]
        return rng.choice(pool)

    by_vendor = lambda v: [i for i in invoices if i.vendor is v]  # noqa: E731
    used_vendors: list[Vendor] = [vnf_vendor]

    # vendor not filed
    for i in by_vendor(vnf_vendor):
        i.case, i.tb_period = "vendor_not_filed", None

    # near-miss: a vendor billing the same amount twice; one of the two numbers is mistyped in 2B.
    def taken_for(v: Vendor) -> set[str]:  # existing + future (sibling, October) numbers of this vendor
        return {N.invoice_no(i.number) for i in invoices if i.vendor is v} | \
            {N.invoice_no(v.fmt(n)) for n in range(v.counter, v.counter + 40)}

    nm = pick(lambda i: near_miss_variant(i.number, taken_for(i.vendor)) is not None, used_vendors)
    v = nm.vendor
    v.counter += 1
    sib_items = make_items(rng, v.rate, target=nm.taxable)
    sib = Inv(invoice_id="", vendor=v, number=v.fmt(v.counter), dt=min(nm.dt + timedelta(days=2), date(2026, 9, 27)),
              items=sib_items, taxable=nm.taxable, igst=nm.igst, cgst=nm.cgst, sgst=nm.sgst, total=nm.total,
              n=v.counter)
    invoices.append(sib)
    nm.case, nm.tb_number = "near_miss", near_miss_variant(nm.number, taken_for(v))
    assert nm.tb_number, "no safe near-miss variant"
    used_vendors.append(v)

    # rounding / format: one amount off by <= Rs 1, one number formatted differently
    r1 = pick(lambda i: True, used_vendors)
    r1.case, r1.tb_total = "rounding", round(r1.total + rng.choice([-1, 1]) * rng.choice([0.4, 0.6, 1.0]), 2)
    r2 = pick(lambda i: number_format_variant(i.number) is not None, used_vendors + [r1.vendor])
    r2.case, r2.tb_number = "rounding", number_format_variant(r2.number)

    # date drift (2 instances for c2)
    for _ in range(2 if idx == 2 else 1):
        dd = pick(lambda i: True, used_vendors)
        dd.case = "date_drift"
        dd.tb_number = drift_variant(dd.vendor, dd.n)
        shift = rng.choice([-3, -2, -1, 1, 2, 3])
        dd.tb_date = min(max(dd.dt + timedelta(days=shift), P_START), P_END)
        used_vendors.append(dd.vendor)

    multi = lambda i: len(by_vendor(i.vendor)) >= 2  # noqa: E731

    def other_gstin(v: Vendor) -> str:
        alt_state = rng.choice([s for s in STATES if s != v.state])
        return make_gstin(alt_state, v.pan, rng.choice("23"))

    # wrong GSTIN: same number + amount, reported under another (valid) GSTIN
    wg = pick(multi, used_vendors)
    wg.case, wg.tb_ctin = "wrong_gstin", other_gstin(wg.vendor)
    used_vendors.append(wg.vendor)

    # amount typo
    at = pick(lambda i: multi(i) and typo_taxable(i.taxable, i.vendor.rate) is not None, used_vendors)
    at.case = "amount_typo"
    at.tb_taxable = typo_taxable(at.taxable, at.vendor.rate)
    used_vendors.append(at.vendor)

    # period mismatch: late-September invoice the vendor reported in October
    pm = pick(multi, used_vendors)
    pm.case, pm.dt, pm.tb_period = "period_mismatch", date(2026, 9, rng.randint(27, 30)), NEXT_PERIOD
    used_vendors.append(pm.vendor)

    # unclear: conflicting signals (variant depends on client)
    uc = pick(multi, used_vendors)
    uc.case = "unclear"
    if idx == 1:  # same number under another GSTIN, but a different amount
        uc.tb_ctin = other_gstin(uc.vendor)
        uc.tb_taxable = round(uc.taxable * 0.87, 2)
    elif idx == 2:  # same no+amount under another GSTIN AND in next period under the right GSTIN
        uc.tb_ctin = other_gstin(uc.vendor)
        uc.extra_rows.append((NEXT_PERIOD, uc.vendor.gstin, uc.number, uc.dt, uc.taxable, uc.total))
    else:  # amount gap this period AND an exact copy in next period
        uc.tb_taxable = typo_taxable(uc.taxable, uc.vendor.rate) or round(uc.taxable * 1.1, 2)
        uc.extra_rows.append((NEXT_PERIOD, uc.vendor.gstin, uc.number, uc.dt, uc.taxable, uc.total))
    used_vendors.append(uc.vendor)

    # unreadable PDF (vendor did file it, so its 2B row stays unmatched)
    ur = pick(multi, used_vendors)
    ur.case, ur.unreadable_mode = "unreadable", ("garbled" if idx == 2 else "image")

    # ids in date order
    invoices.sort(key=lambda i: (i.dt, i.vendor.name))
    for n, inv in enumerate(invoices, 1):
        inv.invoice_id = f"{cid}-inv-{n:03d}"
    return {"client": client, "vendors": vendors, "invoices": invoices}


# ---------------------------------------------------------------- 2B JSON

def build_2b(client: dict, vendors: list[Vendor], invoices: list[Inv], period: str, rng: random.Random) -> dict:
    sup: dict[str, dict] = {}

    def add_row(ctin, trdnm, number, dt, taxable, total, inter, rate):
        ig, cg, sg = taxes(taxable, rate, inter)
        tax = ig + cg + sg
        if total is None:
            total = round(taxable + tax, 2)
        entry = sup.setdefault(ctin, {"ctin": ctin, "trdnm": trdnm, "supfildt": "", "supprd": period, "inv": []})
        file_day = 11 if period == PERIOD else 10
        entry["supfildt"] = f"{file_day:02d}-{int(period[:2]) + 1:02d}-2026"
        entry["inv"].append({
            "inum": number, "typ": "R", "dt": dt.strftime("%d-%m-%Y"), "val": round(total, 2),
            "pos": CLIENT_STATE, "rev": "N", "itcavl": "Y", "rsn": "", "diffprcnt": 1, "srctyp": "",
            "txval": round(taxable, 2), "igst": ig, "cgst": cg, "sgst": sg, "cess": 0,
            "items": [{"num": 1, "rt": rate, "txval": round(taxable, 2), "igst": ig, "cgst": cg, "sgst": sg, "cess": 0}],
        })

    for inv in invoices:
        v = inv.vendor
        inter = v.state != CLIENT_STATE
        if inv.tb_period == period:
            ctin = inv.tb_ctin or v.gstin
            taxable = inv.tb_taxable if inv.tb_taxable is not None else inv.taxable
            total = inv.tb_total if inv.tb_total is not None else (None if inv.tb_taxable is not None else inv.total)
            add_row(ctin, v.name.upper(), inv.tb_number or inv.number, inv.tb_date or inv.dt, taxable, total,
                    ctin[:2] != CLIENT_STATE, v.rate)
        for (p, ctin, number, dt, taxable, total) in inv.extra_rows:
            if p == period:
                add_row(ctin, v.name.upper(), number, dt, taxable, total, inter, v.rate)

    if period == NEXT_PERIOD:  # regular October business for vendors that file
        filers = {i.vendor.gstin: i.vendor for i in invoices if i.case != "vendor_not_filed"}
        for v in filers.values():
            for _ in range(rng.randint(1, 2)):
                v.counter += rng.randint(1, 12)
                taxable = round(sum(i[4] for i in make_items(rng, v.rate)), 2)
                add_row(v.gstin, v.name.upper(), v.fmt(v.counter), date(2026, 10, rng.randint(1, 28)),
                        taxable, None, v.state != CLIENT_STATE, v.rate)

    # supplier order: sorted by ctin, like the portal export
    b2b = [sup[k] for k in sorted(sup)]
    return {
        "chksum": "synthetic",
        "data": {
            "gstin": client["gstin"], "rtnprd": period, "version": "1.0",
            "gendt": f"14-{int(period[:2]) + 1:02d}-2026",
            "docdata": {"b2b": b2b},
        },
    }


# ---------------------------------------------------------------- PDF layouts

def _money(x: float) -> str:
    return N.fmt_inr(x)


def _tax_lines(inv: Inv, style: str) -> list[tuple[str, float]]:
    r = inv.vendor.rate
    names = {
        "short": ("IGST @ {r}%", "CGST @ {h}%", "SGST @ {h}%"),
        "long": ("Integrated Tax ({r}%)", "Central Tax ({h}%)", "State Tax ({h}%)"),
        "add": ("Add: IGST @ {r}%", "Add: CGST @ {h}%", "Add: SGST @ {h}%"),
        "plain": ("IGST", "CGST", "SGST"),
    }[style]
    h = f"{r / 2:g}"
    if inv.igst:
        return [(names[0].format(r=r, h=h), inv.igst)]
    return [(names[1].format(r=r, h=h), inv.cgst), (names[2].format(r=r, h=h), inv.sgst)]


def _items_table(c, inv: Inv, y: float, x0=40, widths=(250, 70, 40, 75, 80)) -> float:
    heads = ["Description", "HSN/SAC", "Qty", "Rate", "Amount"]
    c.setFont("Helvetica-Bold", 9)
    xs = [x0]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    right = x0 + sum(widths)
    c.line(x0, y + 12, right, y + 12)
    for x, h in zip(xs, heads):
        c.drawString(x + 3, y, h)
    c.line(x0, y - 4, right, y - 4)
    c.setFont("Helvetica", 9)
    for desc, hsn, qty, rate, amt in inv.items:
        y -= 16
        c.drawString(xs[0] + 3, y, desc)
        c.drawString(xs[1] + 3, y, hsn)
        c.drawRightString(xs[3] - 5, y, str(qty))
        c.drawRightString(xs[4] - 5, y, _money(rate))
        c.drawRightString(right - 5, y, _money(amt))
    c.line(x0, y - 6, right, y - 6)
    return y - 24


def layout_table(c, inv: Inv, client: dict):
    v = inv.vendor
    W, H = A4
    c.setFont("Helvetica-Bold", 15)
    c.drawString(40, H - 60, v.name)
    c.setFont("Helvetica", 9)
    c.drawString(40, H - 76, v.address[:80])
    c.drawString(40, H - 90, f"GSTIN: {v.gstin}")
    c.drawString(360, H - 76, f"Invoice No: {inv.number}")
    c.drawString(360, H - 90, f"Invoice Date: {inv.dt.strftime('%d-%m-%Y')}")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, H - 120, "Bill To:")
    c.setFont("Helvetica", 9)
    c.drawString(40, H - 134, client["name"])
    c.drawString(40, H - 148, f"GSTIN: {client['gstin']}    Place of Supply: {STATES[CLIENT_STATE]} ({CLIENT_STATE})")
    y = _items_table(c, inv, H - 185)
    rows = [("Taxable Value", inv.taxable)] + _tax_lines(inv, "short") + [("Total", inv.total)]
    for label, val in rows:
        c.setFont("Helvetica-Bold" if label == "Total" else "Helvetica", 10)
        c.drawString(360, y, label)
        c.drawRightString(555, y, f"Rs. {_money(val)}")
        y -= 16
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(40, 60, "This is a computer generated invoice.")


def layout_two_col(c, inv: Inv, client: dict):
    v = inv.vendor
    W, H = A4
    c.setFillGray(0.92)
    c.rect(30, H - 70, W - 60, 40, fill=1, stroke=0)
    c.setFillGray(0)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, H - 55, v.name.upper())
    c.setFont("Helvetica", 9)
    left, right = 40, 320
    y = H - 95
    for l, r in [
        ("Sold By", "Buyer (Bill to)"),
        (v.address[:48], client["name"]),
        (f"GSTIN/UIN: {v.gstin}", f"GSTIN/UIN: {client['gstin']}"),
        (f"State: {STATES[v.state]}, Code: {v.state}", f"State: {STATES[CLIENT_STATE]}, Code: {CLIENT_STATE}"),
        (f"Bill No.: {inv.number}", f"Dated: {inv.dt.strftime('%d %b %Y')}"),
    ]:
        c.drawString(left, y, l)
        c.drawString(right, y, r)
        y -= 15
    y = _items_table(c, inv, y - 20, widths=(240, 70, 40, 80, 85))
    rows = [("Assessable Value", inv.taxable)] + _tax_lines(inv, "long") + [("Amount Payable", inv.total)]
    c.drawString(left, y, "Terms: Payment within 30 days.")
    for label, val in rows:
        c.drawString(right, y, label)
        c.drawRightString(555, y, f"INR {_money(val)}")
        y -= 15


def layout_minimal(c, inv: Inv, client: dict):
    v = inv.vendor
    W, H = A4
    t = c.beginText(60, H - 70)
    t.setFont("Courier", 10)
    lines = [
        v.name,
        f"GSTIN {v.gstin}",
        "",
        f"Inv # {inv.number}      Date {inv.dt.strftime('%d/%m/%Y')}",
        f"To: {client['name']} ({client['gstin']})",
        "",
    ]
    for desc, hsn, qty, rate, amt in inv.items:
        lines.append(f"{qty} x {desc} @ {rate:.2f} = {amt:.2f}")
    lines += ["", f"Subtotal {inv.taxable:.2f}"]
    lines += [f"{lab} {val:.2f}" for lab, val in _tax_lines(inv, "plain")]
    lines += [f"Grand Total {inv.total:.2f}", "", "Thank you for your business."]
    for ln in lines:
        t.textLine(ln)
    c.drawText(t)


def layout_letterhead(c, inv: Inv, client: dict):
    v = inv.vendor
    W, H = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(W / 2, H - 50, "TAX INVOICE")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(W / 2, H - 72, v.name)
    c.setFont("Helvetica", 9)
    c.drawCentredString(W / 2, H - 86, v.address[:90])
    c.drawCentredString(W / 2, H - 100, f"GSTIN: {v.gstin} | PAN: {v.pan}")
    c.line(40, H - 110, W - 40, H - 110)
    c.drawString(40, H - 128, f"Invoice Number: {inv.number}")
    c.drawString(40, H - 142, f"Date of Invoice: {inv.dt.isoformat()}")
    c.drawString(40, H - 156, f"Place of Supply: {STATES[CLIENT_STATE]} ({CLIENT_STATE})")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, H - 180, "Details of Receiver (Billed to)")
    c.setFont("Helvetica", 9)
    c.drawString(40, H - 194, f"{client['name']}, {client['city']}")
    c.drawString(40, H - 208, f"GSTIN: {client['gstin']}")
    y = _items_table(c, inv, H - 240)
    rows = [("Total Taxable Value", inv.taxable)] + _tax_lines(inv, "add") + [("Cess", 0.0), ("Total Invoice Value", inv.total)]
    for label, val in rows:
        c.drawString(330, y, label)
        c.drawRightString(555, y, _money(val))
        y -= 15
    c.drawString(40, y - 10, "Amount in words: As per total above. E. & O.E.")
    c.drawString(400, 80, f"For {v.name}")
    c.drawString(400, 50, "Authorised Signatory")


LAYOUTS = [layout_table, layout_two_col, layout_minimal, layout_letterhead]


def render_unreadable(c, inv: Inv, client: dict, rng: random.Random):
    W, H = A4
    if inv.unreadable_mode == "garbled":
        # text layer exists but is mojibake (broken font encoding)
        c.setFont("Helvetica", 10)
        y = H - 70
        for _ in range(28):
            c.drawString(50, y, "".join(rng.choice("ÃÂ¤§¶¿ÐÞæøþ�#@%&*~^") for _ in range(rng.randint(20, 70))))
            y -= 16
        return
    # image-only scan: faint, rotated, blurred raster with no text layer
    img = Image.new("L", (900, 1270), 235)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    txt = [inv.vendor.name, f"GSTIN {inv.vendor.gstin}", f"Invoice {inv.number}", inv.dt.isoformat(),
           f"Total {inv.total:.2f}"]
    for i, t in enumerate(txt):
        d.text((60, 80 + 50 * i), t, fill=150, font=font)
    for _ in range(9000):
        x, y = rng.randrange(900), rng.randrange(1270)
        d.point((x, y), fill=rng.randint(60, 200))
    img = img.rotate(rng.uniform(-6, 6), fillcolor=220).filter(ImageFilter.GaussianBlur(3.2))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    c.drawImage(ImageReader(buf), 0, 0, W, H)


def write_pdf(path: Path, inv: Inv, client: dict, rng: random.Random):
    c = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    c.setTitle(f"Invoice {inv.invoice_id}")
    if inv.unreadable_mode:
        render_unreadable(c, inv, client, rng)
    else:
        LAYOUTS[inv.vendor.layout](c, inv, client)
    c.showPage()
    c.save()


# ---------------------------------------------------------------- answer key

EXPECT = {
    "exact": (1, None, "none"),
    "rounding": (2, None, "none"),
    "date_drift": (3, None, "none"),
    "near_miss": (4, None, "escalate"),
    "wrong_gstin": (None, "wrong_gstin", "email_correction"),
    "vendor_not_filed": (None, "vendor_not_filed", "email_filing_reminder"),
    "amount_typo": (None, "amount_typo", "email_amount_query"),
    "period_mismatch": (None, "period_mismatch", "flag_next_return"),
    "unclear": (None, "unclear", "escalate"),
    "unreadable": (None, "unreadable", "escalate"),
}


def answer_key(invoices: list[Inv], rows: list) -> list[dict]:
    idx = {(r.supplier_gstin, r.invoice_no): r.row_id for r in rows}
    assert len(idx) == len(rows), "duplicate (ctin, inum) in 2B"
    out = []
    for inv in invoices:
        p, cause, action = EXPECT[inv.case]
        row = idx.get((inv.tb_ctin or inv.vendor.gstin, inv.tb_number or inv.number)) if p else None
        if p and row is None:
            raise RuntimeError(f"no 2B row for matched case {inv.invoice_id}")
        out.append({
            "invoice_id": inv.invoice_id, "case": inv.case, "supplier_gstin": inv.vendor.gstin,
            "invoice_no": inv.number, "total": inv.total,
            "truth": {"supplier_gstin": inv.vendor.gstin, "supplier_name": inv.vendor.name,
                      "invoice_no": inv.number, "invoice_date": inv.dt.isoformat(), "taxable_value": inv.taxable,
                      "igst": inv.igst, "cgst": inv.cgst, "sgst": inv.sgst, "total": inv.total},
            "expected_2b_row_id": row, "expected_pass": p, "expected_cause": cause,
            "expected_action": action,
            "acceptable_actions": ["none", "escalate"] if inv.case == "near_miss" else [action],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "sample"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    clients = []
    for idx in (1, 2, 3):
        g = gen_client(idx, args.seed)
        client, vendors, invoices = g["client"], g["vendors"], g["invoices"]
        cdir = out / client["client_id"]
        (cdir / "invoices").mkdir(parents=True)
        rng = random.Random(args.seed * 1000 + idx)
        for inv in invoices:
            write_pdf(cdir / "invoices" / f"{inv.invoice_id}.pdf", inv, client, rng)
        for period in (PERIOD, NEXT_PERIOD):
            doc = build_2b(client, vendors, invoices, period, rng)
            (cdir / f"gstr2b_{period}.json").write_text(json.dumps(doc, indent=1))
        rows = gstr2b.load(cdir / f"gstr2b_{PERIOD}.json")
        key = answer_key(invoices, rows)
        (cdir / "answer_key.json").write_text(json.dumps(key, indent=1))
        client["vendors"] = [{"name": v.name, "gstin": v.gstin, "state": v.state} for v in vendors]
        client["n_invoices"] = len(invoices)
        clients.append(client)
        counts: dict[str, int] = {}
        for k in key:
            counts[k["case"]] = counts.get(k["case"], 0) + 1
        print(client["client_id"], client["name"], len(invoices), "invoices", counts)
    (out / "clients.json").write_text(json.dumps(clients, indent=1))


if __name__ == "__main__":
    main()
