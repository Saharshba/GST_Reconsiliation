from zero_balance.parsing import parse_text


def _field_ok(got, want):
    if hasattr(got, "isoformat"):
        got = got.isoformat()
    if isinstance(want, float):
        return got is not None and abs(got - want) < 0.005
    if isinstance(want, str) and isinstance(got, str):
        return got.casefold() == want.casefold()
    return got == want


def test_parse_accuracy_at_least_95pct(sample):
    total = ok = 0
    for ds in sample:
        for inv in ds["invoices"]:
            k = ds["key"][inv.invoice_id]
            if k["case"] == "unreadable":
                continue
            for f, want in k["truth"].items():
                total += 1
                ok += _field_ok(getattr(inv, f), want)
    assert ok / total >= 0.95, f"{ok}/{total}"


def test_unreadable_pdfs_are_flagged_not_guessed(sample):
    for ds in sample:
        for inv in ds["invoices"]:
            k = ds["key"][inv.invoice_id]
            assert (inv.parse_status == "unreadable") == (k["case"] == "unreadable"), inv.invoice_id
            if inv.parse_status == "unreadable":
                assert inv.total is None and inv.invoice_no is None and inv.supplier_gstin is None


def test_partial_when_totals_do_not_reconcile():
    text = ("Acme Traders\nGSTIN: 27AAPFU0939F1ZV\nInvoice No: A-1\nInvoice Date: 01-09-2026\n"
            "Taxable Value Rs. 1,000.00\nIGST @ 18% Rs. 180.00\nTotal Rs. 1,500.00\n")
    inv = parse_text(text, invoice_id="x", client_id="c", source_file="x.pdf", client_gstin="29AABCU9603R1ZJ")
    assert inv.parse_status == "partial" and "!=" in inv.parse_notes
    assert inv.supplier_gstin == "27AAPFU0939F1ZV" and inv.invoice_no == "A-1"


def test_buyer_gstin_is_not_taken_as_supplier():
    text = ("Bill To: Client\nGSTIN: 29AABCU9603R1ZJ\nSeller\nGSTIN: 27AAPFU0939F1ZV\nBill No.: 9\n"
            "Dated: 02 Sep 2026\nAssessable Value INR 100.00\nAmount Payable INR 105.00\nIntegrated Tax (5%) INR 5.00")
    inv = parse_text(text, invoice_id="x", client_id="c", source_file="x.pdf", client_gstin="29AABCU9603R1ZJ")
    assert inv.supplier_gstin == "27AAPFU0939F1ZV" and inv.total == 105.0 and inv.igst == 5.0
