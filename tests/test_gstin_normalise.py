from datetime import date

from zero_balance import gstin, normalise as N


def test_known_valid_gstin():
    assert gstin.validate("27AAPFU0939F1ZV")["valid"]


def test_checksum_mismatch_and_format():
    assert gstin.validate("27AAPFU0939F1ZA")["reason"].startswith("checksum")
    assert not gstin.validate("27AAPFU0939F1Z")["valid"]
    assert not gstin.validate("2XAAPFU0939F1ZV")["valid"]


def test_make_gstin_roundtrip():
    g = gstin.make_gstin("29", "ABCDE1234F", "1")
    assert gstin.is_valid(g) and g.startswith("29ABCDE1234F1Z")


def test_invoice_no_normalisation():
    assert N.invoice_no("INV / 2026-27 / 0042") == N.invoice_no("inv_2026/27-42") == "INV-2026-27-42"
    assert N.invoice_no("00042") == "42"
    assert N.invoice_no("GT-1187") != N.invoice_no("GT-1178")


def test_dates_and_amounts():
    for s in ("05-09-2026", "05/09/2026", "2026-09-05", "05 Sep 2026", "Sep 5, 2026"):
        assert N.parse_date(s) == date(2026, 9, 5), s
    assert N.amount("Rs. 1,23,456.70") == 123456.7
    assert N.amount("INR 999.00") == 999.0
    assert N.within_tolerance(18450, 18450.9) and not N.within_tolerance(18450, 18540)
    assert N.period_bounds("092026") == (date(2026, 9, 1), date(2026, 9, 30))
    assert N.next_period("122026") == "012027"
    assert N.fmt_inr(1234567.5) == "12,34,567.50"
