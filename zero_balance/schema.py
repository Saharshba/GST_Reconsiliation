from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Cause = Literal["wrong_gstin", "vendor_not_filed", "amount_typo", "period_mismatch", "unclear", "unreadable"]
CAUSES: tuple[str, ...] = ("wrong_gstin", "vendor_not_filed", "amount_typo", "period_mismatch", "unclear", "unreadable")


class Invoice(BaseModel):
    invoice_id: str
    client_id: str
    source_file: str
    supplier_gstin: Optional[str] = None
    supplier_name: Optional[str] = None
    invoice_no: Optional[str] = None
    invoice_no_norm: Optional[str] = None
    invoice_date: Optional[date] = None
    taxable_value: Optional[float] = None
    igst: float = 0.0
    cgst: float = 0.0
    sgst: float = 0.0
    cess: float = 0.0
    total: Optional[float] = None
    parse_status: Literal["ok", "partial", "unreadable"] = "ok"
    parse_notes: str = ""


class TwoBRow(BaseModel):
    row_id: str
    period: str
    supplier_gstin: str
    supplier_name: str = ""
    invoice_no: str
    invoice_no_norm: str
    invoice_date: date
    taxable_value: float
    igst: float = 0.0
    cgst: float = 0.0
    sgst: float = 0.0
    cess: float = 0.0
    total: float
    itc_available: bool = True


class Match(BaseModel):
    invoice_id: str
    row_id: str
    pass_no: Literal[1, 2, 3, 4]
    confidence: float
    notes: str = ""


class OpenItem(BaseModel):
    invoice_id: str
    kind: Literal["unmatched", "fuzzy_match_candidate"] = "unmatched"
    signals: dict[str, Any] = Field(default_factory=dict)
    suggested_cause: Cause
    candidates: list[str] = Field(default_factory=list)


class Diagnosis(BaseModel):
    invoice_id: str
    cause: Cause
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class EmailDraft(BaseModel):
    invoice_id: str
    to_vendor: str
    to_gstin: str = ""
    subject: str
    body: str
    cause: Cause
    highlights: list[str] = Field(default_factory=list)


class Flag(BaseModel):
    invoice_id: str
    action: Literal["flag_next_return"] = "flag_next_return"
    note: str


class Escalation(BaseModel):
    invoice_id: str
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    agent_note: str = ""
    llm_cause: Optional[str] = None
    llm_confidence: Optional[float] = None


class TraceEvent(BaseModel):
    t: float  # seconds since run start
    kind: Literal["user", "llm_text", "tool_call", "tool_result", "info", "warning", "error"]
    name: str = ""
    data: Any = None
    agent: str = ""  # which agent emitted it (multi-agent runs)


class RunResult(BaseModel):
    run_id: str
    client_id: str
    mode: str = "live"
    model: str = ""
    started_at: str
    finished_at: Optional[str] = None
    duration_s: float = 0.0
    invoices: list[Invoice] = Field(default_factory=list)
    rows: list[TwoBRow] = Field(default_factory=list)
    matches: list[Match] = Field(default_factory=list)
    open_items: list[OpenItem] = Field(default_factory=list)
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    emails: list[EmailDraft] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    escalations: list[Escalation] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    llm_calls: int = 0
    summary: str = ""
