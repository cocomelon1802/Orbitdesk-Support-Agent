"""
Shared graph state.

Every node reads from and writes to this typed state. Keeping it as a
single TypedDict (rather than passing loose args between functions) is
what lets LangGraph merge partial updates from each node and is what the
assignment calls out explicitly as a requirement ("shared typed state").
"""
from __future__ import annotations

import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated


class RetrievedChunk(TypedDict):
    source_id: str          # KB-00X or CASE-XXXX
    passage: str             # excerpt text
    score: float              # similarity score
    doc_status: str            # "current" | "resolved" | "escalated" | "superseded"


class GraphState(TypedDict, total=False):
    # ---- input ----
    question: str
    question_id: str

    # ---- triage ----
    classification: str            # answerable | requires_clarification | requires_escalation | out_of_scope
    triage_reason: str
    clarification_question: Optional[str]

    # ---- retrieval ----
    retrieved: List[RetrievedChunk]

    # ---- generation ----
    draft_answer: str
    draft_sources: List[Dict[str, str]]
    draft_confidence: float

    # ---- verification ----
    verification_passed: bool
    verification_notes: List[str]
    retry_count: int

    # ---- final structured output (matches output_schema.json) ----
    final_response: Dict[str, Any]

    # ---- observability ----
    # Annotated with operator.add so each node's returned {"node_trace": [name]}
    # is APPENDED to the running list by LangGraph's reducer, rather than
    # overwriting it (the default behaviour for plain TypedDict fields).
    node_trace: Annotated[List[str], operator.add]
    warnings: List[str]
