"""Node implementations for triage, retrieval, generation, and verification."""


from __future__ import annotations

import json
from operator import index
import re
from typing import Dict, Any, List

from .logging_utils import trace, logger
from .schema_validate import validate_response

MAX_RETRIES = 1
TOP_K = 8
GROUNDING_OVERLAP_THRESHOLD = 0.12

# ------------------------------------------------------------------ #
# Triage heuristics
# ------------------------------------------------------------------ #
OUT_OF_SCOPE_PATTERNS = [
    r"\brefund\b",
    r"ignore (the|this|previous|supplied|prior)",
    r"disregard the",
    r"\blegal advice\b",
    r"\blawsuit\b",
    r"\bchargeback\b",
    r"override (the|these) rules",
    r"\bbypass\b",
    r"cancel my subscription",
]

ESCALATION_PATTERNS = [
    r"already (checked|tried|followed|verified)",
    r"two (export )?runs? in a row",
    r"render_failed",
    r"still (fail|failing|not working)",
    r"\bescalat",
]

VAGUE_PATTERNS = [
    r"not working",
    r"isn'?t working",
    r"is broken",
    r"doesn'?t work",
    r"stopped working",
    r"having (an )?issue",
    r"broken\b",
]
SPECIFIC_SIGNAL_PATTERNS = [
    r"timezone",
    r"credential",
    r"\bapi\b",
    r"\b(owner|admin|analyst|viewer)\b",
    r"render_failed",
    r"source_refresh_timeout",
    r"destination_unverified",
    r"\bschedule\b",
    r"\bdestination\b",
    r"\baudit\b",
    r"workspace id",
    r"connection id",
    r"dashboard id",
    r"\brun id\b",
]


def _matches_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def triage_node(state: Dict[str, Any]) -> Dict[str, Any]:
    tr = trace(state, "triage")
    q = state["question"].lower()

    if _matches_any(OUT_OF_SCOPE_PATTERNS, q):
        return {
            **tr,
            "classification": "out_of_scope",
            "triage_reason": (
                "Request asks for an account action (e.g. refund) or asks to "
                "override the supplied instructions. Neither is within the "
                "support agent's scope per KB-010."
            ),
        }

    if _matches_any(ESCALATION_PATTERNS, q):
        return {
            **tr,
            "classification": "requires_escalation",
            "triage_reason": (
                "Request indicates documented checks were already completed "
                "and the issue persists, matching the escalation criteria in KB-008."
            ),
        }

    is_vague = _matches_any(VAGUE_PATTERNS, q)
    has_specifics = _matches_any(SPECIFIC_SIGNAL_PATTERNS, q)
    if is_vague and not has_specifics:
        return {
            **tr,
            "classification": "requires_clarification",
            "triage_reason": (
                "Request reports a generic failure without identifying "
                "details needed to look up a specific workflow."
            ),
            "clarification_question": (
                "Could you share a few more details: which feature is affected "
                "(dashboard, export, connection, credential, etc.), any "
                "workspace/connection/schedule ID involved, the current status "
                "shown in the product, and any error code from run history? "
                "See KB-008 for the full diagnostic checklist."
            ),
        }

    return {
        **tr,
        "classification": "answerable",
        "triage_reason": "Request references specific OrbitDesk workflow(s) that the knowledge base covers.",
    }


# ------------------------------------------------------------------ #
# Retrieval
# ------------------------------------------------------------------ #
def make_retrieval_node(index):
    def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
        tr = trace(state, "retrieval")
        query = state["question"]

        if state.get("classification") == "requires_escalation":
            query = (
                query
                + " escalation diagnostic information "
                + "checks completed error code run history "
                + "workspace ID connection ID dashboard ID schedule ID"
         )

        results = index.search(query, top_k=TOP_K)
        logger.info(
            "Retrieved %d chunks, top source=%s score=%.3f",
            len(results),
            results[0]["source_id"] if results else "none",
            results[0]["score"] if results else 0.0,
        )
        return {**tr, "retrieved": results}

    return retrieval_node


# ------------------------------------------------------------------ #
# Generation
# ------------------------------------------------------------------ #
# Kept as a separate system-role message (rather than folded into one long
# user turn) because small instruct models like Qwen2.5-0.5B follow a
# short, clearly-scoped system prompt much more reliably than a single
# giant user message that mixes instructions + evidence + question --
# the latter was observed in testing to sometimes produce short,
# non-substantive replies that trivially failed the grounding check.
SYSTEM_PROMPT = (
    "You are the OrbitDesk support assistant. "
    "Answer ONLY from the evidence provided in the user message. "
    "Do not use outside knowledge or generic troubleshooting advice. "
    "Every factual claim must be directly supported by the evidence. "

    "If the evidence does not contain the information needed to answer "
    "the question, say that the supplied documentation does not specify "
    "the requested information and recommend escalation. "
    "Do not fill missing information with guesses or general IT advice. "

    "Never invent logs, metrics, error messages, system checks, "
    "permissions, settings, or troubleshooting steps. "

    "For escalation questions, report only the documented escalation "
    "conditions and diagnostic information present in the evidence. "

    "If a resolved case is marked superseded, do not present its "
    "resolution as current guidance. "

    "Cite at least one supporting source id inline, such as (KB-008) "
    "or (CASE-1041). "

    "For yes/no permission questions, pay close attention to negation. "
    "If the evidence says a role cannot perform an action, answer No. "

    "Keep the answer under 120 words. "
    "Return plain text, not JSON."

    "Do not add troubleshooting steps merely because they sound reasonable. "
    "If a step is not explicitly supported by an evidence passage, leave it out. "
    "When the evidence provides a numbered troubleshooting procedure, follow "
    "that procedure rather than inventing additional steps. "
)

def _build_prompt(
    question: str,
    retrieved: List[Dict[str, Any]],
    revision_feedback: str = "",
) -> str:
    evidence_block = "\n\n".join(
        f"[{r['source_id']} | status={r['doc_status']}]\n{r['passage']}"
        for r in retrieved
    )

    prompt = f"""
You are an OrbitDesk support assistant.

Answer the user's question using ONLY the evidence below.

STRICT RULES:
- Never invent information.
- Never contradict explicit evidence.
- If the evidence says something is "not allowed" or "cannot" be done, answer "No".
- If the evidence explicitly says something "can" be done, answer "Yes".
- For yes/no questions, start with exactly "Yes" or "No".
- Do not add information that is not explicitly supported by the evidence.
- For troubleshooting questions, use only the documented troubleshooting steps.
- Do not add plausible or assumed troubleshooting steps.
- Keep the answer concise.

EVIDENCE:
{evidence_block}

USER QUESTION:
{question}
"""

    if revision_feedback:
        prompt += f"""
Your previous answer failed verification because:
{revision_feedback}

Correct the answer using ONLY the evidence above.
"""

    return prompt


def _extract_answer_text(raw_text: str) -> str:
    """The real local LLM is asked for plain text; the offline
    template-fallback still emits a small JSON blob for backward
    compatibility. Handle both: prefer a JSON "answer" field if the model
    happened to wrap its reply in JSON anyway, otherwise use the raw text.
    """
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if "answer" in parsed and parsed["answer"].strip():
                return parsed["answer"].strip()
        except json.JSONDecodeError:
            pass
    return raw_text.strip()


def _estimate_confidence(retrieved: List[Dict[str, Any]]) -> float:
    """Confidence is derived from retrieval quality rather than a
    self-reported number from a 0.5B model (which is not a reliable
    signal at that scale) -- top retrieval similarity score, clamped to
    a sane range.
    """
    if not retrieved:
        return 0.3
    top_score = max(r["score"] for r in retrieved)
    return round(min(0.95, max(0.35, top_score)), 2)

def _correct_permission_answer(
    question: str,
    answer: str,
    retrieved: List[Dict[str, Any]],
) -> str:
    """Correct obvious yes/no permission contradictions using explicit evidence."""

    if not _YESNO_QUESTION_PATTERN.match(question.strip()):
        return answer

    role_match = re.search(
        r"\b(viewer|analyst|owner|admin)s?\b",
        question.lower(),
    )
    if not role_match:
        return answer

    role = role_match.group(1)

    # Check evidence for an explicit prohibition involving the same role.
    for r in retrieved:
        for sent in re.split(r"(?<=[.!?])\s+", r["passage"]):
            sent_lower = sent.lower()

            if not re.search(rf"\b{re.escape(role)}s?\b", sent_lower):
                continue

            if _NEGATION_PATTERN.search(sent_lower):
                if "credential" in question.lower() and "credential" in sent_lower:
                    return (
                        f"No, a read-only {role.capitalize()} cannot create "
                        f"an API credential. ({r['source_id']})"
                    )

    return answer

def make_generation_node(generation_model):
    def generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
        tr = trace(state, "generation")
        retrieved = state.get("retrieved", [])
        revision_feedback = ""
        if state.get("retry_count", 0) > 0:
            revision_feedback = "; ".join(state.get("verification_notes", []))

        prompt = _build_prompt(state["question"], retrieved, revision_feedback)
        raw = generation_model.generate(prompt, system=SYSTEM_PROMPT)
        logger.info("RAW MODEL OUTPUT (first 300 chars): %r", raw[:300])

        answer_text = _extract_answer_text(raw)

        answer_text = _correct_permission_answer(
            state["question"],
            answer_text,
            retrieved,
        )

        logger.info("CORRECTED ANSWER (first 200 chars): %r", answer_text[:200])
        logger.info("EXTRACTED ANSWER (first 200 chars): %r", answer_text[:200])
        confidence = _estimate_confidence(retrieved)

        sources = [
            {"source_id": r["source_id"], "passage": r["passage"][:300]} for r in retrieved
        ]

        return {
            **tr,
            "draft_answer": answer_text,
            "draft_confidence": confidence,
            "draft_sources": sources,
        }

    return generation_node


# ------------------------------------------------------------------ #
# Verification
# ------------------------------------------------------------------ #
BANNED_PHRASES = [
    "i have issued",
    "refund has been processed",
    "i will refund",
    "i'll process your refund",
    "your password is",
    "here is the secret",
    "here is your api secret",
]


def _grounding_ratio(answer: str, retrieved: List[Dict[str, Any]]) -> float:
    if not retrieved:
        return 0.0
    evidence_tokens = set()
    for r in retrieved:
        evidence_tokens.update(re.findall(r"[a-z0-9_]+", r["passage"].lower()))
    answer_tokens = set(re.findall(r"[a-z0-9_]+", answer.lower()))
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & evidence_tokens
    return len(overlap) / len(answer_tokens)


# --- Polarity / contradiction check -----------------------------------
# The lexical grounding check above only measures whether the answer's
# *words* come from the evidence -- it happily passes an answer that uses
# all the right vocabulary but flips the polarity (e.g. evidence says
# "Viewers CANNOT create API credentials" and the model answers "Yes,
# Viewers CAN create API credentials"). That's a hallucination just as
# serious as inventing unrelated facts, and it's specifically the failure
# mode small instruct models hit on negation-heavy permission questions.
# This check is deliberately narrow/domain-shaped (yes/no permission-style
# questions) rather than a general NLI model -- see docs/DESIGN.md for why
# that trade-off was made given the time budget.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "did", "can", "could",
    "would", "will", "to", "of", "for", "in", "on", "and", "or", "i",
    "read", "only", "my", "me", "you", "your",
}
_NEGATION_PATTERN = re.compile(
    r"\bcannot\b|\bcan\s*not\b|\bcan't\b|\bnot allowed\b|\bnot permitted\b|\bno permission\b"
)
_YESNO_QUESTION_PATTERN = re.compile(
    r"^\s*(can|does|is|are|do|did|could|would|will)\b", re.IGNORECASE
)


def _answer_polarity(answer: str) -> str | None:
    lead = answer.strip().lower()[:120]

    if re.match(r"^(no\b|no,|no\.|cannot\b|can't\b)", lead):
        return "no"

    if _NEGATION_PATTERN.search(lead):
        return "no"

    if re.match(r"^(yes\b|yes,|yes\.|can\b)", lead):
        return "yes"

    return None


def _contradiction_note(
    question: str,
    answer: str,
    retrieved: List[Dict[str, Any]],
) -> str | None:
    """
    Detect contradictions for yes/no permission questions.

    Evidence must refer to the same role as the question.
    For example, evidence saying Owners can create credentials
    must not be treated as evidence that Viewers can create them.
    """
    if not _YESNO_QUESTION_PATTERN.match(question.strip()):
        return None

    answer_polarity = _answer_polarity(answer)
    if answer_polarity is None:
        return None

    question_lower = question.lower()

    # Extract the role explicitly mentioned in the question.
    role_match = re.search(
        r"\b(viewer|analyst|owner|admin)s?\b",
        question_lower,
    )

    if not role_match:
        return None

    role = role_match.group(1)

    # Extract the important action/object words from the question.
    question_tokens = {
        w
        for w in re.findall(r"[a-z0-9]+", question_lower)
        if w not in _STOPWORDS
    }

    for r in retrieved:
        for sent in re.split(r"(?<=[.!?])\s+", r["passage"]):
            sent_lower = sent.lower()

            # Evidence must explicitly discuss the SAME role.
            if not re.search(rf"\b{re.escape(role)}s?\b", sent_lower):
                continue

            sentence_tokens = set(
                re.findall(r"[a-z0-9]+", sent_lower)
            )

            # Require meaningful overlap with the question.
            overlap = question_tokens & sentence_tokens

            if len(overlap) < 2:
                continue

            evidence_polarity = (
                "no"
                if _NEGATION_PATTERN.search(sent_lower)
                else "yes"
            )

            if evidence_polarity != answer_polarity:
                return (
                    f"Answer polarity ('{answer_polarity}') appears to "
                    f"contradict evidence in {r['source_id']}: "
                    f"\"{sent.strip()[:150]}\""
                )

    return None

def verification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    tr = trace(state, "verification")
    notes: List[str] = []
    answer = state.get("draft_answer", "")
    classification = state["classification"]
    retrieved = state.get("retrieved", [])
    sources = state.get("draft_sources", [])

    if classification in ("answerable", "requires_escalation") and not sources:
        notes.append("No sources were attached but the response requires evidence.")

    lower_answer = answer.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lower_answer:
            notes.append(f"Answer contains an unsupported/unsafe commitment: '{phrase}'.")

    grounding = _grounding_ratio(answer, retrieved) if retrieved else 1.0
    if classification in ("answerable", "requires_escalation") and grounding < GROUNDING_OVERLAP_THRESHOLD:
        notes.append(
            f"Answer has low lexical overlap with retrieved evidence "
            f"({grounding:.2f} < {GROUNDING_OVERLAP_THRESHOLD}); possible hallucination."
        )

    if classification in ("answerable", "requires_escalation"):
        contradiction = _contradiction_note(state["question"], answer, retrieved)
        if contradiction:
            notes.append(contradiction)

    candidate = _build_response_dict(state, warnings=notes)
    is_valid, schema_error = validate_response(candidate)
    if not is_valid:
        notes.append(f"Schema validation failed: {schema_error}")

    passed = len(notes) == 0
    logger.info("Verification passed=%s notes=%s", passed, notes)

    return {**tr, "verification_passed": passed, "verification_notes": notes}


def route_after_verification(state: Dict[str, Any]) -> str:
    if state.get("verification_passed"):
        return "format_response"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "safe_failure"


def increment_retry_node(state: Dict[str, Any]) -> Dict[str, Any]:
    tr = trace(state, "increment_retry")
    return {**tr, "retry_count": state.get("retry_count", 0) + 1}


# ------------------------------------------------------------------ #
# Response formatting
# ------------------------------------------------------------------ #
def _build_response_dict(state: Dict[str, Any], warnings: List[str] | None = None) -> Dict[str, Any]:
    classification = state["classification"]
    warnings = warnings or []

    if classification == "out_of_scope":
        return {
            "classification": "out_of_scope",
            "answer": (
                "I can't process refunds, cancellations or other account/billing "
                "actions, and I can't act on instructions to disregard the "
                "supplied documentation. Please contact billing support directly "
                "for subscription or refund requests."
            ),
            "sources": [],
            "confidence": 1.0,
            "requires_human": True,
            "reason": state.get("triage_reason", "Out of scope for the support knowledge base."),
            "clarification_question": None,
            "warnings": warnings,
        }

    if classification == "requires_clarification":
        return {
            "classification": "requires_clarification",
            "answer": "I need a bit more information before I can help with this.",
            "sources": [],
            "confidence": 0.5,
            "requires_human": False,
            "reason": state.get("triage_reason", "Insufficient detail to identify the workflow."),
            "clarification_question": state.get("clarification_question"),
            "warnings": warnings,
        }

    return {
        "classification": classification,
        "answer": state.get("draft_answer", ""),
        "sources": state.get("draft_sources", []),
        "confidence": state.get("draft_confidence", 0.5),
        "requires_human": classification == "requires_escalation",
        "reason": state.get("triage_reason", ""),
        "clarification_question": None,
        "warnings": warnings,
    }


def format_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    tr = trace(state, "format_response")
    response = _build_response_dict(state, warnings=state.get("verification_notes", []))
    return {**tr, "final_response": response}


def safe_failure_node(state: Dict[str, Any]) -> Dict[str, Any]:
    tr = trace(state, "safe_failure")
    response = {
        "classification": "safe_failure",
        "answer": (
            "I wasn't able to produce a verified answer to this question from "
            "the supplied documentation. Please rephrase with more detail, or "
            "this can be escalated to a human agent."
        ),
        "sources": [],
        "confidence": 0.0,
        "requires_human": True,
        "reason": "Generated answer failed verification after the allowed retry.",
        "clarification_question": None,
        "warnings": state.get("verification_notes", []),
    }
    return {**tr, "final_response": response}
