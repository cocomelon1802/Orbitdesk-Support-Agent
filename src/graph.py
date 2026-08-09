"""
Graph construction.

    START
      |
    triage --(out_of_scope | requires_clarification)--> format_response --> END
      |
      (answerable | requires_escalation)
      |
    retrieval -> generation -> verification --(pass)--> format_response --> END
                                    |
                                    (fail, retry_count < MAX_RETRIES)
                                    v
                              increment_retry -> generation   [loop, capped at 1 retry]
                                    |
                                    (fail, retry_count >= MAX_RETRIES)
                                    v
                              safe_failure --> END

Loop protection: increment_retry_node bumps retry_count each pass; the
conditional edge in route_after_verification hard-routes to safe_failure
once retry_count >= MAX_RETRIES (1), so generation can run at most twice
per request. This is the "protection against an infinite graph loop"
requirement.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from .state import GraphState
from .models import EmbeddingModel, GenerationModel
from .retrieval import RetrievalIndex
from .nodes import (
    triage_node,
    make_retrieval_node,
    make_generation_node,
    verification_node,
    route_after_verification,
    increment_retry_node,
    format_response_node,
    safe_failure_node,
)

logger = logging.getLogger("orbitdesk.graph")


def triage_router(state: Dict[str, Any]) -> str:
    if state["classification"] in ("out_of_scope", "requires_clarification"):
        return "format_response"
    return "retrieval"


def build_models_and_index(data_dir: str = "data"):
    embedder = EmbeddingModel()
    generator = GenerationModel()
    index = RetrievalIndex(data_dir=data_dir, embedder=embedder)

    metadata = {
        "embedding_backend": embedder.backend,
        "embedding_load_time_s": embedder.load_time_seconds,
        "generation_backend": generator.backend,
        "generation_load_time_s": generator.load_time_seconds,
        "indexed_chunks": len(index.chunks),
    }
    logger.info("Model/index metadata: %s", metadata)
    return embedder, generator, index, metadata


def build_graph(data_dir: str = "data", embedder=None, generator=None, index=None):
    """Builds and compiles the LangGraph app.

    If embedder/generator/index are not passed in, real local HF models are
    loaded (see src/models.py). Tests pass in lightweight fakes instead so
    routing can be verified without downloading/loading any model weights.
    """
    if index is None:
        embedder, generator, index, _ = build_models_and_index(data_dir)

    workflow = StateGraph(GraphState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("retrieval", make_retrieval_node(index))
    workflow.add_node("generation", make_generation_node(generator))
    workflow.add_node("verification", verification_node)
    workflow.add_node("increment_retry", increment_retry_node)
    workflow.add_node("format_response", format_response_node)
    workflow.add_node("safe_failure", safe_failure_node)

    workflow.set_entry_point("triage")

    workflow.add_conditional_edges(
        "triage",
        triage_router,
        {"format_response": "format_response", "retrieval": "retrieval"},
    )
    workflow.add_edge("retrieval", "generation")
    workflow.add_edge("generation", "verification")
    workflow.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "format_response": "format_response",
            "retry": "increment_retry",
            "safe_failure": "safe_failure",
        },
    )
    workflow.add_edge("increment_retry", "generation")
    workflow.add_edge("format_response", END)
    workflow.add_edge("safe_failure", END)

    return workflow.compile()
