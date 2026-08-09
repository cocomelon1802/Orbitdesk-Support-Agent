"""Console logging + node-trace helper used across all graph nodes."""
from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("orbitdesk.graph")


def trace(state: dict, node_name: str) -> dict:
    """Emit a log line and return a {"node_trace": [node_name]} fragment.

    This is how the graph satisfies "logs showing which nodes executed":
    every node calls this on entry and merges the returned fragment into
    its own return dict. GraphState declares node_trace as
    Annotated[List[str], operator.add], so LangGraph's reducer appends
    each node's fragment to the running list instead of overwriting it
    (the default behaviour for a plain list field would be to overwrite).
    """
    logger.info("NODE EXECUTED: %s", node_name)
    return {"node_trace": [node_name]}
