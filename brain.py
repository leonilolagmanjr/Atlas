"""Atlas Brain orchestration layer.

The Brain is the central request coordinator. In V3 Phase 1 it is a
passthrough over the stable V2 retrieval and LLM workflow.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from knowledge_search import retrieve
from llm import ask
from vector_store import SearchHit, VectorStore

logger = logging.getLogger(__name__)

UNKNOWN_RESPONSE = "I don't know based on my knowledge base."


@dataclass
class ExecutionContext:
    """State for a single Atlas request execution."""

    user_input: str
    rewritten_queries: list[str] = field(default_factory=list)
    retrieved_chunks: list[SearchHit] = field(default_factory=list)
    retrieval_confidence: Optional[float] = None
    selected_tool: Optional[str] = None
    response: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


class Brain:
    """Coordinate the current Atlas request workflow.

    This class intentionally performs no planning, memory management, tool
    selection, or agent coordination yet.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        system_prompt: str,
        retrieval_template: str,
    ) -> None:
        self._vector_store = vector_store
        self._system_prompt = system_prompt
        self._retrieval_template = retrieval_template

    def process(self, user_input: str) -> str:
        """Process a user request through the V2 knowledge and LLM pipeline."""

        started_at = time.perf_counter()
        context = ExecutionContext(user_input=user_input)

        logger.info("Brain received request")
        logger.info("Brain started execution")

        try:
            context.selected_tool = "knowledge"
            retrieval_result = retrieve(user_input, vector_store=self._vector_store)
            context.retrieved_chunks = retrieval_result.retrieved_chunks
            context.retrieval_confidence = retrieval_result.best_distance
            context.metadata["retrieval_metric"] = "chroma_distance_smaller_is_better"
            context.rewritten_queries = retrieval_result.expanded_queries
            context.metadata["retrieval_diagnostics"] = retrieval_result.diagnostics

            logger.info(
                "Knowledge retrieval completed: chunks=%d best_distance=%s expanded_queries=%d",
                len(context.retrieved_chunks),
                (
                    f"{context.retrieval_confidence:.4f}"
                    if context.retrieval_confidence is not None
                    else "none"
                ),
                len(context.rewritten_queries),
            )

            if not retrieval_result.context:
                if retrieval_result.diagnostics is not None:
                    logger.warning(
                        "Knowledge retrieval rejected: %s",
                        retrieval_result.diagnostics.final_decision.reason,
                    )
                context.response = UNKNOWN_RESPONSE
                return self._complete(context, started_at)

            user_prompt = self._retrieval_template.format(
                context=retrieval_result.context,
                question=user_input,
            )

            logger.info("LLM called")
            context.response = ask(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
            )
            return self._complete(context, started_at)

        except Exception:
            logger.exception("Brain execution failed")
            context.response = UNKNOWN_RESPONSE
            return self._complete(context, started_at)

    def _complete(self, context: ExecutionContext, started_at: float) -> str:
        context.execution_time = time.perf_counter() - started_at
        logger.info(
            "Brain completed request: execution_time=%.4fs selected_tool=%s",
            context.execution_time,
            context.selected_tool,
        )
        return context.response or UNKNOWN_RESPONSE
