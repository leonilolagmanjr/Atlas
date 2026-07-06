"""Knowledge retrieval orchestration.

ONE responsibility: retrieval (no indexing, no document loading).
Applies similarity threshold gating and returns context ready for prompting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from config import LOG_RETRIEVAL, MIN_SIMILARITY, TOP_K
from vector_store import SearchHit, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """Result of retrieval gating."""

    context: str
    best_distance: Optional[float]


def _format_context(hits: List[SearchHit]) -> str:
    parts: List[str] = []
    for h in hits:
        page = h.page_number
        parts.append(
            f"Source: {h.source}\n"
            f"Page: {page if page is not None else 'unknown'}\n"
            f"ChunkIndex: {h.chunk_index}\n\n"
            f"{h.text}"
        )
    return "\n\n-----------------------\n\n".join(parts)


def retrieve(question: str, *, vector_store: VectorStore) -> RetrievalResult:
    """Retrieve context from the knowledge base.

    Uses Chroma distance metric where smaller is usually more similar.

    If best match is worse (distance > MIN_SIMILARITY), Atlas must not call
    the LLM.
    """

    try:
        hits = vector_store.query(query_text=question, top_k=TOP_K)
    except Exception:
        logger.exception("Vector query failed")
        return RetrievalResult(context="", best_distance=None)

    if not hits:
        return RetrievalResult(context="", best_distance=None)

    best = min(h.distance for h in hits)

    if LOG_RETRIEVAL:
        logger.info("Retrieval: best_distance=%.4f hits=%d", best, len(hits))

    # Chroma distance: smaller => more similar.
    if best > MIN_SIMILARITY:
        if LOG_RETRIEVAL:
            logger.warning(
                "Retrieval confidence below threshold (best_distance=%.4f > %.4f); gating off LLM",
                best,
                MIN_SIMILARITY,
            )
        return RetrievalResult(context="", best_distance=best)

    context = _format_context(hits)
    return RetrievalResult(context=context, best_distance=best)

