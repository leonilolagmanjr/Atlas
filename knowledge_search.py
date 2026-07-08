"""Knowledge retrieval orchestration.

ONE responsibility: retrieval (no indexing, no document loading).
Coordinates staged semantic, keyword, and metadata-aware search before making
the final confidence decision.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol

from config import LOG_RETRIEVAL, MIN_SIMILARITY, TOP_K
from vector_store import SearchHit, VectorStore

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "about",
    "am",
    "an",
    "and",
    "are",
    "as",
    "do",
    "does",
    "for",
    "have",
    "i",
    "is",
    "me",
    "my",
    "of",
    "on",
    "the",
    "tell",
    "to",
    "what",
    "which",
    "who",
}

EXPANSION_GROUPS: dict[str, tuple[str, ...]] = {
    "certification": ("certifications", "certificates", "credentials", "training", "licenses", "achievements"),
    "certifications": ("certifications", "certificates", "credentials", "training", "licenses", "achievements"),
    "certificate": ("certifications", "certificates", "credentials", "training", "licenses", "achievements"),
    "certificates": ("certifications", "certificates", "credentials", "training", "licenses", "achievements"),
    "credential": ("certifications", "certificates", "credentials", "training", "licenses", "achievements"),
    "credentials": ("certifications", "certificates", "credentials", "training", "licenses", "achievements"),
    "resume": ("resume", "experience", "education", "skills", "projects", "profile"),
    "cv": ("resume", "experience", "education", "skills", "projects", "profile"),
    "programming": ("programming languages", "languages", "technical skills", "skills", "technologies"),
    "language": ("programming languages", "languages", "technical skills", "skills", "technologies"),
    "languages": ("programming languages", "languages", "technical skills", "skills", "technologies"),
    "project": ("projects", "completed projects", "portfolio", "experience", "achievements"),
    "projects": ("projects", "completed projects", "portfolio", "experience", "achievements"),
    "leonilo": ("Leonilo", "Leonilo Lagman", "Lagman", "candidate", "profile", "personal information"),
    "lagman": ("Leonilo", "Leonilo Lagman", "Lagman", "candidate", "profile", "personal information"),
}


@dataclass(frozen=True)
class RetrievalDecision:
    """Final retrieval confidence decision."""

    accepted: bool
    reason: str
    best_distance: Optional[float]
    semantic_hits: int
    keyword_hits: int
    filename_hits: int


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Explainable retrieval diagnostics for logging and future UI display."""

    original_query: str
    expanded_queries: List[str]
    retrieved_documents: List[str]
    distances: List[float]
    keyword_matches: int
    filename_matches: int
    final_decision: RetrievalDecision


@dataclass(frozen=True)
class RetrievalResult:
    """Result of retrieval after staged confidence evaluation."""

    context: str
    best_distance: Optional[float]
    retrieved_chunks: List[SearchHit] = field(default_factory=list)
    expanded_queries: List[str] = field(default_factory=list)
    diagnostics: Optional[RetrievalDiagnostics] = None


@dataclass(frozen=True)
class RankedHit:
    """Search hit plus hybrid retrieval signals."""

    hit: SearchHit
    semantic_score: float = 0.0
    keyword_matches: int = 0
    filename_matches: int = 0

    @property
    def combined_score(self) -> float:
        return self.semantic_score + (self.keyword_matches * 0.35) + (self.filename_matches * 0.75)


class QueryExpander(Protocol):
    """Interface for replaceable query expansion strategies."""

    def expand(self, question: str) -> List[str]:
        """Return ordered search queries to try."""


class RuleBasedQueryExpander:
    """Simple deterministic expansion strategy.

    This can later be replaced by an LLM or planner without changing the
    retrieval policy.
    """

    def expand(self, question: str) -> List[str]:
        tokens = _tokens(question, keep_stop_words=False)
        queries = [question.strip()]

        for token in tokens:
            for expanded in EXPANSION_GROUPS.get(token, ()):
                queries.append(expanded)

        queries.extend(tokens)
        return _unique_non_empty(queries)


def _unique_non_empty(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for value in values:
        clean = value.strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            unique.append(clean)
    return unique


def _tokens(text: str, *, keep_stop_words: bool = True) -> List[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if keep_stop_words:
        return tokens
    return [token for token in tokens if token not in STOP_WORDS]


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


def _semantic_score(distance: float) -> float:
    return max(0.0, 1.5 - distance)


def _search_text(hit: SearchHit) -> str:
    return f"{hit.source} {hit.doc_id} {hit.text}".lower()


def _metadata_text(hit: SearchHit) -> str:
    return f"{hit.source} {hit.doc_id}".lower()


def _count_matches(terms: Iterable[str], text: str) -> int:
    return sum(1 for term in terms if term and term.lower() in text)


def _merge_ranked_hits(existing: dict[str, RankedHit], ranked: RankedHit) -> None:
    current = existing.get(ranked.hit.chunk_id)
    if current is None:
        existing[ranked.hit.chunk_id] = ranked
        return

    existing[ranked.hit.chunk_id] = RankedHit(
        hit=current.hit if current.hit.distance <= ranked.hit.distance else ranked.hit,
        semantic_score=max(current.semantic_score, ranked.semantic_score),
        keyword_matches=max(current.keyword_matches, ranked.keyword_matches),
        filename_matches=max(current.filename_matches, ranked.filename_matches),
    )


def _semantic_search(query: str, *, vector_store: VectorStore) -> List[RankedHit]:
    hits = vector_store.query(query_text=query, top_k=TOP_K)
    return [RankedHit(hit=hit, semantic_score=_semantic_score(hit.distance)) for hit in hits]


def _keyword_search(queries: List[str], *, vector_store: VectorStore) -> List[RankedHit]:
    terms = _unique_non_empty(
        term
        for query in queries
        for term in ([query] + _tokens(query, keep_stop_words=False))
    )

    ranked_hits: List[RankedHit] = []
    for hit in vector_store.all_chunks():
        keyword_matches = _count_matches(terms, _search_text(hit))
        filename_matches = _count_matches(terms, _metadata_text(hit))
        if keyword_matches or filename_matches:
            ranked_hits.append(
                RankedHit(
                    hit=hit,
                    keyword_matches=keyword_matches,
                    filename_matches=filename_matches,
                )
            )

    return ranked_hits


def _decide(ranked_hits: List[RankedHit]) -> RetrievalDecision:
    if not ranked_hits:
        return RetrievalDecision(
            accepted=False,
            reason="no semantic, keyword, or metadata matches",
            best_distance=None,
            semantic_hits=0,
            keyword_hits=0,
            filename_hits=0,
        )

    distances = [ranked.hit.distance for ranked in ranked_hits if ranked.semantic_score > 0]
    best_distance = min(distances) if distances else None
    strong_semantic = best_distance is not None and best_distance <= MIN_SIMILARITY
    near_semantic = best_distance is not None and best_distance <= (MIN_SIMILARITY + 0.45)
    semantic_hits = sum(1 for ranked in ranked_hits if ranked.semantic_score > 0)
    keyword_hits = sum(1 for ranked in ranked_hits if ranked.keyword_matches > 0)
    filename_hits = sum(1 for ranked in ranked_hits if ranked.filename_matches > 0)
    total_keyword_matches = sum(ranked.keyword_matches for ranked in ranked_hits)
    total_filename_matches = sum(ranked.filename_matches for ranked in ranked_hits)

    if strong_semantic:
        return RetrievalDecision(True, "strong semantic match", best_distance, semantic_hits, keyword_hits, filename_hits)

    if near_semantic and (total_keyword_matches >= 2 or total_filename_matches >= 1):
        return RetrievalDecision(
            True,
            "semantic match supported by keyword or filename evidence",
            best_distance,
            semantic_hits,
            keyword_hits,
            filename_hits,
        )

    if total_filename_matches >= 1 and total_keyword_matches >= 2:
        return RetrievalDecision(
            True,
            "metadata and keyword match without strong semantic distance",
            best_distance,
            semantic_hits,
            keyword_hits,
            filename_hits,
        )

    if keyword_hits >= 2 and total_keyword_matches >= 4:
        return RetrievalDecision(
            True,
            "multiple keyword matches across stored chunks",
            best_distance,
            semantic_hits,
            keyword_hits,
            filename_hits,
        )

    reason = (
        "weak retrieval evidence "
        f"(best_distance={best_distance}, keyword_matches={total_keyword_matches}, "
        f"filename_matches={total_filename_matches})"
    )
    return RetrievalDecision(False, reason, best_distance, semantic_hits, keyword_hits, filename_hits)


def _log_diagnostics(diagnostics: RetrievalDiagnostics) -> None:
    if not LOG_RETRIEVAL:
        return

    decision = diagnostics.final_decision
    logger.info("Retrieval original_query=%r", diagnostics.original_query)
    logger.info("Retrieval expanded_queries=%s", diagnostics.expanded_queries)
    logger.info("Retrieval documents=%s", diagnostics.retrieved_documents)
    logger.info("Retrieval distances=%s", [round(distance, 4) for distance in diagnostics.distances])
    logger.info(
        "Retrieval evidence: keyword_matches=%d filename_matches=%d accepted=%s reason=%s",
        diagnostics.keyword_matches,
        diagnostics.filename_matches,
        decision.accepted,
        decision.reason,
    )

    if not decision.accepted:
        logger.warning("Retrieval rejected: %s", decision.reason)


def retrieve(
    question: str,
    *,
    vector_store: VectorStore,
    query_expander: Optional[QueryExpander] = None,
) -> RetrievalResult:
    """Retrieve context from the knowledge base using staged hybrid search."""

    expander = query_expander or RuleBasedQueryExpander()
    expanded_queries = expander.expand(question)
    ranked_by_chunk: dict[str, RankedHit] = {}

    try:
        for query in expanded_queries[:1]:
            for ranked in _semantic_search(query, vector_store=vector_store):
                _merge_ranked_hits(ranked_by_chunk, ranked)

        first_decision = _decide(list(ranked_by_chunk.values()))

        if not first_decision.accepted:
            for query in expanded_queries[1:]:
                for ranked in _semantic_search(query, vector_store=vector_store):
                    _merge_ranked_hits(ranked_by_chunk, ranked)

        second_decision = _decide(list(ranked_by_chunk.values()))

        if not second_decision.accepted:
            for ranked in _keyword_search(expanded_queries, vector_store=vector_store):
                _merge_ranked_hits(ranked_by_chunk, ranked)

    except Exception:
        logger.exception("Knowledge retrieval failed")
        return RetrievalResult(context="", best_distance=None, expanded_queries=expanded_queries)

    ranked_hits = sorted(
        ranked_by_chunk.values(),
        key=lambda ranked: (-ranked.combined_score, ranked.hit.distance, ranked.hit.source, ranked.hit.chunk_index),
    )
    selected_ranked_hits = ranked_hits[:TOP_K]
    selected_hits = [ranked.hit for ranked in selected_ranked_hits]
    decision = _decide(selected_ranked_hits)
    distances = [ranked.hit.distance for ranked in selected_ranked_hits if ranked.semantic_score > 0]
    diagnostics = RetrievalDiagnostics(
        original_query=question,
        expanded_queries=expanded_queries,
        retrieved_documents=_unique_non_empty(hit.source for hit in selected_hits),
        distances=distances,
        keyword_matches=sum(ranked.keyword_matches for ranked in selected_ranked_hits),
        filename_matches=sum(ranked.filename_matches for ranked in selected_ranked_hits),
        final_decision=decision,
    )
    _log_diagnostics(diagnostics)

    if not decision.accepted:
        return RetrievalResult(
            context="",
            best_distance=decision.best_distance,
            retrieved_chunks=selected_hits,
            expanded_queries=expanded_queries,
            diagnostics=diagnostics,
        )

    return RetrievalResult(
        context=_format_context(selected_hits),
        best_distance=decision.best_distance,
        retrieved_chunks=selected_hits,
        expanded_queries=expanded_queries,
        diagnostics=diagnostics,
    )
