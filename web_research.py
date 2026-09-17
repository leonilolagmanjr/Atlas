"""RAG-style web retrieval: search, read several pages, and isolate the content
that actually answers the request.

A naive hybrid read only the first search result and wrote it wholesale, so a
request like "search the web for the skyrim script and copy it in notepad" could
write a trailer page or a wiki index instead of the script. This module mirrors
the local knowledge pipeline (see ``knowledge_search``): gather candidate
documents, chunk them, score every chunk against the query, drop weak evidence,
and return the best-ranked content with provenance.

It is deterministic and reuses the existing ``web`` search/fetch primitives; it
never executes page content, which stays untrusted data.

All string literals use single quotes so the source survives quote-normalizing
editors without corruption.
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
logger = logging.getLogger(__name__)

#: Chunk sizing for web page text. Pages are prose, so a paragraph-sized window
#: with overlap keeps a relevant passage intact across a boundary.
CHUNK_TARGET_CHARS = 700
CHUNK_OVERLAP_CHARS = 120

#: Terms too common to signal relevance on their own.
_STOPWORDS = {
    'a', 'an', 'the', 'of', 'for', 'to', 'in', 'on', 'is', 'are', 'was', 'be',
    'and', 'or', 'but', 'with', 'about', 'from', 'as', 'at', 'by', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you',
    'your', 'please', 'can', 'could', 'would', 'should', 'do', 'does', 'did',
    'get', 'got', 'find', 'search', 'look', 'up', 'copy', 'paste', 'write',
    'put', 'into', 'notepad', 'web', 'internet', 'online', 'tell', 'give',
    'show', 'me', 'give', 'online', 'read', 'then', 'and', 'full', 'entire',
    'whole', 'complete', 'text', 'content', 'page', 'html', 'retrieve',
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk of page text plus its relevance to the request."""

    url: str
    title: str
    text: str
    score: float
    position: int
    order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'url': self.url,
            'title': self.title,
            'text': self.text,
            'score': round(self.score, 4),
            'position': self.position,
        }


@dataclass
class ResearchResult:
    """The ranked, de-duplicated content retrieved for a request."""

    query: str
    content: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    pages_read: int = 0
    results_seen: int = 0
    notes: list[str] = field(default_factory=list)
    #: The structured retrieval task the pipeline operated on (web_task.RetrievalTask).
    task: Any = None
    #: Whether the extracted content passed task-aware validation.
    validated: bool = False
    detected_type: str = 'generic'
    attempts: int = 0
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'query': self.query,
            'content': self.content,
            'sources': list(self.sources),
            'pages_read': self.pages_read,
            'results_seen': self.results_seen,
            'chunk_count': len(self.chunks),
            'notes': list(self.notes),
            'task': self.task.to_dict() if hasattr(self.task, 'to_dict') else None,
            'validated': self.validated,
            'detected_type': self.detected_type,
            'attempts': self.attempts,
            'candidate_scores': list(self.candidate_scores),
            'validation': dict(self.validation),
        }


def query_terms(query: str) -> list[str]:
    """Return the distinctive terms of a request, longest first."""

    tokens = [token for token in _TOKEN_RE.findall(query.casefold()) if token not in _STOPWORDS and len(token) >= 3]
    # De-duplicate while keeping the longest form of each term.
    seen: dict[str, str] = {}
    for token in tokens:
        current = seen.get(token[:4])
        if current is None or len(token) > len(current):
            seen[token[:4]] = token
    unique = list(dict.fromkeys(seen.values()))
    # Longer (more specific) terms first so they dominate scoring.
    return sorted(unique, key=len, reverse=True)


def chunk_text(text: str, *, target: int = CHUNK_TARGET_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Split page text into overlapping chunks, preferring paragraph breaks."""

    cleaned = re.sub(r'\s+', ' ', text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= target:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + target, length)
        if end < length:
            # Prefer to break at a sentence or word boundary near the target.
            window = cleaned[start:end]
            cut = max(window.rfind('. '), window.rfind('! '), window.rfind('? '))
            if cut < target * 0.5:
                cut = window.rfind(' ')
            if cut > 0:
                end = start + cut + 1
        chunks.append(cleaned[start:end].strip())
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def _prose_factor(chunk: str) -> float:
    # Scale term relevance by how prose-like (not chrome-like) a chunk is. Real
    # content reads as sentences; chrome (menus, tracklists, share rails) is
    # short fragments and list noise that merely repeats keywords, so it must
    # not outrank a genuine passage that contains the actual text.
    words = chunk.split()
    if not words:
        return 0.0
    sentences = len(re.findall(r"[.!?]", chunk))
    average_word = sum(len(word) for word in words) / len(words)
    length_factor = min(1.0, len(words) / 40.0)
    sentence_factor = min(1.0, sentences / 5.0)
    word_factor = 1.0 if 3.0 <= average_word <= 8.0 else 0.4
    return 0.15 + 0.85 * (length_factor * sentence_factor * word_factor)


def score_chunk(chunk: str, terms: list[str]) -> float:
    """Score a chunk by weighted, de-duplicated term coverage."""

    if not terms:
        return 0.0
    haystack = chunk.casefold()
    coverage = 0.0
    for term in terms:
        # Presence, not frequency: repetition is what chrome does (a tracklist
        # repeating "Bee Movie (Script)"), so counting repeats would let chrome
        # beat the real passage. Longer terms are more distinctive.
        if term in haystack:
            coverage += len(term)
    if coverage <= 0.0:
        return 0.0
    # Coverage is a 0..1 fraction of the request's distinctive term weight; the
    # prose factor then decides how much of that relevance is real content.
    total = sum(len(term) for term in terms) or 1.0
    return (coverage / total) * 10.0 * _prose_factor(chunk)


def rank_chunks(
    pages: list[Mapping[str, Any]],
    *,
    query: str,
) -> list[ScoredChunk]:
    """Chunk and rank every page, returning the best chunks across pages."""

    terms = query_terms(query)
    scored: list[ScoredChunk] = []
    order = 0
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        url = str(page.get('url') or '')
        title = str(page.get('title') or '')
        text = str(page.get('text') or '')
        for position, chunk in enumerate(chunk_text(text)):
            order += 1
            scored.append(
                ScoredChunk(
                    url=url,
                    title=title,
                    text=chunk,
                    score=score_chunk(chunk, terms),
                    position=position,
                    order=order,
                )
            )
    # Highest score first; ties keep the earliest document/chunk order.
    scored.sort(key=lambda item: (-item.score, item.order))
    return scored


def _dedupe_adjacent(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Drop chunks whose content is already covered by an accepted chunk."""

    kept: list[ScoredChunk] = []
    for chunk in chunks:
        fingerprint = chunk.text[:120].casefold()
        if any(fingerprint == existing.text[:120].casefold() for existing in kept):
            continue
        if any(fingerprint in existing.text.casefold() or existing.text[:120].casefold() in chunk.text.casefold() for existing in kept):
            continue
        kept.append(chunk)
    return kept


def research(
    query: str,
    *,
    search: Callable[[str, int], Mapping[str, Any]],
    fetch: Callable[[str], Mapping[str, Any] | None],
    max_results: int = 8,
    max_pages: int = 5,
    max_chunks: int = 6,
    min_score: float = 4.0,
    task: Any = None,
    max_attempts: int = 3,
    validate: bool = True,
    ask: Any = None,
) -> ResearchResult:
    """Task-aware web retrieval: search, rank sources, extract, validate, retry.

    ``search(query, max_results)`` returns ``{results: [{title,url,snippet}]}``.
    ``fetch(url)`` returns ``{url,title,text}`` or ``None``.
    
    The pipeline mirrors the architecture: REASON -> PLAN -> SEARCH ->
    RANK SOURCES -> EXTRACT -> VALIDATE -> (ACT | reformulate + retry). A page
    that merely describes the requested subject is rejected when the subject
    itself was asked for, and a bounded retry searches again with a query
    biased toward the missing content type.
    """

    from web_task import (
        infer_retrieval_task, reformulate_query, score_source, validate_content,
    )

    retrieval_task = task if task is not None else infer_retrieval_task(query)
    result = ResearchResult(query=query, content='')
    result.task = retrieval_task
    query_variants = list(retrieval_task.queries) or [query]
    logger.info(
        "web retrieval start: query=%r goal=%s content_type=%s artifact=%s destination=%s",
        query, retrieval_task.goal, retrieval_task.content_type,
        retrieval_task.must_be_artifact, retrieval_task.destination,
    )

    best: ResearchResult | None = None
    for attempt in range(max(1, max_attempts)):
        current_query = query_variants[min(attempt, len(query_variants) - 1)]
        result.attempts = attempt + 1
        attempt_result = _single_attempt(
            current_query, retrieval_task,
            search=search, fetch=fetch,
            max_results=max_results, max_pages=max_pages, max_chunks=max_chunks,
            min_score=min_score, validate=validate, ask=ask,
        )
        # Record diagnostics from every attempt so failures are explainable.
        result.candidate_scores = attempt_result.candidate_scores
        result.results_seen = attempt_result.results_seen
        result.pages_read = max(result.pages_read, attempt_result.pages_read)
        result.detected_type = attempt_result.detected_type
        result.validation = attempt_result.validation
        result.notes.extend(attempt_result.notes)

        logger.info(
            "web retrieval attempt %d: query=%r candidates=%d accepted=%s validated=%s",
            attempt + 1, current_query, len(attempt_result.candidate_scores),
            sum(1 for candidate in attempt_result.candidate_scores if candidate.get('accepted')),
            attempt_result.validated,
        )
        if attempt_result.validated and attempt_result.content.strip():
            logger.info(
                "web retrieval selected %s (type=%s, %d chars)",
                attempt_result.sources[0] if attempt_result.sources else '?',
                attempt_result.detected_type, len(attempt_result.content),
            )
            attempt_result.query = query
            attempt_result.attempts = attempt + 1
            return attempt_result
        # Keep the strongest failing attempt as a fallback so a genuine
        # best-effort answer can still be surfaced honestly.
        if best is None or len(attempt_result.content) > len(best.content):
            best = attempt_result
        if attempt + 1 >= max_attempts:
            break
        reason = str(attempt_result.validation.get('reason') or 'no suitable content')
        reformulated = reformulate_query(retrieval_task, reason=reason, attempt=attempt)
        if reformulated and reformulated not in query_variants:
            query_variants.append(reformulated)
            result.notes.append(f'retrying with: {reformulated!r} ({reason})')

    if best is not None:
        best.query = query
        best.attempts = result.attempts
        return best
    return result


def _single_attempt(
    query: str,
    task: Any,
    *,
    search: Callable[[str, int], Mapping[str, Any]],
    fetch: Callable[[str], Mapping[str, Any] | None],
    max_results: int,
    max_pages: int,
    max_chunks: int,
    min_score: float,
    validate: bool,
    ask: Any = None,
) -> ResearchResult:
    """One search -> rank-sources -> extract -> validate pass."""

    from web_task import llm_validate_content, score_source, validate_content

    result = ResearchResult(query=query, content='')
    result.task = task
    search_output = search(query, max_results) or {}
    results = search_output.get('results')
    if not isinstance(results, list) or not results:
        result.notes.append('no search results were returned')
        return result
    result.results_seen = len(results)

    # Fetch candidate pages first (bounded), then rank the sources by their
    # metadata + body before ranking chunks inside the winner.
    candidates: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for item in results[:max_pages]:
        if not isinstance(item, Mapping) or not item.get('url'):
            continue
        page = fetch(str(item['url']))
        if page is None:
            continue
        candidates.append((item, page))
    result.pages_read = len(candidates)
    if not candidates:
        result.notes.append('no pages could be read from the search results')
        return result

    scored_sources = []
    for item, page in candidates:
        url = str(page.get('url') or item.get('url') or '')
        title = str(page.get('title') or item.get('title') or '')
        source_score = score_source(
            task, url=url, title=title,
            snippet=str(item.get('snippet') or ''),
            text=str(page.get('text') or ''),
        )
        scored_sources.append((source_score, page, title))
    scored_sources.sort(key=lambda entry: entry[0].score, reverse=True)
    result.candidate_scores = [entry[0].to_dict() for entry in scored_sources]

    accepted = [entry for entry in scored_sources if entry[0].accepted]
    ordered = accepted or scored_sources

    # Rank chunks within the best sources (highest source score first).
    for rank, (source_score, page, title) in enumerate(ordered):
        enriched = dict(page)
        if not enriched.get('title'):
            enriched['title'] = title
        ranked = rank_chunks([enriched], query=query)
        if not ranked:
            continue
        strong = [chunk for chunk in ranked if chunk.score >= min_score]
        # When the source itself is a strong task match but no single chunk
        # cleared the threshold (common for short media/link/overview pages),
        # fall back to the source's own content rather than returning nothing.
        if not strong and source_score.accepted:
            body = str(page.get('text') or '').strip()
            if body:
                strong = [c for c in ranked if c.score > 0][:1] or ranked[:1]
        if not strong:
            strong = ranked[:1]
        # A document request ("the full script") wants the whole artifact, so
        # the winning source's complete text is returned, not a top-N sample.
        # A summary/information request keeps the chunk-ranked selection.
        wants_full_text = str(getattr(task, 'desired_output', '')) == 'full_text'
        if wants_full_text:
            full_text = str(page.get('text') or '').strip()
            selected = sorted(ranked, key=lambda item: item.position)
            content = full_text or '\n\n'.join(chunk.text for chunk in selected)
        else:
            selected = _dedupe_adjacent(sorted(strong, key=lambda c: c.position))[:max_chunks]
            selected.sort(key=lambda item: item.order)
            content = '\n\n'.join(chunk.text for chunk in selected)
        detected = source_score.detected_type

        validation = None
        if validate:
            validation = validate_content(
                task, content=content, detected_type=detected,
                title=title, url=source_score.url,
            )
            # Escalate to the model only on a genuine borderline: the source
            # ranked well (so it is plausible) but the deterministic check
            # rejected it. Deterministic rules stay authoritative everywhere else.
            if not validation.ok and ask is not None and source_score.score >= 0.45:
                model_view = llm_validate_content(
                    task, content=content, detected_type=detected,
                    title=title, url=source_score.url, request=query, ask=ask,
                )
                if model_view is not None and model_view.ok:
                    model_view.reason = 'model accepted after deterministic rejection: ' + model_view.reason
                    validation = model_view
        ok = validation.ok if validation is not None else bool(content.strip())
        result.validation = validation.to_dict() if validation is not None else {}
        result.detected_type = detected
        if ok and content.strip():
            result.content = content
            result.chunks = selected
            result.sources = [source_score.url]
            result.validated = True
            return result
        # Nothing accepted yet: remember the best rejected content as fallback.
        if not result.content and content.strip():
            result.content = content
            result.chunks = selected
            result.sources = [source_score.url]
        if validation is not None:
            result.notes.append('rejected ' + source_score.url + ': ' + validation.reason)

    result.validated = False
    return result

