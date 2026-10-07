"""Experience retrieval and ranking.

Experience memory is deliberately **not** merged with factual RAG. Atlas keeps
four conceptually distinct memories apart (spec sections 7 and 20):

* KNOWLEDGE  — "what is true?"          (``vector_store`` / ``knowledge_search``)
* EXPERIENCE — "what worked or failed?" (this package)
* USER MEMORY— "what does this user prefer?"
* REASONING  — "which principles apply?" (future reasoning KB)

Retrieval here is: metadata filter -> semantic-or-lexical similarity -> a
deterministic rank over trust signals. It is bounded (never the whole store),
cheap (no LLM call, no full scan when a candidate set can be narrowed), and it
degrades gracefully: Chroma embeddings are used only when an experience
collection is available, otherwise a deterministic lexical similarity is used.

Embeddings, when present, come from the *same* ChromaDB directory Atlas already
uses for knowledge, but in a **separate collection** with a ``memory_type``
metadata namespace so an experience never contaminates a factual answer. This is
why an unavailable embedder is a normal state rather than an error: the store is
still fully usable and searchable without it.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from config import (
    DATABASE_FOLDER,
    EXPERIENCE_COLLECTION_NAME,
    EXPERIENCE_MIN_RELEVANCE,
    EXPERIENCE_RETRIEVAL_LIMIT,
)
from experience.models import Experience, sanitize_text
from experience.store import ExperienceStore

logger = logging.getLogger(__name__)

#: Words that carry no retrieval signal. Kept small on purpose: over-filtering a
#: short request destroys the signal that makes an experience relevant.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "to", "in", "into", "on", "of", "for",
        "with", "please", "me", "my", "i", "you", "it", "that", "this", "is",
        "are", "was", "be", "do", "does", "can", "could", "would", "should",
        "then", "than", "as", "at", "by", "from", "up", "out", "some", "any",
    }
)

#: Confidence bands surfaced to the planner. They are advisory labels; the raw
#: score is also exposed so nothing downstream has to re-derive a threshold.
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"


def _tokens(text: str) -> set[str]:
    """Tokenize for lexical similarity: lowercase alphanumeric words, no stops."""

    words = re.findall(r"[a-z0-9]+", (text or "").casefold())
    return {word for word in words if word not in _STOPWORDS and len(word) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


@dataclass
class RetrievedExperience:
    """One ranked experience plus the reasons it was retrieved."""

    experience: Experience
    score: float = 0.0
    semantic_similarity: float = 0.0
    task_similarity: float = 0.0
    confidence: str = CONFIDENCE_LOW
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience.experience_id,
            "score": round(self.score, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "task_similarity": round(self.task_similarity, 4),
            "confidence": self.confidence,
            "outcome": self.experience.outcome,
            "reasons": list(self.reasons),
        }


@dataclass
class ExperienceContext:
    """The bounded, formatted experience context handed to planning.

    ``text`` is what actually reaches the model; ``items`` is the structured
    version kept for observability so the UI/API can show *which* experiences
    were used without exposing prompts.
    """

    text: str = ""
    items: list[RetrievedExperience] = field(default_factory=list)
    considered: int = 0

    def __bool__(self) -> bool:
        return bool(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "considered": self.considered,
            "items": [item.to_dict() for item in self.items],
        }


class ExperienceEmbedder:
    """Thin wrapper over the experience Chroma collection (optional).

    It reuses Atlas's existing ChromaDB persistence directory but a *separate*
    collection, and tags every record with ``memory_type="experience"`` so the
    namespace is filterable (spec section 22). Every operation is best-effort: an
    absent chromadb/sentence-transformers install simply disables embeddings.
    """

    def __init__(
        self,
        *,
        persist_dir: str | None = None,
        collection_name: str = EXPERIENCE_COLLECTION_NAME,
    ) -> None:
        self._persist_dir = persist_dir or str(DATABASE_FOLDER)
        self._collection_name = collection_name
        self._client = None
        self._collection = None

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - embeddings are optional by design
            logger.info("Experience embeddings unavailable; using lexical retrieval")
            self._collection = False
        return self._collection

    def index(self, experience: Experience) -> bool:
        """Embed and store one experience. Returns True when it was indexed."""

        collection = self._ensure_collection()
        if not collection:
            return False
        try:
            collection.upsert(
                ids=[experience.experience_id],
                documents=[experience.retrieval_text()],
                metadatas=[experience_metadata(experience)],
            )
            return True
        except Exception:  # noqa: BLE001
            logger.debug("Experience embedding failed for %s", experience.experience_id, exc_info=True)
            return False

    def similarity_scores(self, query: str, experience_ids: Iterable[str]) -> dict[str, float]:
        """Return id -> similarity (1 - distance) for the given candidate ids."""

        ids = [identifier for identifier in experience_ids if identifier]
        if not ids:
            return {}
        collection = self._ensure_collection()
        if not collection:
            return {}
        try:
            result = collection.query(
                query_texts=[query],
                n_results=min(len(ids), 25),
                where={"$and": [{"memory_type": "experience"}, {"experience_id": {"$in": ids}}]},
                include=["distances"],
            )
        except Exception:  # noqa: BLE001 - a filter Chroma rejects must not fail retrieval
            try:
                result = collection.query(
                    query_texts=[query],
                    n_results=min(len(ids), 25),
                    where={"memory_type": "experience"},
                    include=["distances"],
                )
            except Exception:  # noqa: BLE001
                return {}
        returned_ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        scores: dict[str, float] = {}
        for identifier, distance in zip(returned_ids, distances):
            try:
                scores[str(identifier)] = max(0.0, 1.0 - float(distance))
            except (TypeError, ValueError):
                continue
        return scores


def experience_metadata(experience: Experience) -> dict[str, Any]:
    """Return the filterable metadata namespace for one experience.

    Chroma metadata values must be scalars, so lists are joined and ``None``
    becomes an empty string rather than an unsupported type.
    """

    return {
        "memory_type": "experience",
        "experience_type": experience.experience_type,
        "outcome": experience.outcome,
        "task_type": experience.task_type,
        "goal": experience.goal,
        "destination": experience.requested_destination,
        "failure_category": experience.failure_category,
        "state": experience.state,
        "experience_id": experience.experience_id,
        "tools": ",".join(experience.tools_used),
    }


class ExperienceMemory:
    """Retrieve and rank relevant experiences for a new request.

    The public entry point is :meth:`retrieve`, which returns a bounded
    :class:`ExperienceContext`. Everything is deterministic given the same store
    contents, and nothing here mutates user-facing behaviour on its own.
    """

    def __init__(
        self,
        *,
        store: ExperienceStore | None = None,
        embedder: ExperienceEmbedder | None = None,
        limit: int = EXPERIENCE_RETRIEVAL_LIMIT,
        min_relevance: float = EXPERIENCE_MIN_RELEVANCE,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._limit = max(0, int(limit))
        self._min_relevance = float(min_relevance)

    @property
    def store(self) -> ExperienceStore | None:
        return self._store

    # -- retrieval -----------------------------------------------------------

    def retrieve(
        self,
        request: str,
        *,
        task: Any = None,
        limit: int | None = None,
        include_failures: bool = True,
    ) -> ExperienceContext:
        """Return the experiences most relevant to ``request``.

        Signals, in the order they narrow the candidate set (spec section 8):

        1. semantic similarity over the request-shaped text (embeddings when
           available, deterministic lexical similarity otherwise),
        2. metadata agreement (task type, goal, destination, tool requirements),
        3. trust/recency ranking (spec section 9).

        It never returns the whole database, and it never raises into a request.
        """

        if self._store is None:
            return ExperienceContext()
        try:
            candidates = self._store.experiences()
        except Exception:  # noqa: BLE001 - retrieval must never break a request
            logger.exception("Experience retrieval failed")
            return ExperienceContext()
        if not candidates:
            return ExperienceContext()

        if not include_failures:
            candidates = [item for item in candidates if not item.is_failure()]
        if not candidates:
            return ExperienceContext()

        semantics = self._semantic_scores(request, candidates)
        query_tokens = _tokens(request)
        wanted_destination = _destination_of(task)
        wanted_tools = _tools_of(task)

        ranked: list[RetrievedExperience] = []
        for experience in candidates:
            score, semantic, task_similarity, reasons = self._score(
                experience,
                query_tokens=query_tokens,
                semantic=semantics.get(experience.experience_id, 0.0),
                wanted_destination=wanted_destination,
                wanted_tools=wanted_tools,
                request=request,
            )
            if score < self._min_relevance:
                continue
            ranked.append(
                RetrievedExperience(
                    experience=experience,
                    score=score,
                    semantic_similarity=semantic,
                    task_similarity=task_similarity,
                    confidence=_confidence_band(experience, score),
                    reasons=reasons,
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.experience.timestamp))
        top = ranked[: max(1, int(limit if limit is not None else self._limit))]
        return ExperienceContext(
            text=render_experience_context(top),
            items=top,
            considered=len(candidates),
        )

    # -- scoring -------------------------------------------------------------

    def _semantic_scores(self, request: str, candidates: list[Experience]) -> dict[str, float]:
        if not request or self._embedder is None:
            return {}
        try:
            return self._embedder.similarity_scores(
                request, [item.experience_id for item in candidates]
            )
        except Exception:  # noqa: BLE001
            logger.debug("Experience embedding lookup failed", exc_info=True)
            return {}

    def _score(
        self,
        experience: Experience,
        *,
        query_tokens: set[str],
        semantic: float,
        wanted_destination: str,
        wanted_tools: set[str],
        request: str,
    ) -> tuple[float, float, float, list[str]]:
        """Blend similarity and trust into one deterministic relevance score."""

        reasons: list[str] = []
        lexical = _jaccard(query_tokens, _tokens(experience.retrieval_text()))
        # The stronger of the two similarity signals, so embeddings never make
        # retrieval *worse* than the lexical fallback when both are present.
        similarity = max(semantic, lexical)
        if semantic >= lexical and semantic > 0:
            reasons.append("semantically similar task")
        elif lexical > 0:
            reasons.append("lexically similar task")

        task_similarity = 0.0
        if wanted_destination and experience.requested_destination:
            if wanted_destination == experience.requested_destination:
                task_similarity += 0.25
                reasons.append(f"same destination ({wanted_destination})")
            else:
                task_similarity -= 0.1
                reasons.append("different destination")
        if wanted_tools and experience.tools_used:
            overlap = wanted_tools & set(experience.tools_used)
            if overlap:
                task_similarity += 0.15
                reasons.append("same tools")
        if _same_intent(request, experience):
            task_similarity += 0.1
            reasons.append("same task wording")

        # Trust: verified successes and explicit corrections outweigh a single
        # unverified interaction; failures stay visible as "this failed before".
        trust = experience.reliability()
        if experience.is_success():
            reasons.append("previous success")
        elif experience.is_failure():
            reasons.append("previous failure")
            if experience.failure_category:
                reasons.append(f"failed before: {experience.failure_category}")
        if experience.user_correction:
            reasons.append("user correction available")
        if experience.state == "reliable":
            reasons.append("repeated/confirmed experience")

        score = (similarity * 0.6) + (max(-0.2, task_similarity) * 0.5) + (trust * 0.4)
        # Recency is a small tie-breaker, never a dominant signal.
        score += 0.05 * _recency(experience.timestamp)
        return max(0.0, min(1.0, score)), similarity, task_similarity, reasons


def _same_intent(request: str, experience: Experience) -> bool:
    """True when the request is near-identical to the recorded one."""

    left = _tokens(request)
    right = _tokens(experience.original_user_request)
    return _jaccard(left, right) >= 0.75


def _destination_of(task: Any) -> str:
    """Extract the requested destination (application or file) from a Task."""

    if task is None:
        return ""
    entities = getattr(task, "entities", None)
    if not isinstance(entities, dict):
        return ""
    for key in ("application", "filename", "destination"):
        value = entities.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    return ""


def _tools_of(task: Any) -> set[str]:
    if task is None:
        return set()
    actions = getattr(task, "actions", None)
    if not isinstance(actions, list):
        return set()
    tools: set[str] = set()
    for action in actions:
        capability = getattr(action, "capability", None)
        if isinstance(capability, str) and capability:
            tools.add(capability)
    return tools


def _recency(timestamp: str) -> float:
    """Return 1.0 for a fresh experience decaying to 0.0 over ~30 days."""

    try:
        parsed = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - parsed).total_seconds()
    if age_seconds <= 0:
        return 1.0
    # exp(-age / half_life): ~30 days to reach ~0.37, ~90 days to ~0.05.
    return math.exp(-age_seconds / (30 * 24 * 3600))


def _confidence_band(experience: Experience, score: float) -> str:
    if experience.state == "reliable" and experience.is_success() and score >= 0.55:
        return CONFIDENCE_HIGH
    if experience.is_success() and score >= 0.45:
        return CONFIDENCE_MEDIUM
    if experience.is_failure() and score >= 0.45:
        # A well-matched failure is *high-confidence evidence about a failure*,
        # even though it is not evidence about a successful approach.
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def render_experience_context(items: list[RetrievedExperience]) -> str:
    """Render retrieved experiences as compact, unambiguous planning context.

    The wording is deliberately explicit that this is *supporting context from
    previous tasks* and that the current request always wins. That instruction is
    part of the contract (spec section 11), not decoration.
    """

    if not items:
        return ""
    lines: list[str] = [
        "EXPERIENCE MEMORY (supporting context only: the current request always "
        "wins and overrides it):"
    ]
    for item in items:
        experience = item.experience
        label = "SUCCESS" if experience.is_success() else (
            "FAILURE" if experience.is_failure() else "UNVERIFIED"
        )
        confidence = item.confidence.upper()
        request = experience.original_user_request or "(unrecorded request)"
        lines.append(
            f"- [{label} / {confidence} CONFIDENCE] Previous task: {request}"
        )
        if experience.desired_outcome:
            lines.append(f"  Desired outcome: {experience.desired_outcome}")
        if experience.requested_destination:
            lines.append(f"  Requested destination: {experience.requested_destination}")
        if experience.tools_used:
            lines.append(f"  Tools used: {', '.join(experience.tools_used)}")
        if label == "SUCCESS" and experience.generated_plan:
            numbered = "; ".join(
                f"{index}. {step}" for index, step in enumerate(experience.generated_plan, 1)
            )
            lines.append(f"  Successful workflow: {numbered}")
        if experience.failure_category:
            lines.append(f"  Failure category: {experience.failure_category}")
        if experience.expected_behavior:
            lines.append(f"  Expected behaviour: {experience.expected_behavior}")
        if experience.actual_behavior:
            lines.append(f"  Actual behaviour: {experience.actual_behavior}")
        if experience.user_correction:
            lines.append(f"  User correction: {experience.user_correction}")
        unmet = experience.completion_checks.unmet()
        if unmet:
            lines.append(f"  Unmet requirement(s): {', '.join(unmet)}")
        lines.append(
            "  Guidance: "
            + (
                "reuse this approach where it still matches the current request."
                if label == "SUCCESS"
                else "avoid repeating this approach unless the current request clearly differs."
            )
        )
    lines.append(
        "Requirements stated in the current request (for example a delivery destination) "
        "must be preserved regardless of what previous tasks did."
    )
    return "\n".join(lines)


def format_experience_summary(items: list[RetrievedExperience]) -> str:
    """One-line summary for logs/diagnostics (never contains prompt text)."""

    if not items:
        return "no relevant experience"
    return ", ".join(
        f"{item.experience.outcome}/{item.confidence}({round(item.score, 2)})"
        for item in items
    )


def sanitized_destination(value: Any) -> str:
    """Public helper used by the experience builder for destination text."""

    return sanitize_text(value, limit=120)
