"""Long-term user memory (Milestone 5).

Conversation history answers *"what was said?"*. User memory answers a different
question: *"what does this user durably want Atlas to know?"*

The distinction matters because they have opposite failure modes. A history
record is cheap to keep and expensive to lose; a memory record is expensive to
keep (it will be injected into future prompts) and cheap to lose. So this store
is deliberately conservative:

* Nothing is remembered automatically. Only statements that are *structurally*
  durable are candidates — a stated preference, an explicit instruction to
  remember, a stable fact about the user's setup. "Tell me about Docker" is not
  a memory; "remember that I prefer local-first architecture" is.
* Everything stored is a small, typed record: category, text, provenance
  (which conversation and message it came from), confidence, and timestamps.
  Nothing is inferred about the user and presented as fact.
* Retrieval is bounded and deterministic-with-optional-embeddings, exactly like
  experience memory, and it never calls a model.
* Full user control exists on the backend: view, search, edit, delete, clear,
  and disable. Memory is local by default and stays local.

This package does not replace the existing conversation store, factual RAG, or
experience memory; it is the fourth, separate memory.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config import (
    ENABLE_USER_MEMORY,
    USER_MEMORY_FILE,
    USER_MEMORY_MAX_RECORDS,
    USER_MEMORY_RETRIEVAL_LIMIT,
)

logger = logging.getLogger(__name__)

#: The kinds of things worth remembering long-term. A closed vocabulary keeps
#: "memory" meaningful instead of a dumping ground.
MEMORY_KINDS: frozenset[str] = frozenset(
    {"preference", "instruction", "fact", "project", "workflow"}
)

#: Keys redacted before a candidate is stored. This is a small damage limiter,
#: not a secret detector (see SECURITY.md): the real control is that only
#: explicit, user-stated memory is stored at all.
_SENSITIVE_HINTS: tuple[str, ...] = ("password", "api key", "apikey", "token", "secret", "passphrase")

#: Explicit requests to remember something. These are strong signals: the user
#: asked for persistence in so many words.
_EXPLICIT_REMEMBER: tuple[str, ...] = (
    "remember that", "remember:", "please remember", "keep in mind that",
    "note that i", "don't forget that", "do not forget that", "make a note that",
    "remember i", "remember my", "remember this:",
)

#: Durable preference / instruction framing. Requires first-person ownership so
#: a general statement about the world ("people like Python") is not captured.
_PREFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi\s+(?:always|usually|generally|prefer|like|want|need|hate|dislike)\s+(.{3,160})", re.I),
    re.compile(r"\bi(?:'m| am)\s+working\s+on\s+(.{3,160})", re.I),
    re.compile(r"\bmy\s+(?:preferred|favourite|favorite)\s+([a-z0-9 ._-]{3,80})\s+is\s+(.{1,120})", re.I),
    re.compile(r"\bi\s+use\s+([a-z0-9 ._+-]{2,60})\s+(?:for|as)\s+(.{3,120})", re.I),
    re.compile(r"\bfrom now on\s*,?\s*(.{3,160})", re.I),
    re.compile(r"\balways\s+(.{3,160})", re.I),
    re.compile(r"\bnever\s+(.{3,160})", re.I),
)

_PROJECT_PATTERN = re.compile(r"\b(?:my|our)\s+project\s+(?:is\s+)?(?:called\s+)?([A-Za-z0-9 ._-]{2,60})", re.I)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _has_sensitive(value: str) -> bool:
    lowered = value.casefold()
    return any(hint in lowered for hint in _SENSITIVE_HINTS)


@dataclass
class UserMemory:
    """One durable, user-controlled memory record."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kind: str = "preference"
    text: str = ""
    summary: str = ""
    source: str = "user_message"
    conversation_id: str = ""
    message_id: str = ""
    confidence: float = 0.6
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind if self.kind in MEMORY_KINDS else "fact",
            "text": self.text,
            "summary": self.summary or self.text[:120],
            "source": self.source,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "confidence": round(float(self.confidence), 3),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "UserMemory":
        def _parse(value: Any) -> datetime:
            try:
                return datetime.fromisoformat(str(value))
            except (TypeError, ValueError):
                return _now()

        kind = str(d.get("kind") or "preference")
        return UserMemory(
            id=str(d.get("id") or uuid.uuid4()),
            kind=kind if kind in MEMORY_KINDS else "fact",
            text=_clean(d.get("text")),
            summary=_clean(d.get("summary")),
            source=str(d.get("source") or "user_message"),
            conversation_id=str(d.get("conversation_id") or ""),
            message_id=str(d.get("message_id") or ""),
            confidence=float(d.get("confidence") or 0.6),
            created_at=_parse(d.get("created_at")),
            updated_at=_parse(d.get("updated_at")),
            metadata=dict(d.get("metadata") or {}),
        )


def candidate_memories(
    text: str, *, conversation_id: str = "", message_id: str = ""
) -> list[UserMemory]:
    """Extract *candidate* durable memories from one user message.

    This is intentionally narrow. It only proposes a memory when the sentence
    carries an explicit request to remember or a first-person durable preference
    or instruction. A question, a task request, or a passing remark produces
    nothing, so ordinary conversation never becomes permanent memory.

    Extraction is *whole-message*, not per-pattern: one statement produces one
    memory even when several patterns match it ("remember that I prefer X"
    matches both the explicit-remember rule and the preference rule). Otherwise
    a single sentence would be stored twice under different kinds.
    """

    cleaned = _clean(text)
    if len(cleaned) < 4:
        return []
    lowered = cleaned.casefold()

    # 1. An explicit request to remember wins outright: whatever the user asked
    #    Atlas to keep is the memory, and only that.
    for phrase in _EXPLICIT_REMEMBER:
        index = lowered.find(phrase)
        if index < 0:
            continue
        value = _clean(cleaned[index + len(phrase):]).strip(" .,:;-")
        if len(value) < 4 or _has_sensitive(value):
            return []
        return [
            UserMemory(
                kind="instruction",
                text=value,
                summary=value[:120],
                conversation_id=conversation_id,
                message_id=message_id,
                confidence=0.9,
                metadata={"trigger": "explicit"},
            )
        ]

    # 2. Otherwise, the first durable preference/instruction pattern that fires
    #    is the memory. Competing patterns on the same sentence are the same
    #    statement restated, not several facts.
    for pattern in _PREFERENCE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        groups = [group for group in match.groups() if group]
        value = " ".join(groups) if groups else match.group(0)
        value = _clean(value).strip(" .,:;-")
        if len(value) < 4 or _has_sensitive(value):
            return []
        kind = "preference" if match.group(0).casefold().startswith("i ") else "instruction"
        return [
            UserMemory(
                kind=kind,
                text=value,
                summary=value[:120],
                conversation_id=conversation_id,
                message_id=message_id,
                confidence=0.72,
                metadata={"trigger": "pattern"},
            )
        ]

    # 3. A named project is durable even without a preference construction.
    project = _PROJECT_PATTERN.search(cleaned)
    if project:
        value = f"Project: {_clean(project.group(1)).strip(' .,:;-')}"
        if len(value) > 12 and not _has_sensitive(value):
            return [
                UserMemory(
                    kind="project",
                    text=value,
                    summary=value[:120],
                    conversation_id=conversation_id,
                    message_id=message_id,
                    confidence=0.7,
                    metadata={"trigger": "pattern"},
                )
            ]
    return []


class UserMemoryEmbedder:
    """Optional embeddings for user memory, in their own collection.

    Reuses the existing ChromaDB directory and embedding model, in a separate
    collection tagged ``memory_type="user_memory"`` so a preference can never be
    retrieved as a factual document.
    """

    def __init__(self, *, persist_dir: str | None = None, collection_name: str | None = None) -> None:
        from config import DATABASE_FOLDER, USER_MEMORY_COLLECTION_NAME

        self._persist_dir = persist_dir or str(DATABASE_FOLDER)
        self._collection_name = collection_name or USER_MEMORY_COLLECTION_NAME
        self._collection: Any = None

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection if self._collection is not False else None
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = client.get_or_create_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - optional by design
            logger.info("User-memory embeddings unavailable; using lexical retrieval")
            self._collection = False
        return self._collection if self._collection is not False else None

    def index(self, memory: UserMemory) -> bool:
        collection = self._ensure_collection()
        if collection is None:
            return False
        try:
            collection.upsert(
                ids=[memory.id],
                documents=[memory.text],
                metadatas=[{
                    "memory_type": "user_memory",
                    "kind": memory.kind,
                    "memory_id": memory.id,
                }],
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    def similarities(self, query: str, ids: Iterable[str]) -> dict[str, float]:
        identifiers = [identifier for identifier in ids if identifier]
        collection = self._ensure_collection()
        if collection is None or not identifiers:
            return {}
        try:
            result = collection.query(
                query_texts=[query],
                n_results=min(len(identifiers), 25),
                where={"$and": [{"memory_type": "user_memory"}, {"memory_id": {"$in": identifiers}}]},
                include=["distances"],
            )
        except Exception:  # noqa: BLE001
            return {}
        distances = (result.get("distances") or [[]])[0]
        ordered = (result.get("ids") or [[]])[0]
        return {
            identifier: max(0.0, 1.0 - float(distance) / 2.0)
            for identifier, distance in zip(ordered, distances)
        }

    def delete(self, memory_id: str) -> None:
        collection = self._ensure_collection()
        if collection is None:
            return
        try:
            collection.delete(ids=[memory_id])
        except Exception:  # noqa: BLE001
            logger.debug("Could not remove memory %s from the index", memory_id, exc_info=True)


class UserMemoryStore:
    """Durable, bounded, user-controlled long-term memory."""

    def __init__(
        self,
        *,
        path: Path | str = USER_MEMORY_FILE,
        enabled: bool = ENABLE_USER_MEMORY,
        limit: int = USER_MEMORY_MAX_RECORDS,
        embedder: UserMemoryEmbedder | None = None,
    ) -> None:
        self._path = Path(path)
        self._enabled = bool(enabled)
        self._limit = max(1, int(limit))
        self._embedder = embedder if embedder is not None else UserMemoryEmbedder()

    # -- lifecycle ---------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable user memory. Disabling stops writes and retrievals."""

        self._enabled = bool(enabled)

    def _load(self) -> list[UserMemory]:
        if not self._path.exists():
            return []
        records: list[UserMemory] = []
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(UserMemory.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        except OSError:
            logger.exception("Could not read the user memory store")
            return []
        return records

    def _save(self, records: list[UserMemory]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records[-self._limit:]:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        temporary.replace(self._path)

    # -- CRUD --------------------------------------------------------------------

    def create(
        self,
        *,
        text: str,
        kind: str = "preference",
        conversation_id: str = "",
        message_id: str = "",
        confidence: float = 0.7,
        source: str = "user_message",
    ) -> UserMemory | None:
        cleaned = _clean(text)
        if not self._enabled or not cleaned or _has_sensitive(cleaned):
            return None
        records = self._load()
        # An identical memory is updated (and touched), not duplicated.
        for existing in records:
            if existing.text.casefold() == cleaned.casefold():
                existing.updated_at = _now()
                existing.confidence = max(existing.confidence, float(confidence))
                self._save(records)
                self._embedder.index(existing)
                return existing
        memory = UserMemory(
            kind=kind if kind in MEMORY_KINDS else "fact",
            text=cleaned,
            summary=cleaned[:120],
            conversation_id=conversation_id,
            message_id=message_id,
            confidence=float(confidence),
            source=source,
        )
        records.append(memory)
        self._save(records)
        self._embedder.index(memory)
        logger.info("User memory stored: kind=%s id=%s", memory.kind, memory.id)
        return memory

    def list(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._load()]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        for record in self._load():
            if record.id == memory_id:
                return record.to_dict()
        return None

    def update(self, memory_id: str, *, text: str | None = None, kind: str | None = None) -> dict[str, Any] | None:
        records = self._load()
        for record in records:
            if record.id != memory_id:
                continue
            if text is not None:
                cleaned = _clean(text)
                if not cleaned or _has_sensitive(cleaned):
                    return None
                record.text = cleaned
                record.summary = cleaned[:120]
            if kind is not None and kind in MEMORY_KINDS:
                record.kind = kind
            record.updated_at = _now()
            self._save(records)
            self._embedder.index(record)
            return record.to_dict()
        return None

    def delete(self, memory_id: str) -> bool:
        records = self._load()
        remaining = [record for record in records if record.id != memory_id]
        if len(remaining) == len(records):
            return False
        self._save(remaining)
        self._embedder.delete(memory_id)
        return True

    def clear(self) -> int:
        records = self._load()
        for record in records:
            self._embedder.delete(record.id)
        self._save([])
        return len(records)

    # -- observation and retrieval -------------------------------------------------

    def observe_user_message(
        self, text: str, *, conversation_id: str = "", message_id: str = ""
    ) -> list[dict[str, Any]]:
        """Store the durable part of a user message, if it has one.

        Called for every user turn, but almost always a no-op: see
        :func:`candidate_memories`. Returning an empty list is the normal case.
        """

        if not self._enabled:
            return []
        stored: list[dict[str, Any]] = []
        for candidate in candidate_memories(
            text, conversation_id=conversation_id, message_id=message_id
        ):
            record = self.create(
                text=candidate.text,
                kind=candidate.kind,
                conversation_id=conversation_id,
                message_id=message_id,
                confidence=candidate.confidence,
                source=candidate.source,
            )
            if record is not None:
                stored.append(record.to_dict())
        return stored

    def retrieve(self, query: str, *, limit: int = USER_MEMORY_RETRIEVAL_LIMIT) -> list[dict[str, Any]]:
        """Return bounded, ranked memories relevant to ``query``."""

        if not self._enabled:
            return []
        records = self._load()
        if not records:
            return []
        query_tokens = _query_tokens(query)
        embeddings = self._embedder.similarities(query, [record.id for record in records])
        scored: list[tuple[float, UserMemory]] = []
        for record in records:
            lexical = _overlap(query_tokens, _query_tokens(record.text))
            semantic = embeddings.get(record.id, 0.0)
            # A stored preference is not a search result: it has a baseline
            # relevance so a durable instruction ("always answer briefly") still
            # applies to a turn that shares no vocabulary with it.
            base = 0.35 if record.kind in {"instruction", "preference"} else 0.2
            score = max(semantic, lexical) * 0.7 + base + 0.05 * record.confidence
            if record.kind in {"instruction", "preference"}:
                score += 0.15
            scored.append((min(score, 1.0), record))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[UserMemory] = []
        seen: set[str] = set()
        for score, record in scored:
            if record.id in seen or score < 0.3:
                continue
            seen.add(record.id)
            selected.append(record)
            if len(selected) >= max(1, int(limit)):
                break
        return [
            {**record.to_dict(), "score": round(score, 4)}
            for score, record in scored
            if record.id in seen
        ][: max(1, int(limit))]

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Deterministic full search for the memory-management API."""

        needle = _clean(query).casefold()
        if not needle:
            return self.list()[:limit]
        return [
            record.to_dict()
            for record in self._load()
            if needle in record.text.casefold() or needle in record.summary.casefold()
        ][:limit]


def _query_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").casefold())
    return {word for word in words if len(word) > 2}


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
