"""Structured memory of interpretation corrections and recurring task patterns.

Atlas does not train the model. Instead it stores a small, curated set of
*structured* lessons that describe how a class of phrasings should be
interpreted, so the same misreading is not repeated across requests.

Design rules:

* Only reusable, high-confidence lessons are stored — never every conversation
  turn, and never an uncontrolled vector dump.
* A lesson is data with an explicit ``type`` and a ``pattern``; lookups are
  deterministic substring/pattern matches, not embeddings.
* Storage is the authoritative ``correction_lessons`` table in ``atlas.db``, and
  is best-effort: a database error degrades to "no lessons" and never raises
  into a request.

``memory/interpretation_lessons.jsonl`` is imported once by
:mod:`persistence.migration`; it is no longer read or written at runtime.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from persistence.repositories import SqliteCorrectionMemoryRepository

logger = logging.getLogger(__name__)

#: Lesson types Atlas understands.
LESSON_TYPES: frozenset[str] = frozenset(
    {"interpretation_correction", "task_pattern"}
)


@dataclass
class Lesson:
    """A single structured, reusable interpretation lesson."""

    type: str
    pattern: str
    #: The behavior the interpreter should produce for this pattern's class.
    correct_behavior: list[str] = field(default_factory=list)
    #: Optional explicit fields that win when the pattern matches.
    fields: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "pattern": self.pattern,
            "correct_behavior": list(self.correct_behavior),
            "fields": dict(self.fields),
            "confidence": self.confidence,
            "hits": self.hits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Lesson | None":
        if not isinstance(data, dict):
            return None
        lesson_type = str(data.get("type") or "").strip()
        pattern = str(data.get("pattern") or "").strip()
        if lesson_type not in LESSON_TYPES or not pattern:
            return None
        behavior = data.get("correct_behavior")
        if isinstance(behavior, str):
            behavior = [behavior]
        try:
            confidence = float(data.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            type=lesson_type,
            pattern=pattern,
            correct_behavior=[str(item) for item in behavior] if isinstance(behavior, list) else [],
            fields=data.get("fields") if isinstance(data.get("fields"), dict) else {},
            confidence=max(0.0, min(1.0, confidence)),
            hits=int(data.get("hits") or 0) if str(data.get("hits") or "0").isdigit() else 0,
        )


class CorrectionMemory:
    """Load, query, and append structured interpretation lessons.

    Only lessons whose pattern matches a request and whose confidence is at or
    above :meth:`_MIN_CONFIDENCE` are returned by :meth:`match`.
    """

    #: Lessons below this confidence are never applied automatically.
    _MIN_CONFIDENCE = 0.8

    def __init__(
        self,
        path: Path | None = None,
        *,
        repository: SqliteCorrectionMemoryRepository | None = None,
    ) -> None:
        # ``path`` is accepted for backward compatibility with callers that used
        # to point at a JSONL file. It is retained as documentation of origin;
        # lessons now live in ``atlas.db``.
        self._path = Path(path) if path is not None else None
        self._repository = repository
        self._lessons: list[Lesson] | None = None

    @property
    def repository(self) -> SqliteCorrectionMemoryRepository:
        if self._repository is None:
            from persistence.context import default_persistence

            self._repository = default_persistence().correction_memory
        return self._repository

    # -- reading -----------------------------------------------------------------

    def lessons(self) -> list[Lesson]:
        if self._lessons is None:
            self._lessons = self._load()
        return self._lessons

    def _load(self) -> list[Lesson]:
        try:
            loaded: list[Lesson] = []
            for row in self.repository.list_lessons():
                lesson = Lesson.from_dict(row)
                if lesson is not None:
                    loaded.append(lesson)
            return loaded
        except Exception:  # noqa: BLE001 - a read failure means "no lessons"
            logger.warning("Could not read interpretation lessons", exc_info=True)
            return []

    def match(self, text: str) -> list[Lesson]:
        """Return the high-confidence lessons whose pattern matches ``text``."""

        lowered = text.casefold()
        matched: list[Lesson] = []
        for lesson in self.lessons():
            if lesson.confidence < self._MIN_CONFIDENCE:
                continue
            if _pattern_matches(lesson.pattern, lowered):
                matched.append(lesson)
        return matched

    # -- writing -----------------------------------------------------------------

    def record(self, lesson: Lesson) -> bool:
        """Store a single high-confidence lesson. Returns True when stored.

        An identical lesson (same type and pattern) is never taught twice; the
        repository enforces that with a unique constraint, so re-recording is a
        no-op rather than a duplicate.
        """

        if lesson.confidence < self._MIN_CONFIDENCE or not lesson.pattern:
            return False
        if any(
            existing.type == lesson.type and existing.pattern == lesson.pattern
            for existing in self.lessons()
        ):
            return False
        try:
            stored = self.repository.add_lesson(lesson.to_dict())
        except Exception:  # noqa: BLE001 - a write failure must not break a request
            logger.warning("Could not persist interpretation lesson", exc_info=True)
            return False
        if not stored:
            return False
        self._lessons = [*self.lessons(), lesson]
        return True

    def record_correction(
        self,
        *,
        pattern: str,
        correct_behavior: list[str],
        fields: dict[str, Any] | None = None,
        confidence: float = 0.9,
    ) -> bool:
        """Convenience wrapper for an ``interpretation_correction`` lesson."""

        return self.record(
            Lesson(
                type="interpretation_correction",
                pattern=pattern,
                correct_behavior=correct_behavior,
                fields=fields or {},
                confidence=confidence,
            )
        )


def _pattern_matches(pattern: str, lowered_text: str) -> bool:
    """Match a lesson pattern: a regex if it compiles, else a substring test."""

    pattern = pattern.strip()
    if not pattern:
        return False
    try:
        return bool(re.search(pattern, lowered_text, re.IGNORECASE))
    except re.error:
        return pattern.casefold() in lowered_text
