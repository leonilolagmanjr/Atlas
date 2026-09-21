"""Structured memory of interpretation corrections and recurring task patterns.

Atlas does not train the model. Instead it stores a small, curated set of
*structured* lessons that describe how a class of phrasings should be
interpreted, so the same misreading is not repeated across requests.

Design rules:

* Only reusable, high-confidence lessons are stored — never every conversation
  turn, and never an uncontrolled vector dump.
* A lesson is data with an explicit ``type`` and a ``pattern``; lookups are
  deterministic substring/pattern matches, not embeddings.
* Storage is append-only JSONL under ``memory/`` and is best-effort: a missing
  or corrupt file degrades to "no lessons" and never raises into a request.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Where correction lessons live. Kept beside the other memory artifacts.
DEFAULT_LESSONS_PATH: Path = PROJECT_ROOT / "memory" / "interpretation_lessons.jsonl"

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

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_LESSONS_PATH
        self._lessons: list[Lesson] | None = None

    # -- reading -----------------------------------------------------------------

    def lessons(self) -> list[Lesson]:
        if self._lessons is None:
            self._lessons = self._load()
        return self._lessons

    def _load(self) -> list[Lesson]:
        try:
            if not self._path.exists():
                return []
            loaded: list[Lesson] = []
            for raw_line in self._path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    lesson = Lesson.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue
                if lesson is not None:
                    loaded.append(lesson)
            return loaded
        except OSError:
            logger.warning("Could not read interpretation lessons at %s", self._path)
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
        """Append a single high-confidence lesson. Returns True when stored."""

        if lesson.confidence < self._MIN_CONFIDENCE or not lesson.pattern:
            return False
        if any(
            existing.type == lesson.type and existing.pattern == lesson.pattern
            for existing in self.lessons()
        ):
            return False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("Could not persist interpretation lesson to %s", self._path)
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
