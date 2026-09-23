"""Durable storage for experiences and pending feedback.

Two append-only JSONL files live under ``database/``:

* ``experiences.jsonl`` — one evaluated :class:`experience.models.Experience` per
  line. Writing is append-only, so a crash mid-write can never corrupt earlier
  records, and reading skips any unparsable line instead of raising.
* ``feedback.jsonl`` — one feedback event per line, recording what the user
  clicked *before* the record is evaluated. This is what makes a click durable
  even if evaluation is skipped, and it also gives us the "prevent accidental
  duplicate feedback" history.

This module does no retrieval, ranking, or reasoning, and it never executes
anything. It exists so the loop survives an application restart (spec sections
22 and 24).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from config import EXPERIENCE_MAX_RECORDS, EXPERIENCE_STORE_FILE, FEEDBACK_STORE_FILE
from experience.models import Experience, sanitize_text, utc_now

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    """One user feedback click, recorded durably and immediately."""

    record_id: str
    outcome: str
    reason: str = ""
    failure_category: str = ""
    correction: str = ""
    expected_behavior: str = ""
    response_quality: str = ""
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "failure_category": self.failure_category,
            "correction": self.correction,
            "expected_behavior": self.expected_behavior,
            "response_quality": self.response_quality,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "FeedbackEvent | None":
        if not isinstance(data, dict):
            return None
        record_id = sanitize_text(data.get("record_id"), limit=64)
        if not record_id:
            return None
        return cls(
            record_id=record_id,
            outcome=sanitize_text(data.get("outcome"), limit=32) or "unknown",
            reason=sanitize_text(data.get("reason"), limit=240),
            failure_category=sanitize_text(data.get("failure_category"), limit=64),
            correction=sanitize_text(data.get("correction")),
            expected_behavior=sanitize_text(data.get("expected_behavior"), limit=500),
            response_quality=sanitize_text(data.get("response_quality"), limit=32),
            timestamp=sanitize_text(data.get("timestamp"), limit=64) or utc_now(),
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping unparsable lines. Never raises on read."""

    try:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows
    except OSError:
        logger.warning("Could not read experience storage at %s", path)
        return []


def _append_jsonl(path: Path, payload: dict[str, Any]) -> bool:
    """Append one JSON object as a line. Best-effort; returns True on success."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except (OSError, TypeError, ValueError):
        logger.warning("Could not persist to %s", path)
        return False


class ExperienceStore:
    """Append-only persistent store for experiences and feedback events.

    The store is thread-safe (the API runs tasks on a worker thread while the
    request thread may record feedback) and it caches the in-memory list so a
    retrieval never re-reads the whole file for every message.
    """

    def __init__(
        self,
        *,
        experience_path: Path | None = None,
        feedback_path: Path | None = None,
        max_records: int = EXPERIENCE_MAX_RECORDS,
    ) -> None:
        self._experience_path = Path(experience_path or EXPERIENCE_STORE_FILE)
        self._feedback_path = Path(feedback_path or FEEDBACK_STORE_FILE)
        self._max_records = max(1, int(max_records))
        self._lock = threading.RLock()
        self._experiences: list[Experience] | None = None
        self._feedback: list[FeedbackEvent] | None = None

    # -- experiences ---------------------------------------------------------

    def experiences(self) -> list[Experience]:
        """Return all stored experiences (loaded once, then cached)."""

        with self._lock:
            if self._experiences is None:
                loaded: list[Experience] = []
                for row in _read_jsonl(self._experience_path):
                    parsed = Experience.from_dict(row)
                    if parsed is not None:
                        loaded.append(parsed)
                self._experiences = loaded
            return self._experiences

    def _rewrite_experiences(self, experiences: Iterable[Experience]) -> None:
        """Atomically rewrite the whole store (used only for bounded compaction)."""

        import os
        import tempfile

        records = list(experiences)
        try:
            self._experience_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f"{self._experience_path.stem}-",
                suffix=".tmp",
                dir=self._experience_path.parent,
                text=True,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self._experience_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        except OSError:
            logger.warning("Could not compact experience store at %s", self._experience_path)

    def add(self, experience: Experience) -> bool:
        """Append one experience record and update the cache."""

        with self._lock:
            # Materialize the cache BEFORE writing, otherwise the lazy load would
            # read back the line this call just appended and count it twice.
            current = self.experiences()
            if not _append_jsonl(self._experience_path, experience.to_dict()):
                return False
            stored = [*current, experience]
            # Bound the history: drop the oldest records past the cap. This is a
            # rare, explicit compaction, never a per-message full scan.
            if len(stored) > self._max_records:
                stored = stored[-self._max_records:]
                self._experiences = stored
                self._rewrite_experiences(stored)
            else:
                self._experiences = stored
            return True

    def replace(self, experience: Experience) -> bool:
        """Replace a record in place (used for promotion/confirmation updates)."""

        with self._lock:
            current = self.experiences()
            replaced = False
            updated: list[Experience] = []
            for existing in current:
                if existing.experience_id == experience.experience_id:
                    updated.append(experience)
                    replaced = True
                else:
                    updated.append(existing)
            if not replaced:
                return False
            self._experiences = updated
            self._rewrite_experiences(updated)
            return True

    def find_by_id(self, experience_id: str) -> Experience | None:
        for experience in self.experiences():
            if experience.experience_id == experience_id:
                return experience
        return None

    def find_for_record(self, record_id: str) -> list[Experience]:
        """Return every experience derived from one API task record."""

        if not record_id:
            return []
        return [
            experience
            for experience in self.experiences()
            if experience.record_id == record_id
        ]

    def counts(self) -> dict[str, int]:
        experiences = self.experiences()
        return {
            "total": len(experiences),
            "success": sum(1 for item in experiences if item.is_success()),
            "failure": sum(1 for item in experiences if item.is_failure()),
            "reliable": sum(1 for item in experiences if item.state == "reliable"),
            "unevaluated": sum(1 for item in experiences if item.outcome == "unknown"),
        }

    # -- feedback events -----------------------------------------------------

    def feedback_events(self) -> list[FeedbackEvent]:
        with self._lock:
            if self._feedback is None:
                loaded: list[FeedbackEvent] = []
                for row in _read_jsonl(self._feedback_path):
                    parsed = FeedbackEvent.from_dict(row)
                    if parsed is not None:
                        loaded.append(parsed)
                self._feedback = loaded
            return self._feedback

    def record_feedback(self, event: FeedbackEvent) -> bool:
        with self._lock:
            # Same ordering rule as ``add``: read the cache before appending.
            current = self.feedback_events()
            if not _append_jsonl(self._feedback_path, event.to_dict()):
                return False
            self._feedback = [*current, event]
            return True

    def latest_feedback(self, record_id: str) -> FeedbackEvent | None:
        """Return the most recent feedback for a record (changeable feedback)."""

        latest: FeedbackEvent | None = None
        for event in self.feedback_events():
            if event.record_id == record_id:
                latest = event
        return latest

    def clear_cache(self) -> None:
        """Drop the in-memory caches so the next read reflects the file."""

        with self._lock:
            self._experiences = None
            self._feedback = None
