"""Durable storage for experiences and pending feedback.

Both kinds of record are rows in the authoritative ``atlas.db``:

* ``experiences`` — one evaluated :class:`experience.models.Experience` per row.
  The store keeps a bounded history: past the configured cap the oldest records
  are trimmed, which is a cheap SQL predicate rather than a whole-file rewrite.
* ``feedback_events`` — one feedback event per row, recording what the user
  clicked *before* the record is evaluated. Events are only ever appended, so a
  click stays durable even if evaluation is skipped, and "most recent feedback
  for this record" is an indexed lookup rather than a scan.

This module does no retrieval, ranking, or reasoning, and it never executes
anything. It exists so the loop survives an application restart (spec sections
22 and 24).

Storage moved from ``database/experiences.jsonl`` and ``database/feedback.jsonl``
to SQLite; those files were a database in disguise and are imported once by
:mod:`persistence.migration`. This class is a thin, model-aware façade over
:class:`persistence.repositories.SqliteExperienceRepository` and its feedback
counterpart — it holds no SQL and no file format of its own.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import EXPERIENCE_MAX_RECORDS
from experience.models import Experience, sanitize_text, utc_now
from persistence.context import database_for_legacy_path, default_persistence
from persistence.database import AtlasDatabase
from persistence.repositories import SqliteExperienceRepository, SqliteFeedbackRepository

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


class ExperienceStore:
    """Persistent store for experiences and feedback events.

    Records live in ``atlas.db``; retrieval is a bounded SQL query rather than a
    whole-file read. The store caches the in-memory list so a retrieval never
    re-queries for every message, and it is thread-safe (the API runs tasks on a
    worker thread while the request thread may record feedback).
    """

    def __init__(
        self,
        *,
        experience_path: Path | None = None,
        feedback_path: Path | None = None,
        max_records: int = EXPERIENCE_MAX_RECORDS,
        repository: SqliteExperienceRepository | None = None,
        feedback_repository: SqliteFeedbackRepository | None = None,
        database: AtlasDatabase | None = None,
    ) -> None:
        self._owned_database = database
        self._owns_database = False
        if repository is not None or feedback_repository is not None:
            # An injected repository: its owner controls the database lifecycle.
            self._experiences_repo = repository
            self._feedback_repo = feedback_repository
        elif database is not None:
            self._experiences_repo = SqliteExperienceRepository(database)
            self._feedback_repo = SqliteFeedbackRepository(database)
        elif experience_path is not None or feedback_path is not None:
            # Legacy constructor form: derive an isolated database from the given
            # path so a test/tool never touches the process-wide ``atlas.db``.
            anchor = experience_path or feedback_path
            self._owned_database = database_for_legacy_path(anchor)
            self._owns_database = True
            self._experiences_repo = SqliteExperienceRepository(self._owned_database)
            self._feedback_repo = SqliteFeedbackRepository(self._owned_database)
        else:
            context = default_persistence()
            self._experiences_repo = context.experiences
            self._feedback_repo = context.feedback

        if self._experiences_repo is None:
            raise ValueError("ExperienceStore needs an experience repository or a database")
        if self._feedback_repo is None:
            raise ValueError("ExperienceStore needs a feedback repository or a database")

        self._experience_path = Path(experience_path) if experience_path is not None else None
        self._feedback_path = Path(feedback_path) if feedback_path is not None else None
        self._max_records = max(1, int(max_records))
        self._lock = threading.RLock()
        self._experiences: list[Experience] | None = None
        self._feedback: list[FeedbackEvent] | None = None

    # -- identity / lifecycle ------------------------------------------------

    @property
    def experience_repository(self) -> SqliteExperienceRepository:
        return self._experiences_repo

    @property
    def feedback_repository(self) -> SqliteFeedbackRepository:
        return self._feedback_repo

    def close(self) -> None:
        """Release this store's SQLite connection when it owns the database."""

        if self._owns_database and self._owned_database is not None:
            self._owned_database.close()

    def __del__(self) -> None:  # pragma: no cover - GC timing is not deterministic
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            pass

    def __enter__(self) -> "ExperienceStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def experiences(self) -> list[Experience]:
        """Return all stored experiences (loaded once, then cached)."""

        with self._lock:
            if self._experiences is None:
                loaded: list[Experience] = []
                for row in self._experiences_repo.list_experiences():
                    parsed = Experience.from_dict(row)
                    if parsed is not None:
                        loaded.append(parsed)
                self._experiences = loaded
            return self._experiences

    def add(self, experience: Experience) -> bool:
        """Store one experience record and update the cache."""

        with self._lock:
            # Cache is populated FIRST so a subsequent read cannot count the row
            # this call just inserted twice.
            self.experiences()
            if not self._experiences_repo.add_experience(experience.to_dict()):
                return False
            # Bound the history: drop the oldest records past the cap. SQL does
            # the trimming, so this is not a whole-store rewrite.
            self._experiences_repo.trim_to(self._max_records)
            self._experiences = None
            return True

    def replace(self, experience: Experience) -> bool:
        """Replace a record in place (used for promotion/confirmation updates)."""

        with self._lock:
            if not self._experiences_repo.replace_experience(experience.to_dict()):
                return False
            self._experiences = None
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
                for row in self._feedback_repo.list_feedback():
                    parsed = FeedbackEvent.from_dict(row)
                    if parsed is not None:
                        loaded.append(parsed)
                self._feedback = loaded
            return self._feedback

    def record_feedback(self, event: FeedbackEvent) -> bool:
        with self._lock:
            # Same ordering rule as ``add``: read the cache before writing.
            self.feedback_events()
            if not self._feedback_repo.record_feedback(event.to_dict()):
                return False
            self._feedback = None
            return True

    def latest_feedback(self, record_id: str) -> FeedbackEvent | None:
        """Return the most recent feedback for a record (changeable feedback)."""

        stored = self._feedback_repo.latest_feedback(record_id)
        return FeedbackEvent.from_dict(stored) if stored is not None else None

    def clear_cache(self) -> None:
        """Drop the in-memory caches so the next read reflects the database."""

        with self._lock:
            self._experiences = None
            self._feedback = None
