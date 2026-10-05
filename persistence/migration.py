"""One-time, idempotent import of Atlas's legacy JSON/JSONL state into SQLite.

Before SQLite, Atlas kept structured state in loose files under ``database/`` and
``memory/``::

    database/tasks.json
    database/user_memory.jsonl
    database/experiences.jsonl
    database/feedback.jsonl
    database/index_state.json
    memory/active_session.json
    memory/sessions/<id>/{metadata,messages}.json, summary.txt
    memory/interpretation_lessons.jsonl

Those files were a database in disguise. This module imports them into
``atlas.db`` once, then leaves them exactly where they were — nothing is deleted.

Guarantees:

* **Idempotent.** Each source is fingerprinted and recorded in the
  ``legacy_migrations`` table. Running the migration three times imports nothing
  the second or third time.
* **Atomic.** A source's rows and its bookkeeping row are written in one
  transaction, so an interrupted run leaves no partial state.
* **Non-destructive.** Legacy files are read, never moved or removed. (The one
  exception is the Chroma vector directory, which is *relocated* — not deleted —
  from ``database/`` to its dedicated ``chroma/`` directory.)
* **Honest about failure.** Malformed JSON lines are skipped and counted; a
  missing file is simply a no-op.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from persistence.database import AtlasDatabase
from persistence.paths import AtlasDataPaths
from persistence.repositories import (
    SqliteConversationRepository,
    SqliteCorrectionMemoryRepository,
    SqliteExperienceRepository,
    SqliteFeedbackRepository,
    SqliteIndexStateRepository,
    SqliteTaskRepository,
    SqliteUserMemoryRepository,
    utc_now,
)

logger = logging.getLogger(__name__)

#: The legacy layout this module owns. ``config`` exposes the same names as paths
#: for documentation and for path-exclusion rules, but the migrator is the only
#: code that reads them, so the filenames are declared here too — importing
#: ``config`` from inside ``persistence`` would create an import cycle.
LEGACY_TASK_STORE_FILE = "tasks.json"
LEGACY_USER_MEMORY_FILE = "user_memory.jsonl"
LEGACY_EXPERIENCE_STORE_FILE = "experiences.jsonl"
LEGACY_FEEDBACK_STORE_FILE = "feedback.jsonl"
LEGACY_INDEX_STATE_FILE = "index_state.json"
LEGACY_LESSONS_FILE = "interpretation_lessons.jsonl"

#: Legacy structured-state files that must *not* be relocated as Chroma data.
LEGACY_DATA_FILES = frozenset(
    {
        LEGACY_TASK_STORE_FILE,
        LEGACY_USER_MEMORY_FILE,
        LEGACY_EXPERIENCE_STORE_FILE,
        LEGACY_FEEDBACK_STORE_FILE,
        LEGACY_INDEX_STATE_FILE,
        LEGACY_LESSONS_FILE,
        "atlas.log",
        "atlas.db",
    }
)


class MigrationValidationError(RuntimeError):
    """Raised when imported record counts do not match the legacy source."""


@dataclass
class MigrationReport:
    """What the migration did, per source (useful for logs and tests)."""

    applied: dict[str, dict[str, int]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    @property
    def imported_total(self) -> int:
        return sum(summary.get("imported", 0) for summary in self.applied.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "skipped": list(self.skipped),
            "imported_total": self.imported_total,
        }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Legacy migration: could not parse %s", path)
        return None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read a JSONL file, skipping (and counting) unparsable lines."""

    rows: list[dict[str, Any]] = []
    failed = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    failed += 1
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
                else:
                    failed += 1
    except OSError:
        logger.warning("Legacy migration: could not read %s", path)
    return rows, failed


def _fingerprint(items: list[Path]) -> str:
    """A stable fingerprint over a set of files (path + size + content hash)."""

    digest = hashlib.sha256()
    for path in sorted(items, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


class LegacyMigrator:
    """Import legacy JSON/JSONL state into ``atlas.db`` exactly once."""

    def __init__(self, database: AtlasDatabase, paths: AtlasDataPaths) -> None:
        self._db = database
        self._paths = paths
        self._conversations = SqliteConversationRepository(database)
        self._tasks = SqliteTaskRepository(database)
        self._user_memory = SqliteUserMemoryRepository(database)
        self._experiences = SqliteExperienceRepository(database)
        self._feedback = SqliteFeedbackRepository(database)
        self._index_state = SqliteIndexStateRepository(database)
        self._corrections = SqliteCorrectionMemoryRepository(database)

    # -- legacy locations ------------------------------------------------------

    def _legacy_roots(self) -> list[Path]:
        """Where legacy files may live.

        Exactly one place: the data root. In a development run that *is* the
        application root, so a checkout keeping ``database/`` and ``memory/``
        beside the source is covered. A separate root (``%LOCALAPPDATA%\\Atlas``,
        or a test's temporary directory) must never reach back into an
        installation directory: those files belong to a different Atlas, and
        importing them would be wrong.
        """

        return [self._paths.root]

    def _find(self, relative: str) -> Path | None:
        for root in self._legacy_roots():
            candidate = root / relative
            if candidate.exists():
                return candidate
        return None

    @property
    def _legacy_db_dir(self) -> Path | None:
        return self._find("database")

    @property
    def _legacy_memory_dir(self) -> Path | None:
        return self._find("memory")

    # -- bookkeeping -----------------------------------------------------------

    def _apply_if_changed(self, source: str, fingerprint: str) -> bool:
        """True when this source still needs importing (new or changed)."""

        with self._db.connection() as connection:
            row = connection.execute(
                "SELECT fingerprint FROM legacy_migrations WHERE source = ?", (source,)
            ).fetchone()
        return row is None or str(row["fingerprint"]) != fingerprint

    def _record(
        self,
        source: str,
        fingerprint: str,
        *,
        imported: int = 0,
        skipped: int = 0,
        failed: int = 0,
        detail: str = "",
    ) -> None:
        with self._db.transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO legacy_migrations "
                "(source, fingerprint, imported, skipped, failed, detail, applied_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (source, fingerprint, imported, skipped, failed, detail, utc_now()),
            )

    # -- entry point -----------------------------------------------------------

    def run(self) -> MigrationReport:
        """Run every import; safe to call repeatedly."""

        report = MigrationReport()
        self._relocate_chroma(report)
        for source, handler in (
            ("tasks", self._import_tasks),
            ("user_memory", self._import_user_memory),
            ("experiences", self._import_experiences),
            ("feedback", self._import_feedback),
            ("index_state", self._import_index_state),
            ("correction_memory", self._import_correction_memory),
            ("conversations", self._import_conversations),
        ):
            try:
                summary = handler()
            except Exception:  # noqa: BLE001 - a broken import must not block startup
                logger.exception("Legacy migration failed for %s", source)
                continue
            if summary is None:
                continue
            if summary.get("skipped"):
                report.skipped.append(source)
            else:
                report.applied[source] = summary
        return report

    # -- Chroma relocation -----------------------------------------------------

    def _relocate_chroma(self, report: MigrationReport) -> None:
        """Move legacy Chroma artifacts out of ``database/`` into ``chroma/``.

        Nothing is deleted: the files are moved to their dedicated directory so
        the relational database and the vector index stop sharing a folder.
        """

        legacy_dir = self._legacy_db_dir
        if legacy_dir is None:
            return
        target = self._paths.chroma_dir
        if (target / "chroma.sqlite3").exists():
            return
        movable = [entry for entry in legacy_dir.iterdir() if entry.name not in LEGACY_DATA_FILES]
        if not any(entry.name == "chroma.sqlite3" for entry in movable):
            return
        target.mkdir(parents=True, exist_ok=True)
        moved = 0
        for entry in movable:
            destination = target / entry.name
            if destination.exists():
                continue
            try:
                shutil.move(str(entry), str(destination))
                moved += 1
            except OSError:
                logger.warning("Could not relocate legacy Chroma entry %s", entry)
        self._record("chroma_relocation", "relocated", imported=moved)
        report.applied["chroma_relocation"] = {"imported": moved}

    # -- imports ---------------------------------------------------------------

    def _import_tasks(self) -> dict[str, int] | None:
        path = self._legacy_file(LEGACY_TASK_STORE_FILE)
        if path is None:
            return None
        fingerprint = _fingerprint([path])
        if not self._apply_if_changed("tasks", fingerprint):
            return {"skipped": True}
        payload = _read_json(path)
        records = {
            str(key): value
            for key, value in (payload.items() if isinstance(payload, dict) else [])
            if isinstance(value, dict)
        }
        with self._db.transaction():
            self._tasks.save_tasks(records)
            self._validate(len(records), len(self._tasks.list_tasks()), "tasks")
            self._record("tasks", fingerprint, imported=len(records))
        return {"imported": len(records), "skipped": 0, "failed": 0}

    def _import_user_memory(self) -> dict[str, int] | None:
        path = self._legacy_file(LEGACY_USER_MEMORY_FILE)
        if path is None:
            return None
        fingerprint = _fingerprint([path])
        if not self._apply_if_changed("user_memory", fingerprint):
            return {"skipped": True}
        rows, failed = _read_jsonl(path)
        valid = [row for row in rows if str(row.get("id") or "")]
        distinct = {str(row["id"]) for row in valid}
        with self._db.transaction():
            for row in valid:
                self._user_memory.add_memory(row)
            self._validate(len(distinct), len(self._user_memory.list_memories()), "user memory")
            self._record("user_memory", fingerprint, imported=len(distinct), failed=failed)
        return {"imported": len(distinct), "skipped": 0, "failed": failed}

    def _import_index_state(self) -> dict[str, int] | None:
        path = self._legacy_file(LEGACY_INDEX_STATE_FILE)
        if path is None:
            return None
        fingerprint = _fingerprint([path])
        if not self._apply_if_changed("index_state", fingerprint):
            return {"skipped": True}
        payload = _read_json(path)
        doc_hashes = payload.get("doc_hashes") if isinstance(payload, dict) else {}
        doc_hashes = doc_hashes if isinstance(doc_hashes, dict) else {}
        with self._db.transaction():
            for doc_id, content_hash in doc_hashes.items():
                self._index_state.set_hash(str(doc_id), str(content_hash))
            self._validate(len(doc_hashes), len(self._index_state.get_hashes()), "index state")
            self._record("index_state", fingerprint, imported=len(doc_hashes))
        return {"imported": len(doc_hashes), "skipped": 0, "failed": 0}

    def _import_experiences(self) -> dict[str, int] | None:
        path = self._legacy_file(LEGACY_EXPERIENCE_STORE_FILE)
        if path is None:
            return None
        fingerprint = _fingerprint([path])
        if not self._apply_if_changed("experiences", fingerprint):
            return {"skipped": True}
        rows, failed = _read_jsonl(path)
        valid = [row for row in rows if str(row.get("experience_id") or "")]
        distinct = {str(row["experience_id"]) for row in valid}
        with self._db.transaction():
            for row in valid:
                self._experiences.add_experience(row)
            self._validate(len(distinct), self._experiences.count_experiences(), "experiences")
            self._record("experiences", fingerprint, imported=len(distinct), failed=failed)
        return {"imported": len(distinct), "skipped": 0, "failed": failed}

    def _import_feedback(self) -> dict[str, int] | None:
        path = self._legacy_file(LEGACY_FEEDBACK_STORE_FILE)
        if path is None:
            return None
        fingerprint = _fingerprint([path])
        if not self._apply_if_changed("feedback", fingerprint):
            return {"skipped": True}
        rows, failed = _read_jsonl(path)
        valid = [row for row in rows if str(row.get("record_id") or "")]
        with self._db.transaction():
            for row in valid:
                self._feedback.record_feedback(row)
            self._validate(len(valid), len(self._feedback.list_feedback()), "feedback")
            self._record("feedback", fingerprint, imported=len(valid), failed=failed)
        return {"imported": len(valid), "skipped": 0, "failed": failed}

    def _import_correction_memory(self) -> dict[str, int] | None:
        memory_dir = self._legacy_memory_dir
        if memory_dir is None:
            return None
        path = memory_dir / LEGACY_LESSONS_FILE
        if not path.exists():
            return None
        fingerprint = _fingerprint([path])
        if not self._apply_if_changed("correction_memory", fingerprint):
            return {"skipped": True}
        rows, failed = _read_jsonl(path)
        valid = [row for row in rows if row.get("type") and row.get("pattern")]
        distinct = {(str(row["type"]), str(row["pattern"])) for row in valid}
        with self._db.transaction():
            for row in valid:
                self._corrections.add_lesson(row)
            self._validate(len(distinct), len(self._corrections.list_lessons()), "correction memory")
            self._record("correction_memory", fingerprint, imported=len(distinct), failed=failed)
        return {"imported": len(distinct), "skipped": 0, "failed": failed}

    def _import_conversations(self) -> dict[str, int] | None:
        memory_dir = self._legacy_memory_dir
        if memory_dir is None:
            return None
        sessions_dir = memory_dir / "sessions"
        active_path = memory_dir / "active_session.json"
        session_folders = sorted(p for p in sessions_dir.iterdir() if p.is_dir()) if sessions_dir.exists() else []
        if not session_folders and not active_path.exists():
            return None
        interesting = [active_path] if active_path.exists() else []
        for folder in session_folders:
            interesting.extend(
                path for path in folder.iterdir() if path.is_file()
            )
        fingerprint = _fingerprint(interesting)
        if not self._apply_if_changed("conversations", fingerprint):
            return {"skipped": True}

        imported = 0
        failed = 0
        with self._db.transaction():
            for folder in session_folders:
                metadata = _read_json(folder / "metadata.json")
                if not isinstance(metadata, dict):
                    failed += 1
                    continue
                session_id = str(metadata.get("id") or folder.name)
                metadata["id"] = session_id
                summary_path = folder / "summary.txt"
                if summary_path.exists():
                    try:
                        metadata["summary"] = summary_path.read_text(encoding="utf-8")
                    except OSError:
                        logger.warning("Could not read legacy summary at %s", summary_path)
                self._conversations.save_conversation(session_id, metadata)
                messages = _read_json(folder / "messages.json")
                if isinstance(messages, list):
                    self._conversations.save_messages(
                        session_id, [m for m in messages if isinstance(m, dict)]
                    )
                imported += 1

            active = _read_json(active_path)
            if isinstance(active, dict) and active.get("session_id"):
                conversation_id = str(active["session_id"])
                if self._conversations.get_conversation(conversation_id) is not None:
                    self._conversations.set_active_conversation_id(conversation_id)

            self._validate(
                imported,
                len(self._conversations.list_conversation_ids()),
                "conversations",
            )
            self._record("conversations", fingerprint, imported=imported, failed=failed)
        return {"imported": imported, "skipped": 0, "failed": failed}

    # -- helpers ---------------------------------------------------------------

    def _legacy_file(self, name: str) -> Path | None:
        legacy_dir = self._legacy_db_dir
        if legacy_dir is None:
            return None
        candidate = legacy_dir / name
        return candidate if candidate.exists() else None

    def _validate(self, expected: int, actual: int, source: str, *, at_least: bool = False) -> None:
        """Fail the import (rolling it back) when records did not land.

        A legacy file may legitimately contain fewer *distinct* records than it
        has lines -- duplicate ids collapse, and rows without a usable id are
        skipped -- so the check is "as many rows as were actually accepted", not
        "as many lines as were read". A shortfall means rows were silently lost,
        which is the failure this guard exists to catch.
        """

        if actual < expected:
            raise MigrationValidationError(
                f"{source}: expected {expected} record(s) after import, found {actual}"
            )




