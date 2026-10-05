"""Startup/shutdown wiring for Atlas's persistence layer.

This is the one place that resolves the data root, opens ``atlas.db``, applies
the schema, runs the one-time legacy import, and hands repositories to the rest
of Atlas. It is deliberately small: a :class:`PersistenceContext` is just a
bundle of paths, the database, and the repositories built on top of it.

Atlas requests a context through :func:`default_persistence`, which builds one
lazily on first use. Tests and tools that want an isolated database can call
:func:`open_persistence` with their own
:class:`~persistence.paths.AtlasDataPaths` (or construct a repository directly).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from persistence.database import AtlasDatabase
from persistence.migration import LegacyMigrator, MigrationReport
from persistence.paths import AtlasDataPaths, resolve_data_paths
from persistence.repositories import (
    SqliteConversationRepository,
    SqliteCorrectionMemoryRepository,
    SqliteExperienceRepository,
    SqliteFeedbackRepository,
    SqliteIndexStateRepository,
    SqliteTaskRepository,
    SqliteUserMemoryRepository,
)

logger = logging.getLogger(__name__)


@dataclass
class PersistenceContext:
    """The database, its paths, and the repositories built on top of it."""

    paths: AtlasDataPaths
    database: AtlasDatabase
    conversations: SqliteConversationRepository
    tasks: SqliteTaskRepository
    user_memory: SqliteUserMemoryRepository
    experiences: SqliteExperienceRepository
    feedback: SqliteFeedbackRepository
    index_state: SqliteIndexStateRepository
    correction_memory: SqliteCorrectionMemoryRepository
    migration: MigrationReport | None = field(default=None)

    @property
    def database_file(self) -> Path:
        return self.database.path

    def close(self) -> None:
        """Release the SQLite connections this context owns."""

        self.database.close()


def open_persistence(
    paths: AtlasDataPaths | None = None,
    *,
    migrate: bool = True,
    ensure: bool = True,
) -> PersistenceContext:
    """Open (creating if needed) the authoritative database and its repositories.

    Safe to call repeatedly; ``initialize`` is idempotent and the legacy import
    only runs once per source.
    """

    resolved = paths if paths is not None else resolve_data_paths()
    if ensure:
        resolved.ensure()
    database = AtlasDatabase(resolved.database_file)
    database.initialize()
    report: MigrationReport | None = None
    if migrate:
        report = LegacyMigrator(database, resolved).run()
        if report.imported_total:
            logger.info(
                "Legacy migration imported %d record(s): %s",
                report.imported_total,
                ", ".join(sorted(report.applied)),
            )
    return PersistenceContext(
        paths=resolved,
        database=database,
        conversations=SqliteConversationRepository(database),
        tasks=SqliteTaskRepository(database),
        user_memory=SqliteUserMemoryRepository(database),
        experiences=SqliteExperienceRepository(database),
        feedback=SqliteFeedbackRepository(database),
        index_state=SqliteIndexStateRepository(database),
        correction_memory=SqliteCorrectionMemoryRepository(database),
        migration=report,
    )


_default: PersistenceContext | None = None
_lock = threading.RLock()


def default_persistence() -> PersistenceContext:
    """Return the process-wide persistence context, building it on first use."""

    global _default
    if _default is None:
        with _lock:
            if _default is None:
                _default = open_persistence()
    return _default


def database_for_legacy_path(path: Path | str) -> AtlasDatabase:
    """Open a small, self-contained database next to a legacy file/directory.

    This keeps test stores (and any tool that constructs a store with an explicit
    path) fully isolated: a ``tasks.json`` becomes ``tasks.db`` in the same
    directory rather than touching the process-wide ``atlas.db``.

    The database is *ephemeral*: it holds no connection between operations, so
    nothing keeps the surrounding directory locked on Windows.
    """

    candidate = Path(path)
    db_path = candidate / "atlas.db" if candidate.suffix == "" else candidate.with_suffix(".db")
    database = AtlasDatabase(db_path, ephemeral=True)
    database.initialize()
    return database



def close_default_persistence() -> None:
    """Close and forget the process-wide context (used at shutdown)."""

    global _default
    with _lock:
        if _default is not None:
            _default.close()
            _default = None


def set_default_persistence(context: PersistenceContext | None) -> None:
    """Install a specific context (tests/embedded use); closes the previous one."""

    global _default
    with _lock:
        if _default is not None and _default is not context:
            _default.close()
        _default = context
