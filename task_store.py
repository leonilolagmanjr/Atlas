"""Durable storage for API task records.

Task execution remains owned by Brain and Executor. This module only persists
API-facing task snapshots so history survives an API restart.

The authoritative store is now the ``tasks`` table in ``atlas.db``. This class is
a thin, model-aware façade over
:class:`persistence.repositories.SqliteTaskRepository` — it holds no SQL and no
file format of its own.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from persistence.context import database_for_legacy_path, default_persistence
from persistence.database import AtlasDatabase
from persistence.repositories import SqliteTaskRepository

logger = logging.getLogger(__name__)


class TaskStore:
    """Persist task records in ``atlas.db`` (or an injected repository/database)."""

    def __init__(
        self,
        *,
        path: Path | str | None = None,
        repository: SqliteTaskRepository | None = None,
        database: AtlasDatabase | None = None,
    ) -> None:
        self._owned_database = database
        self._owns_database = False
        if repository is not None:
            self._repository = repository
        elif database is not None:
            self._repository = SqliteTaskRepository(database)
        elif path is not None:
            self._owned_database = database_for_legacy_path(path)
            self._owns_database = True
            self._repository = SqliteTaskRepository(self._owned_database)
        else:
            context = default_persistence()
            self._owned_database = context.database
            self._repository = context.tasks
        if self._owned_database is not None:
            self._path = self._owned_database.path
        else:
            self._path = Path(path) if path is not None else default_persistence().database_file

    def load(self) -> dict[str, dict[str, Any]]:
        return {str(record.get("id")): record for record in self._repository.list_tasks()}

    def save(self, records: dict[str, Any]) -> None:
        payload = {
            str(task_id): _to_json_safe(record)
            for task_id, record in records.items()
        }
        self._repository.save_tasks(payload)

    def close(self) -> None:
        """Release this store's SQLite connection when it owns the database.

        A store built on the process-wide context, or handed an explicit
        repository/database, must not close it out from under its owner; only a
        store that created its own database (an explicit ``path``) releases it.
        """

        if self._owns_database and self._owned_database is not None:
            self._owned_database.close()

    def __enter__(self) -> "TaskStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - GC timing is not deterministic
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            pass


def _to_json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return value
