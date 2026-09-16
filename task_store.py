"""Durable storage for API task records.

Task execution remains owned by Brain and Executor. This module only persists
API-facing task snapshots so history survives an API restart.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class TaskStore:
    """Persist task records as one JSON document under the configured root."""

    def __init__(self, *, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(task_id): value
            for task_id, value in payload.items()
            if isinstance(value, dict)
        }

    def save(self, records: dict[str, Any]) -> None:
        payload = {
            str(task_id): _to_json_safe(record)
            for task_id, record in records.items()
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{self._path.stem}-",
            suffix=".tmp",
            dir=self._path.parent,
            text=True,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
                temporary_file.write(serialized)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_name, self._path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


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
