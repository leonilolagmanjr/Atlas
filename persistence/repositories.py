"""SQLite implementations of Atlas's persistence contracts.

Each class here is the *only* place a SQL statement for its domain exists. The
rest of Atlas hands these repositories plain dictionaries (the same shapes the
models already serialize to) and never sees a connection, a cursor or a PRAGMA.

Design notes:

* Dictionaries in, dictionaries out. Repositories deal in the models' existing
  ``to_dict``/``from_dict`` shapes, so a store can delegate without translating
  field-by-field semantics.
* Variable, nested values (attachments, tool calls, plans, evidence) are stored
  as JSON *in a single column* — a representation of a field, never a database
  in disguise.
* Writes that must be atomic (replacing a transcript, importing a legacy store)
  run inside ``AtlasDatabase.transaction`` so a crash cannot leave half a record
  set behind.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from persistence.database import AtlasDatabase

logger = logging.getLogger(__name__)

#: Key under which the active-conversation pointer lives in ``app_state``.
ACTIVE_CONVERSATION_KEY = "active_conversation_id"


def utc_now() -> str:
    """An ISO-8601 UTC timestamp, matching the models' own timestamps."""

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# value helpers
# ---------------------------------------------------------------------------


def _dump(value: Any, default: Any) -> str:
    """Serialize a JSON column, falling back to ``default`` for empty input."""

    if value is None:
        value = default
    return json.dumps(value, ensure_ascii=False)


def _load(value: Any, default: Any) -> Any:
    """Parse a JSON column, degrading to ``default`` for missing/corrupt data."""

    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default
    return parsed


def _load_list(value: Any) -> list[Any]:
    parsed = _load(value, [])
    return parsed if isinstance(parsed, list) else []


def _load_dict(value: Any) -> dict[str, Any]:
    parsed = _load(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _bool_int(value: Any) -> int:
    return 1 if value else 0


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------


class SqliteConversationRepository:
    """Conversations, their transcript, and the active-conversation pointer."""

    def __init__(self, database: AtlasDatabase) -> None:
        self._db = database

    # -- conversations ---------------------------------------------------------

    @staticmethod
    def _conversation(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "last_modified": row["last_modified"],
            "message_count": int(row["message_count"]),
            "summary": row["summary"],
            "archived": bool(row["archived"]),
            "model": row["model"],
            "metadata": _load_dict(row["metadata"]),
        }

    def list_conversation_ids(self) -> list[str]:
        with self._db.connection() as connection:
            rows = connection.execute(
                "SELECT id FROM conversations ORDER BY last_modified DESC, id ASC"
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._db.connection() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return self._conversation(row) if row is not None else None

    def create_conversation(self, conversation_id: str, title: str) -> dict[str, Any]:
        stamp = utc_now()
        with self._db.transaction() as connection:
            connection.execute(
                "INSERT INTO conversations (id, title, created_at, last_modified) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, title or "Untitled", stamp, stamp),
            )
        stored = self.get_conversation(conversation_id)
        assert stored is not None
        return stored

    def save_conversation(self, conversation_id: str, metadata: dict[str, Any]) -> None:
        """Insert or replace a conversation from a serialized metadata dict."""

        with self._db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO conversations
                    (id, title, created_at, last_modified, message_count, summary,
                     archived, model, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    created_at = excluded.created_at,
                    last_modified = excluded.last_modified,
                    message_count = excluded.message_count,
                    summary = excluded.summary,
                    archived = excluded.archived,
                    model = excluded.model,
                    metadata = excluded.metadata
                """,
                (
                    conversation_id,
                    str(metadata.get("title") or "Untitled"),
                    str(metadata.get("created_at") or ""),
                    str(metadata.get("last_modified") or ""),
                    int(metadata.get("message_count") or 0),
                    metadata.get("summary"),
                    _bool_int(metadata.get("archived")),
                    str(metadata.get("model") or ""),
                    _dump(metadata.get("metadata"), {}),
                ),
            )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._db.transaction() as connection:
            connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    # -- messages --------------------------------------------------------------

    @staticmethod
    def _message(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["created_at"],
            "turn_number": int(row["turn_number"]),
            "attachments": _load_list(row["attachments"]),
            "tool_calls": _load_list(row["tool_calls"]),
            "tool_results": _load_list(row["tool_results"]),
            "citations": _load_list(row["citations"]),
            "execution_state": row["execution_state"],
            "metadata": _load_dict(row["metadata"]),
        }

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._db.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY ordinal ASC",
                (conversation_id,),
            ).fetchall()
        return [self._message(row) for row in rows]

    def save_messages(self, conversation_id: str, messages: list[dict[str, Any]]) -> None:
        """Replace a conversation's transcript atomically, preserving ids/order."""

        with self._db.transaction() as connection:
            connection.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            for ordinal, message in enumerate(messages):
                connection.execute(
                    """
                    INSERT INTO messages
                        (id, conversation_id, ordinal, role, content, created_at,
                         turn_number, attachments, tool_calls, tool_results,
                         citations, execution_state, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(message.get("id") or ""),
                        conversation_id,
                        ordinal,
                        str(message.get("role") or "user"),
                        str(message.get("content") or ""),
                        str(message.get("timestamp") or ""),
                        int(message.get("turn_number") or 0),
                        _dump(message.get("attachments"), []),
                        _dump(message.get("tool_calls"), []),
                        _dump(message.get("tool_results"), []),
                        _dump(message.get("citations"), []),
                        str(message.get("execution_state") or "completed"),
                        _dump(message.get("metadata"), {}),
                    ),
                )
            connection.execute(
                "UPDATE conversations SET message_count = ?, last_modified = ? WHERE id = ?",
                (len(messages), utc_now(), conversation_id),
            )

    _MESSAGE_COLUMNS: dict[str, tuple[str, Any]] = {
        "role": ("role", str),
        "content": ("content", str),
        "attachments": ("attachments", lambda value: _dump(value, [])),
        "tool_calls": ("tool_calls", lambda value: _dump(value, [])),
        "tool_results": ("tool_results", lambda value: _dump(value, [])),
        "citations": ("citations", lambda value: _dump(value, [])),
        "execution_state": ("execution_state", str),
        "metadata": ("metadata", lambda value: _dump(value, {})),
    }

    def update_message(
        self, conversation_id: str, message_id: str, changes: dict[str, Any]
    ) -> dict[str, Any] | None:
        assignments: list[str] = []
        parameters: list[Any] = []
        for key, value in changes.items():
            column = self._MESSAGE_COLUMNS.get(key)
            if column is None or value is None:
                continue
            name, coerce = column
            assignments.append(f"{name} = ?")
            parameters.append(coerce(value))
        if assignments:
            parameters.extend([message_id, conversation_id])
            with self._db.transaction() as connection:
                connection.execute(
                    f"UPDATE messages SET {', '.join(assignments)} "
                    "WHERE id = ? AND conversation_id = ?",
                    parameters,
                )
        with self._db.connection() as connection:
            row = connection.execute(
                "SELECT * FROM messages WHERE id = ? AND conversation_id = ?",
                (message_id, conversation_id),
            ).fetchone()
        return self._message(row) if row is not None else None

    # -- active-conversation pointer ------------------------------------------

    def get_active_conversation_id(self) -> str | None:
        with self._db.connection() as connection:
            row = connection.execute(
                "SELECT value FROM app_state WHERE key = ?", (ACTIVE_CONVERSATION_KEY,)
            ).fetchone()
        return str(row["value"]) if row is not None and row["value"] else None

    def set_active_conversation_id(self, conversation_id: str) -> None:
        with self._db.transaction() as connection:
            connection.execute(
                "INSERT INTO app_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (ACTIVE_CONVERSATION_KEY, conversation_id),
            )

    def clear_active_conversation_id(self) -> None:
        with self._db.transaction() as connection:
            connection.execute("DELETE FROM app_state WHERE key = ?", (ACTIVE_CONVERSATION_KEY,))


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


class SqliteTaskRepository:
    """Durable API task snapshots (execution stays in the Brain/Executor)."""

    #: Columns stored as JSON text; everything else is a scalar column.
    _JSON_COLUMNS = frozenset(
        {
            "plan", "tool_calls", "errors", "warnings", "web_sources",
            "web_results", "reasoning", "provenance", "citations", "evidence",
        }
    )
    #: JSON columns that default to ``None`` (not ``[]``/``{}``) when empty.
    _NULLABLE_JSON = frozenset({"plan", "evidence"})
    _COLUMNS = (
        "id", "request", "status", "response", "created_at", "updated_at",
        "task_id", "plan", "tool_calls", "errors", "warnings", "web_sources",
        "web_results", "reasoning", "provenance", "citations", "response_mode",
        "evidence", "feedback_available", "feedback_outcome", "feedback_category",
        "feedback_category_label", "feedback_reason", "feedback_correction",
        "experience_id",
    )

    def __init__(self, database: AtlasDatabase) -> None:
        self._db = database

    @classmethod
    def _record(cls, row: Any) -> dict[str, Any]:
        record: dict[str, Any] = {}
        for column in cls._COLUMNS:
            value = row[column]
            if column in cls._JSON_COLUMNS:
                default = None if column in cls._NULLABLE_JSON else (
                    [] if column in {"errors", "warnings", "tool_calls", "web_sources",
                                     "web_results", "reasoning", "provenance", "citations"} else {}
                )
                record[column] = _load(value, default)
            elif column == "feedback_available":
                record[column] = bool(value)
            else:
                record[column] = value
        return record

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._db.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY updated_at DESC, id ASC"
            ).fetchall()
        return [self._record(row) for row in rows]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._db.connection() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._record(row) if row is not None else None

    @classmethod
    def _cell(cls, column: str, record: dict[str, Any]) -> Any:
        value = record.get(column)
        if column in cls._JSON_COLUMNS:
            if value is None:
                return None if column in cls._NULLABLE_JSON else _dump(
                    [] if column != "evidence" else {}, None
                )
            return _dump(value, None)
        if column == "feedback_available":
            return _bool_int(value)
        if column in {"created_at", "updated_at"}:
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return value

    def save_tasks(self, records: dict[str, dict[str, Any]]) -> None:
        """Persist the given tasks and drop any task no longer present."""

        columns = ", ".join(self._COLUMNS)
        placeholders = ", ".join("?" for _ in self._COLUMNS)
        with self._db.transaction() as connection:
            ids = list(records)
            if ids:
                marks = ", ".join("?" for _ in ids)
                connection.execute(f"DELETE FROM tasks WHERE id NOT IN ({marks})", ids)
            else:
                connection.execute("DELETE FROM tasks")
            for task_id, record in records.items():
                normalized = {**record, "id": str(record.get("id") or task_id)}
                values = [self._cell(column, normalized) for column in self._COLUMNS]
                connection.execute(
                    f"INSERT OR REPLACE INTO tasks ({columns}) VALUES ({placeholders})",
                    values,
                )

    def clear(self) -> None:
        with self._db.transaction() as connection:
            connection.execute("DELETE FROM tasks")


# ---------------------------------------------------------------------------
# user memory
# ---------------------------------------------------------------------------


class SqliteUserMemoryRepository:
    """Authoritative long-term user memory (Chroma only indexes it)."""

    _FIELDS = (
        "id", "kind", "text", "summary", "source", "conversation_id",
        "message_id", "confidence", "created_at", "updated_at",
    )

    def __init__(self, database: AtlasDatabase) -> None:
        self._db = database

    @staticmethod
    def _record(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "text": row["text"],
            "summary": row["summary"],
            "source": row["source"],
            "conversation_id": row["conversation_id"],
            "message_id": row["message_id"],
            "confidence": float(row["confidence"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": _load_dict(row["metadata"]),
        }

    def list_memories(self) -> list[dict[str, Any]]:
        with self._db.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM user_memories ORDER BY ordinal ASC"
            ).fetchall()
        return [self._record(row) for row in rows]

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self._db.connection() as connection:
            row = connection.execute(
                "SELECT * FROM user_memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._record(row) if row is not None else None

    def add_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        with self._db.transaction() as connection:
            next_ordinal = int(
                connection.execute(
                    "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM user_memories"
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO user_memories
                    (id, kind, text, summary, source, conversation_id, message_id,
                     confidence, created_at, updated_at, ordinal, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(memory.get("id") or ""),
                    str(memory.get("kind") or "preference"),
                    str(memory.get("text") or ""),
                    str(memory.get("summary") or ""),
                    str(memory.get("source") or "user_message"),
                    str(memory.get("conversation_id") or ""),
                    str(memory.get("message_id") or ""),
                    float(memory.get("confidence") or 0.6),
                    str(memory.get("created_at") or utc_now()),
                    str(memory.get("updated_at") or utc_now()),
                    next_ordinal,
                    _dump(memory.get("metadata"), {}),
                ),
            )
        return memory

    def update_memory(self, memory_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        assignments: list[str] = []
        parameters: list[Any] = []
        for key, value in changes.items():
            if key not in self._FIELDS or value is None:
                continue
            assignments.append(f"{key} = ?")
            parameters.append(float(value) if key == "confidence" else str(value))
        if assignments:
            assignments.append("updated_at = ?")
            parameters.append(utc_now())
            parameters.append(memory_id)
            with self._db.transaction() as connection:
                connection.execute(
                    f"UPDATE user_memories SET {', '.join(assignments)} WHERE id = ?",
                    parameters,
                )
        return self.get_memory(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        with self._db.transaction() as connection:
            cursor = connection.execute("DELETE FROM user_memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def clear_memories(self) -> int:
        with self._db.transaction() as connection:
            cursor = connection.execute("DELETE FROM user_memories")
        return cursor.rowcount or 0

    def trim_to(self, max_records: int) -> int:
        """Drop the oldest memories past ``max_records``; returns rows removed."""

        keep = max(1, int(max_records))
        with self._db.transaction() as connection:
            cursor = connection.execute(
                """
                DELETE FROM user_memories
                WHERE ordinal < (
                    SELECT ordinal FROM user_memories ORDER BY ordinal DESC
                    LIMIT 1 OFFSET ?
                )
                """,
                (keep - 1,),
            )
        return cursor.rowcount or 0


# ---------------------------------------------------------------------------
# experiences
# ---------------------------------------------------------------------------


class SqliteExperienceRepository:
    """Authoritative experience records (Chroma only indexes them)."""

    #: dict key -> (column, kind) with kind in {"text", "json", "int"}.
    #: The model serializes its timestamp as ``timestamp``; the column is
    #: ``recorded_at``, so the mapping is explicit rather than implicit.
    _MAPPING: dict[str, tuple[str, str]] = {
        "experience_id": ("experience_id", "text"),
        "timestamp": ("recorded_at", "text"),
        "experience_type": ("experience_type", "text"),
        "record_id": ("record_id", "text"),
        "task_id": ("task_id", "text"),
        "conversation_id": ("conversation_id", "text"),
        "model_used": ("model_used", "text"),
        "original_user_request": ("original_user_request", "text"),
        "interpreted_intent": ("interpreted_intent", "text"),
        "desired_outcome": ("desired_outcome", "text"),
        "task_type": ("task_type", "text"),
        "goal": ("goal", "text"),
        "request_type": ("request_type", "text"),
        "extracted_entities": ("extracted_entities", "json"),
        "requested_destination": ("requested_destination", "text"),
        "requested_format": ("requested_format", "text"),
        "constraints": ("constraints", "json"),
        "generated_plan": ("generated_plan", "json"),
        "tools_used": ("tools_used", "json"),
        "execution_steps": ("execution_steps", "json"),
        "verification_result": ("verification_result", "json"),
        "final_result": ("final_result", "text"),
        "completion_checks": ("completion_checks", "json"),
        "outcome": ("outcome", "text"),
        "feedback": ("feedback", "text"),
        "feedback_reason": ("feedback_reason", "text"),
        "failure_category": ("failure_category", "text"),
        "user_correction": ("user_correction", "text"),
        "failed_step": ("failed_step", "text"),
        "expected_behavior": ("expected_behavior", "text"),
        "actual_behavior": ("actual_behavior", "text"),
        "response_quality": ("response_quality", "text"),
        "state": ("state", "text"),
        "retrieved_experience_ids": ("retrieved_experience_ids", "json"),
        "reasoning_strategy_ids": ("reasoning_strategy_ids", "json"),
        "attempt": ("attempt", "int"),
        "recovers_experience_id": ("recovers_experience_id", "text"),
        "confirmations": ("confirmations", "int"),
        "pattern_key": ("pattern_key", "text"),
    }

    #: JSON columns that hold an object rather than a list.
    _DICT_JSON = frozenset(
        {"extracted_entities", "verification_result", "completion_checks"}
    )

    def __init__(self, database: AtlasDatabase) -> None:
        self._db = database

    @classmethod
    def _record(cls, row: Any) -> dict[str, Any]:
        record: dict[str, Any] = {}
        for key, (column, kind) in cls._MAPPING.items():
            value = row[column]
            if kind == "json":
                record[key] = _load(value, {} if column in cls._DICT_JSON else [])
            elif kind == "int":
                record[key] = int(value)
            else:
                record[key] = value
        return record

    def list_experiences(self) -> list[dict[str, Any]]:
        with self._db.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM experiences ORDER BY ordinal ASC"
            ).fetchall()
        return [self._record(row) for row in rows]

    def get_experience(self, experience_id: str) -> dict[str, Any] | None:
        with self._db.connection() as connection:
            row = connection.execute(
                "SELECT * FROM experiences WHERE experience_id = ?", (experience_id,)
            ).fetchone()
        return self._record(row) if row is not None else None

    def count_experiences(self) -> int:
        with self._db.connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM experiences").fetchone()[0])

    @classmethod
    def _columns_and_values(cls, experience: dict[str, Any]) -> tuple[list[str], list[Any]]:
        columns: list[str] = []
        values: list[Any] = []
        for key, (column, kind) in cls._MAPPING.items():
            value = experience.get(key)
            if kind == "json":
                default = {} if column in cls._DICT_JSON else []
                value = _dump(value if value is not None else default, default)
            elif kind == "int":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = 0
            else:
                value = str(value if value is not None else "")
            columns.append(column)
            values.append(value)
        return columns, values

    def add_experience(self, experience: dict[str, Any]) -> bool:
        experience_id = str(experience.get("experience_id") or "")
        if not experience_id:
            return False
        columns, values = self._columns_and_values(experience)
        marks = ", ".join("?" for _ in columns)
        with self._db.transaction() as connection:
            next_ordinal = int(
                connection.execute(
                    "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM experiences"
                ).fetchone()[0]
            )
            connection.execute(
                f"INSERT OR REPLACE INTO experiences (ordinal, {', '.join(columns)}) "
                f"VALUES (?, {marks})",
                [next_ordinal, *values],
            )
        return True

    def replace_experience(self, experience: dict[str, Any]) -> bool:
        experience_id = str(experience.get("experience_id") or "")
        if not experience_id:
            return False
        columns, values = self._columns_and_values(experience)
        assignments = ", ".join(f"{column} = ?" for column in columns)
        with self._db.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE experiences SET {assignments} WHERE experience_id = ?",
                [*values, experience_id],
            )
        return cursor.rowcount > 0

    def delete_experiences_below_ordinal(self, keep_from: int) -> int:
        with self._db.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM experiences WHERE ordinal < ?", (int(keep_from),)
            )
        return cursor.rowcount or 0

    def trim_to(self, max_records: int) -> int:
        """Keep the newest ``max_records`` experiences; returns rows removed."""

        keep = max(1, int(max_records))
        with self._db.transaction() as connection:
            cursor = connection.execute(
                """
                DELETE FROM experiences
                WHERE ordinal < (
                    SELECT ordinal FROM experiences ORDER BY ordinal DESC
                    LIMIT 1 OFFSET ?
                )
                """,
                (keep - 1,),
            )
        return cursor.rowcount or 0

    def clear_experiences(self) -> None:
        with self._db.transaction() as connection:
            connection.execute("DELETE FROM experiences")


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------


class SqliteFeedbackRepository:
    """Durable feedback clicks, recorded before any experience is promoted."""

    #: dict key -> column. The model serializes its time as ``timestamp``.
    _MAPPING = {
        "record_id": "record_id",
        "outcome": "outcome",
        "reason": "reason",
        "failure_category": "failure_category",
        "correction": "correction",
        "expected_behavior": "expected_behavior",
        "response_quality": "response_quality",
        "timestamp": "recorded_at",
    }

    def __init__(self, database: AtlasDatabase) -> None:
        self._db = database

    @classmethod
    def _record(cls, row: Any) -> dict[str, Any]:
        return {key: row[column] for key, column in cls._MAPPING.items()}

    def list_feedback(self) -> list[dict[str, Any]]:
        with self._db.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM feedback_events ORDER BY ordinal ASC"
            ).fetchall()
        return [self._record(row) for row in rows]

    def record_feedback(self, event: dict[str, Any]) -> bool:
        columns = list(self._MAPPING.values())
        marks = ", ".join("?" for _ in columns)
        values = [str(event.get(key) or "") for key in self._MAPPING]
        with self._db.transaction() as connection:
            connection.execute(
                f"INSERT INTO feedback_events ({', '.join(columns)}) VALUES ({marks})",
                values,
            )
        return True

    def latest_feedback(self, record_id: str) -> dict[str, Any] | None:
        with self._db.connection() as connection:
            row = connection.execute(
                "SELECT * FROM feedback_events WHERE record_id = ? "
                "ORDER BY ordinal DESC LIMIT 1",
                (record_id,),
            ).fetchone()
        return self._record(row) if row is not None else None

    def clear_feedback(self) -> None:
        with self._db.transaction() as connection:
            connection.execute("DELETE FROM feedback_events")


# ---------------------------------------------------------------------------
# knowledge index state
# ---------------------------------------------------------------------------


class SqliteIndexStateRepository:
    """Which documents are indexed. Documents stay on disk; Chroma holds vectors."""

    def __init__(self, database: AtlasDatabase) -> None:
        self._db = database

    def get_hashes(self) -> dict[str, str]:
        with self._db.connection() as connection:
            rows = connection.execute(
                "SELECT doc_id, content_hash FROM index_documents"
            ).fetchall()
        return {str(row["doc_id"]): str(row["content_hash"]) for row in rows}

    def set_hash(self, doc_id: str, content_hash: str, *, path: str = "") -> None:
        with self._db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO index_documents (doc_id, path, content_hash, last_indexed)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    path = excluded.path,
                    content_hash = excluded.content_hash,
                    last_indexed = excluded.last_indexed
                """,
                (doc_id, path, content_hash, utc_now()),
            )

    def remove(self, doc_id: str) -> None:
        with self._db.transaction() as connection:
            connection.execute("DELETE FROM index_documents WHERE doc_id = ?", (doc_id,))

    def describe(self) -> list[dict[str, Any]]:
        with self._db.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM index_documents ORDER BY doc_id ASC"
            ).fetchall()
        return [
            {
                "doc_id": row["doc_id"],
                "path": row["path"],
                "content_hash": row["content_hash"],
                "chunk_count": int(row["chunk_count"]),
                "status": row["status"],
                "last_indexed": row["last_indexed"],
                "metadata": _load_dict(row["metadata"]),
            }
            for row in rows
        ]


# ---------------------------------------------------------------------------
# correction memory (structured interpretation lessons)
# ---------------------------------------------------------------------------


class SqliteCorrectionMemoryRepository:
    """Durable, curated interpretation lessons (never a vector dump)."""

    def __init__(self, database: AtlasDatabase) -> None:
        self._db = database

    @staticmethod
    def _record(row: Any) -> dict[str, Any]:
        return {
            "type": row["lesson_type"],
            "pattern": row["pattern"],
            "correct_behavior": _load_list(row["correct_behavior"]),
            "fields": _load_dict(row["fields"]),
            "confidence": float(row["confidence"]),
            "hits": int(row["hits"]),
        }

    def list_lessons(self) -> list[dict[str, Any]]:
        with self._db.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM correction_lessons ORDER BY id ASC"
            ).fetchall()
        return [self._record(row) for row in rows]

    def add_lesson(self, lesson: dict[str, Any]) -> bool:
        lesson_type = str(lesson.get("type") or "")
        pattern = str(lesson.get("pattern") or "")
        if not lesson_type or not pattern:
            return False
        with self._db.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO correction_lessons
                    (lesson_type, pattern, correct_behavior, fields, confidence,
                     hits, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lesson_type,
                    pattern,
                    _dump(lesson.get("correct_behavior"), []),
                    _dump(lesson.get("fields"), {}),
                    float(lesson.get("confidence") or 0.0),
                    int(lesson.get("hits") or 0),
                    utc_now(),
                ),
            )
        return cursor.rowcount > 0

    def increment_hits(self, lesson_type: str, pattern: str) -> None:
        with self._db.transaction() as connection:
            connection.execute(
                "UPDATE correction_lessons SET hits = hits + 1 "
                "WHERE lesson_type = ? AND pattern = ?",
                (lesson_type, pattern),
            )


