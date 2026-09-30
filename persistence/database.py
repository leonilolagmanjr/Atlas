"""Connection lifecycle for ``atlas.db``.

Atlas is a local-first desktop application, so the database is embedded and
needs no server. What it does need is a clear owner: one place that decides how
a connection is opened, how pragmas are applied, and when it is closed.

This module is that owner. Everything above it (repositories, stores, the Brain,
the API) receives an :class:`AtlasDatabase` or a repository and never touches
``sqlite3`` itself.

Concurrency: Atlas serves requests on a FastAPI thread pool and runs tasks on a
worker thread, so a connection cannot simply be shared. Connections are
therefore thread-local and created lazily, while the schema, pragmas and
transaction helpers live here. SQLite's WAL mode plus a busy timeout lets
readers and the single writer coexist without an application-level lock.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from persistence.schema import SCHEMA_SQL, SCHEMA_VERSION

logger = logging.getLogger(__name__)

#: Wait rather than fail when another thread holds the write lock. Long enough
#: to cover a normal transaction, short enough that a stuck writer is reported
#: instead of hanging the API.
BUSY_TIMEOUT_MS = 5000


class DatabaseError(RuntimeError):
    """Raised when the database cannot be opened or used."""


class SchemaVersionError(DatabaseError):
    """Raised when ``atlas.db`` was written by a newer Atlas."""


class AtlasDatabase:
    """Owns ``atlas.db``: creation, pragmas, schema, transactions, shutdown."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._lock = threading.RLock()
        self._closed = False

    # -- identity ---------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def schema_version(self) -> int:
        """The schema version currently stored in the database file."""

        with self._lock:
            return int(self._connect().execute("PRAGMA user_version").fetchone()[0])

    @property
    def exists(self) -> bool:
        return self._path.exists()

    # -- connections -----------------------------------------------------------

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=BUSY_TIMEOUT_MS / 1000,
            # Rows are addressed by name so repository code reads like the
            # domain fields it stores rather than positional tuples.
            detect_types=0,
            check_same_thread=False,
            isolation_level=None,  # explicit transactions only; see transaction()
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        # WAL lets the API read while a task writes. It is not appropriate for
        # every filesystem, so a failure degrades to the default journal rather
        # than failing startup.
        try:
            connection.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:  # pragma: no cover - filesystem dependent
            logger.warning("Could not enable WAL for %s; using the default journal", self._path)
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise DatabaseError("The Atlas database has been closed")
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._new_connection()
            self._local.connection = connection
            with self._lock:
                self._connections.append(connection)
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield this thread's connection (no transaction is implied)."""

        yield self._connect()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside one transaction, committing or rolling back.

        Nesting is safe: an inner block joins the outer transaction rather than
        opening a second one, so a repository never commits half of a caller's
        logical operation.
        """

        connection = self._connect()
        depth = getattr(self._local, "transaction_depth", 0)
        if depth:
            self._local.transaction_depth = depth + 1
            try:
                yield connection
            finally:
                self._local.transaction_depth = depth
            return
        connection.execute("BEGIN IMMEDIATE")
        self._local.transaction_depth = 1
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")
        finally:
            self._local.transaction_depth = 0

    # -- schema ----------------------------------------------------------------

    def initialize(self) -> int:
        """Create or migrate the schema. Returns the resulting version.

        Safe to call repeatedly: creating an existing database is a no-op, and a
        database already at this version is left alone.
        """

        with self._lock:
            connection = self._connect()
            existing = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if existing > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{self._path} uses schema version {existing}, but this Atlas "
                    f"understands up to version {SCHEMA_VERSION}"
                )
            connection.executescript(SCHEMA_SQL)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            if existing != SCHEMA_VERSION:
                logger.info(
                    "Atlas database ready at %s (schema version %s)", self._path, SCHEMA_VERSION
                )
            return SCHEMA_VERSION

    def foreign_keys_enabled(self) -> bool:
        with self._lock:
            return bool(self._connect().execute("PRAGMA foreign_keys").fetchone()[0])

    def table_names(self) -> set[str]:
        with self._lock:
            rows = self._connect().execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        return {str(row[0]) for row in rows}

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Release every connection this database opened."""

        with self._lock:
            for connection in self._connections:
                try:
                    connection.close()
                except sqlite3.Error:  # pragma: no cover - defensive
                    logger.debug("Error closing an Atlas connection", exc_info=True)
            self._connections.clear()
            self._closed = True
            self._local = threading.local()

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "AtlasDatabase":
        self.initialize()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
