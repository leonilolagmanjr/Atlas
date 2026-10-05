"""Atlas persistence layer.

This package owns *all* structured persistent state and the paths that state
lives in. Everything above it (the Brain, the Conversation Runtime, the
experience loop, the planner, the API) talks to repository contracts and a small
:class:`~persistence.context.PersistenceContext`; nothing else in Atlas opens a
SQLite connection, writes a SQL string, or decides where ``atlas.db`` lives.

Layout::

    persistence.paths        -- application vs. user-data roots
    persistence.schema       -- the SQLite schema + version
    persistence.database     -- connection/schema/transaction lifecycle
    persistence.interfaces    -- repository contracts (the core-facing API)
    persistence.repositories -- the SQLite implementations of those contracts
    persistence.migration    -- one-time import of legacy JSON/JSONL state
    persistence.context      -- startup/shutdown wiring that ties it together
"""

from __future__ import annotations

from persistence.context import (
    PersistenceContext,
    close_default_persistence,
    default_persistence,
    open_persistence,
)
from persistence.database import AtlasDatabase, DatabaseError, SchemaVersionError
from persistence.paths import (
    AtlasApplicationPaths,
    AtlasDataPaths,
    application_root,
    is_frozen,
    local_app_data,
    resolve_application_paths,
    resolve_data_paths,
)

__all__ = [
    "AtlasApplicationPaths",
    "AtlasDataPaths",
    "AtlasDatabase",
    "DatabaseError",
    "PersistenceContext",
    "SchemaVersionError",
    "application_root",
    "close_default_persistence",
    "default_persistence",
    "is_frozen",
    "local_app_data",
    "open_persistence",
    "resolve_application_paths",
    "resolve_data_paths",
]
