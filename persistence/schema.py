"""SQLite schema for Atlas's authoritative structured state.

One database, one normalized schema, and an explicit version. The migration
history lives in ``PRAGMA user_version`` so a future schema change is a
numbered step rather than an implicit "if the column exists" check scattered
across modules.

Design rules:

* Structured state is relational. JSON is only ever a *representation of a
  single field* (attachments, tool calls, tool results, plan snapshots), never
  the database.
* Documents, PDFs and artifacts stay on the filesystem; the tables below hold
  only metadata that points at them.
* Vector data never enters SQLite. Chroma owns embeddings; these tables own
  the records the embeddings describe.
"""

from __future__ import annotations

#: Bumped whenever the schema below changes. ``atlas.db`` records the version
#: it was created/migrated to, and refuses to open against a newer one.
SCHEMA_VERSION = 1

#: The full schema. Written as one script so a fresh database and a migrated
#: one converge on exactly the same shape.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT 'Untitled',
    created_at      TEXT NOT NULL,
    last_modified   TEXT NOT NULL,
    message_count   INTEGER NOT NULL DEFAULT 0,
    summary         TEXT,
    archived        INTEGER NOT NULL DEFAULT 0,
    model           TEXT NOT NULL DEFAULT '',
    metadata        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS ix_conversations_last_modified
    ON conversations(last_modified DESC);

-- The active-conversation pointer. A single row keyed by a constant id keeps it
-- a relational fact rather than a one-field JSON file.
CREATE TABLE IF NOT EXISTS app_state (
    key     TEXT PRIMARY KEY,
    value   TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    role            TEXT NOT NULL DEFAULT 'user',
    content         TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    turn_number     INTEGER NOT NULL DEFAULT 0,
    attachments     TEXT NOT NULL DEFAULT '[]',
    tool_calls      TEXT NOT NULL DEFAULT '[]',
    tool_results    TEXT NOT NULL DEFAULT '[]',
    citations        TEXT NOT NULL DEFAULT '[]',
    execution_state TEXT NOT NULL DEFAULT 'completed',
    metadata        TEXT NOT NULL DEFAULT '{}'
);

-- Transcript reconstruction is always "this conversation, in order".
CREATE INDEX IF NOT EXISTS ix_messages_conversation_ordinal
    ON messages(conversation_id, ordinal);

CREATE TABLE IF NOT EXISTS tasks (
    id                       TEXT PRIMARY KEY,
    request                  TEXT NOT NULL DEFAULT '',
    status                   TEXT NOT NULL DEFAULT '',
    response                 TEXT,
    created_at               REAL NOT NULL DEFAULT 0,
    updated_at               REAL NOT NULL DEFAULT 0,
    task_id                  TEXT,
    plan                     TEXT,
    tool_calls               TEXT NOT NULL DEFAULT '[]',
    errors                   TEXT NOT NULL DEFAULT '[]',
    warnings                 TEXT NOT NULL DEFAULT '[]',
    web_sources              TEXT NOT NULL DEFAULT '[]',
    web_results              TEXT NOT NULL DEFAULT '[]',
    reasoning                TEXT NOT NULL DEFAULT '[]',
    provenance               TEXT NOT NULL DEFAULT '[]',
    citations                TEXT NOT NULL DEFAULT '[]',
    response_mode            TEXT,
    evidence                 TEXT,
    feedback_available       INTEGER NOT NULL DEFAULT 0,
    feedback_outcome         TEXT NOT NULL DEFAULT 'unknown',
    feedback_category        TEXT NOT NULL DEFAULT '',
    feedback_category_label  TEXT NOT NULL DEFAULT '',
    feedback_reason          TEXT NOT NULL DEFAULT '',
    feedback_correction      TEXT NOT NULL DEFAULT '',
    experience_id            TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS ix_tasks_updated_at ON tasks(updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks(status);

CREATE TABLE IF NOT EXISTS user_memories (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL DEFAULT 'preference',
    text            TEXT NOT NULL DEFAULT '',
    summary         TEXT NOT NULL DEFAULT '',
    source          TEXT NOT NULL DEFAULT 'user_message',
    conversation_id TEXT NOT NULL DEFAULT '',
    message_id      TEXT NOT NULL DEFAULT '',
    confidence      REAL NOT NULL DEFAULT 0.6,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    ordinal         INTEGER NOT NULL DEFAULT 0,
    metadata        TEXT NOT NULL DEFAULT '{}'
);

-- Insertion order is what "the stored history" means for memory, and the store
-- is bounded, so an explicit ordinal is cheaper and clearer than timestamps.
CREATE INDEX IF NOT EXISTS ix_user_memories_ordinal ON user_memories(ordinal);

CREATE TABLE IF NOT EXISTS experiences (
    experience_id             TEXT PRIMARY KEY,
    recorded_at               TEXT NOT NULL,
    ordinal                   INTEGER NOT NULL DEFAULT 0,
    experience_type           TEXT NOT NULL DEFAULT 'successful_task',
    record_id                 TEXT NOT NULL DEFAULT '',
    task_id                   TEXT NOT NULL DEFAULT '',
    conversation_id           TEXT NOT NULL DEFAULT '',
    model_used                TEXT NOT NULL DEFAULT '',
    original_user_request     TEXT NOT NULL DEFAULT '',
    interpreted_intent        TEXT NOT NULL DEFAULT '',
    desired_outcome           TEXT NOT NULL DEFAULT '',
    task_type                 TEXT NOT NULL DEFAULT 'unknown',
    goal                      TEXT NOT NULL DEFAULT 'unknown',
    request_type              TEXT NOT NULL DEFAULT 'unknown',
    requested_destination     TEXT NOT NULL DEFAULT '',
    requested_format          TEXT NOT NULL DEFAULT '',
    final_result              TEXT NOT NULL DEFAULT '',
    outcome                   TEXT NOT NULL DEFAULT 'unknown',
    feedback                  TEXT NOT NULL DEFAULT 'unknown',
    feedback_reason           TEXT NOT NULL DEFAULT '',
    failure_category          TEXT NOT NULL DEFAULT '',
    user_correction           TEXT NOT NULL DEFAULT '',
    failed_step               TEXT NOT NULL DEFAULT '',
    expected_behavior         TEXT NOT NULL DEFAULT '',
    actual_behavior           TEXT NOT NULL DEFAULT '',
    response_quality          TEXT NOT NULL DEFAULT '',
    state                     TEXT NOT NULL DEFAULT 'new',
    attempt                   INTEGER NOT NULL DEFAULT 1,
    recovers_experience_id    TEXT NOT NULL DEFAULT '',
    confirmations             INTEGER NOT NULL DEFAULT 0,
    pattern_key               TEXT NOT NULL DEFAULT '',
    -- Genuinely variable shapes stay serialized, but only as single columns.
    extracted_entities        TEXT NOT NULL DEFAULT '{}',
    constraints               TEXT NOT NULL DEFAULT '[]',
    generated_plan            TEXT NOT NULL DEFAULT '[]',
    tools_used                TEXT NOT NULL DEFAULT '[]',
    execution_steps           TEXT NOT NULL DEFAULT '[]',
    verification_result       TEXT NOT NULL DEFAULT '{}',
    completion_checks         TEXT NOT NULL DEFAULT '{}',
    retrieved_experience_ids  TEXT NOT NULL DEFAULT '[]',
    reasoning_strategy_ids    TEXT NOT NULL DEFAULT '[]'
);

-- Retrieval groups by outcome and scans the whole bounded history, so the two
-- hot predicates get real indexes rather than a table scan.
CREATE INDEX IF NOT EXISTS ix_experiences_outcome ON experiences(outcome);
CREATE INDEX IF NOT EXISTS ix_experiences_record ON experiences(record_id);
CREATE INDEX IF NOT EXISTS ix_experiences_ordinal ON experiences(ordinal);

CREATE TABLE IF NOT EXISTS feedback_events (
    ordinal           INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id         TEXT NOT NULL,
    outcome           TEXT NOT NULL DEFAULT 'unknown',
    reason            TEXT NOT NULL DEFAULT '',
    failure_category  TEXT NOT NULL DEFAULT '',
    correction        TEXT NOT NULL DEFAULT '',
    expected_behavior TEXT NOT NULL DEFAULT '',
    response_quality  TEXT NOT NULL DEFAULT '',
    recorded_at       TEXT NOT NULL DEFAULT ''
);

-- "Most recent feedback for this record" is a real query, not a scan.
CREATE INDEX IF NOT EXISTS ix_feedback_events_record
    ON feedback_events(record_id, ordinal DESC);

-- Knowledge indexing state. The document itself stays on disk; this is the
-- metadata that says whether its embeddings are current.
CREATE TABLE IF NOT EXISTS index_documents (
    doc_id         TEXT PRIMARY KEY,
    path           TEXT NOT NULL DEFAULT '',
    content_hash   TEXT NOT NULL DEFAULT '',
    chunk_count    INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'indexed',
    last_indexed   TEXT NOT NULL DEFAULT '',
    metadata       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS correction_lessons (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_type      TEXT NOT NULL,
    pattern          TEXT NOT NULL,
    correct_behavior TEXT NOT NULL DEFAULT '[]',
    fields           TEXT NOT NULL DEFAULT '{}',
    confidence       REAL NOT NULL DEFAULT 0.0,
    hits             INTEGER NOT NULL DEFAULT 0,
    recorded_at      TEXT NOT NULL DEFAULT '',
    -- A lesson is identified by what it corrects, so re-running an import can
    -- never teach the same lesson twice.
    UNIQUE (lesson_type, pattern)
);

-- Legacy import bookkeeping. Idempotent migration needs to know what has
-- already been imported without re-reading (or re-writing) every legacy file.
CREATE TABLE IF NOT EXISTS legacy_migrations (
    source      TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL DEFAULT '',
    imported    INTEGER NOT NULL DEFAULT 0,
    skipped     INTEGER NOT NULL DEFAULT 0,
    failed      INTEGER NOT NULL DEFAULT 0,
    detail      TEXT NOT NULL DEFAULT '',
    applied_at  TEXT NOT NULL DEFAULT ''
);
"""
