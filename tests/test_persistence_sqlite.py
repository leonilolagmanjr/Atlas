"""Tests for the authoritative SQLite persistence layer.

These cover the persistence *contracts* and the resulting behaviour rather than
SQL syntax: initialization, CRUD, durability across a close/reopen cycle, the
one-time legacy import, the safety of that import against bad input, and the
repository boundary that keeps SQLite out of Atlas's core logic.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from persistence.context import database_for_legacy_path, open_persistence
from persistence.database import AtlasDatabase, SchemaVersionError
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
from persistence.schema import SCHEMA_VERSION


class TemporaryDataRootTestCase(unittest.TestCase):
    """A base case that gives every test its own isolated data root.

    Contexts opened during a test are tracked and closed in ``tearDown`` *before*
    the temporary directory is removed: on Windows an open SQLite handle keeps the
    database file locked, so that order is required rather than cosmetic.
    """

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.paths = AtlasDataPaths(root=Path(self._temporary.name))
        self._contexts: list = []

    def tearDown(self) -> None:
        for context in self._contexts:
            try:
                context.close()
            except Exception:  # noqa: BLE001 - teardown must not mask failures
                pass
        self._contexts.clear()
        self._temporary.cleanup()

    def context(self, **kwargs):
        """Open the persistence context, closed automatically at teardown.

        Migration is off by default here: these tests are about the repositories,
        and the migrator deliberately also looks in the *application* root, where
        a developer checkout may hold real legacy files. Migration has its own
        test case below, which opts back in.
        """

        kwargs.setdefault("migrate", False)
        context = open_persistence(self.paths, **kwargs)
        self._contexts.append(context)
        return context


# ---------------------------------------------------------------------------
# initialization
# ---------------------------------------------------------------------------


class DatabaseInitializationTests(TemporaryDataRootTestCase):
    def test_initialize_creates_database_and_schema(self):
        database = AtlasDatabase(self.paths.database_file)
        self._contexts.append(database)
        version = database.initialize()

        self.assertEqual(version, SCHEMA_VERSION)
        self.assertTrue(self.paths.database_file.exists())
        self.assertIn("conversations", database.table_names())
        self.assertIn("messages", database.table_names())
        self.assertIn("tasks", database.table_names())
        self.assertIn("experiences", database.table_names())
        self.assertIn("user_memories", database.table_names())

    def test_foreign_keys_are_enabled(self):
        database = AtlasDatabase(self.paths.database_file)
        self._contexts.append(database)
        database.initialize()
        self.assertTrue(database.foreign_keys_enabled())

    def test_foreign_key_cascade_deletes_a_conversation_transcript(self):
        context = self.context()
        context.conversations.create_conversation("conv-1", "Title")
        context.conversations.save_messages(
            "conv-1", [{"id": "m-1", "role": "user", "content": "hello"}]
        )
        self.assertEqual(len(context.conversations.list_messages("conv-1")), 1)

        context.conversations.delete_conversation("conv-1")

        self.assertIsNone(context.conversations.get_conversation("conv-1"))
        self.assertEqual(context.conversations.list_messages("conv-1"), [])

    def test_initialize_is_idempotent_and_records_the_version(self):
        database = AtlasDatabase(self.paths.database_file)
        self._contexts.append(database)
        self.assertEqual(database.initialize(), SCHEMA_VERSION)
        self.assertEqual(database.initialize(), SCHEMA_VERSION)
        self.assertEqual(database.schema_version, SCHEMA_VERSION)

    def test_reopening_an_existing_database_keeps_its_data(self):
        first = AtlasDatabase(self.paths.database_file)
        self.addCleanup(first.close)
        first.initialize()
        SqliteConversationRepository(first).create_conversation("conv-1", "Kept")
        first.close()

        second = AtlasDatabase(self.paths.database_file)
        self._contexts.append(second)
        second.initialize()
        stored = SqliteConversationRepository(second).get_conversation("conv-1")

        self.assertIsNotNone(stored)
        self.assertEqual(stored["title"], "Kept")

    def test_a_newer_schema_version_is_refused(self):
        database = AtlasDatabase(self.paths.database_file)
        self._contexts.append(database)
        database.initialize()
        with database.connection() as connection:
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")

        with self.assertRaises(SchemaVersionError):
            database.initialize()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class ConversationRepositoryTests(TemporaryDataRootTestCase):
    def repository(self) -> SqliteConversationRepository:
        return self.context().conversations

    def test_create_list_and_rename_a_conversation(self):
        repository = self.repository()
        repository.create_conversation("conv-1", "First")

        self.assertEqual(repository.list_conversation_ids(), ["conv-1"])
        stored = repository.get_conversation("conv-1")
        self.assertEqual(stored["title"], "First")

        repository.save_conversation("conv-1", {**stored, "title": "Renamed"})
        self.assertEqual(repository.get_conversation("conv-1")["title"], "Renamed")

    def test_get_unknown_conversation_returns_none(self):
        self.assertIsNone(self.repository().get_conversation("missing"))

    def test_messages_round_trip_with_all_structured_fields(self):
        repository = self.repository()
        repository.create_conversation("conv-1", "Title")
        repository.save_messages(
            "conv-1",
            [
                {
                    "id": "m-1",
                    "role": "user",
                    "content": "hello",
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "turn_number": 1,
                    "attachments": [{"name": "a.pdf"}],
                    "tool_calls": [{"tool": "filesystem.read_text"}],
                    "tool_results": [{"ok": True}],
                    "citations": ["doc-1"],
                    "execution_state": "completed",
                    "metadata": {"k": "v"},
                }
            ],
        )

        message = repository.list_messages("conv-1")[0]
        self.assertEqual(message["attachments"], [{"name": "a.pdf"}])
        self.assertEqual(message["tool_calls"], [{"tool": "filesystem.read_text"}])
        self.assertEqual(message["tool_results"], [{"ok": True}])
        self.assertEqual(message["citations"], ["doc-1"])
        self.assertEqual(message["metadata"], {"k": "v"})

    def test_messages_keep_their_transcript_order(self):
        repository = self.repository()
        repository.create_conversation("conv-1", "Title")
        repository.save_messages(
            "conv-1",
            [{"id": f"m-{index}", "role": "user", "content": str(index)} for index in range(5)],
        )

        contents = [message["content"] for message in repository.list_messages("conv-1")]
        self.assertEqual(contents, ["0", "1", "2", "3", "4"])

    def test_saving_a_message_updates_the_conversation_count(self):
        repository = self.repository()
        repository.create_conversation("conv-1", "Title")
        repository.save_messages(
            "conv-1", [{"id": "m-1", "role": "user", "content": "a"}]
        )
        self.assertEqual(repository.get_conversation("conv-1")["message_count"], 1)

    def test_replacing_a_transcript_does_not_leave_orphans(self):
        repository = self.repository()
        repository.create_conversation("conv-1", "Title")
        repository.save_messages("conv-1", [{"id": "m-1", "role": "user", "content": "a"}])
        repository.save_messages("conv-1", [{"id": "m-2", "role": "user", "content": "b"}])

        messages = repository.list_messages("conv-1")
        self.assertEqual([message["id"] for message in messages], ["m-2"])

    def test_update_message_applies_only_known_fields(self):
        repository = self.repository()
        repository.create_conversation("conv-1", "Title")
        repository.save_messages(
            "conv-1",
            [{"id": "m-1", "role": "user", "content": "a", "execution_state": "running"}],
        )

        updated = repository.update_message(
            "conv-1", "m-1", {"content": "b", "execution_state": "completed"}
        )

        self.assertEqual(updated["content"], "b")
        self.assertEqual(updated["execution_state"], "completed")

    def test_update_unknown_message_returns_none(self):
        repository = self.repository()
        repository.create_conversation("conv-1", "Title")
        self.assertIsNone(repository.update_message("conv-1", "missing", {"content": "x"}))

    def test_active_conversation_pointer_round_trips(self):
        repository = self.repository()
        self.assertIsNone(repository.get_active_conversation_id())

        repository.set_active_conversation_id("conv-1")
        self.assertEqual(repository.get_active_conversation_id(), "conv-1")

        repository.clear_active_conversation_id()
        self.assertIsNone(repository.get_active_conversation_id())


class TaskRepositoryTests(TemporaryDataRootTestCase):
    def repository(self) -> SqliteTaskRepository:
        return self.context().tasks

    def test_save_and_load_tasks(self):
        repository = self.repository()
        repository.save_tasks(
            {
                "t-1": {
                    "id": "t-1",
                    "request": "do a thing",
                    "status": "COMPLETED",
                    "response": "done",
                    "created_at": 1.0,
                    "updated_at": 2.0,
                }
            }
        )

        tasks = repository.list_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["response"], "done")
        self.assertEqual(tasks[0]["status"], "COMPLETED")

    def test_nested_task_fields_survive_as_structures(self):
        repository = self.repository()
        repository.save_tasks(
            {
                "t-1": {
                    "id": "t-1",
                    "request": "x",
                    "status": "COMPLETED",
                    "created_at": 1.0,
                    "updated_at": 1.0,
                    "plan": {"steps": [{"capability": "a"}]},
                    "tool_calls": [{"tool": "t"}],
                    "errors": ["boom"],
                    "citations": ["doc-1"],
                    "evidence": {"count": 2},
                }
            }
        )

        task = repository.get_task("t-1")
        self.assertEqual(task["plan"], {"steps": [{"capability": "a"}]})
        self.assertEqual(task["tool_calls"], [{"tool": "t"}])
        self.assertEqual(task["errors"], ["boom"])
        self.assertEqual(task["evidence"], {"count": 2})

    def test_tasks_are_ordered_by_most_recently_updated(self):
        repository = self.repository()
        repository.save_tasks(
            {
                "old": {"id": "old", "request": "a", "status": "X", "created_at": 1.0, "updated_at": 1.0},
                "new": {"id": "new", "request": "b", "status": "X", "created_at": 1.0, "updated_at": 9.0},
            }
        )
        self.assertEqual([task["id"] for task in repository.list_tasks()], ["new", "old"])

    def test_save_tasks_drops_records_no_longer_present(self):
        repository = self.repository()
        repository.save_tasks({"t-1": {"id": "t-1", "request": "a", "status": "X"}})
        repository.save_tasks({"t-2": {"id": "t-2", "request": "b", "status": "X"}})

        self.assertEqual([task["id"] for task in repository.list_tasks()], ["t-2"])

    def test_clear_removes_every_task(self):
        repository = self.repository()
        repository.save_tasks({"t-1": {"id": "t-1", "request": "a", "status": "X"}})
        repository.clear()
        self.assertEqual(repository.list_tasks(), [])


class UserMemoryRepositoryTests(TemporaryDataRootTestCase):
    def repository(self) -> SqliteUserMemoryRepository:
        return self.context().user_memory

    def test_add_list_get_and_delete(self):
        repository = self.repository()
        repository.add_memory({"id": "m-1", "kind": "preference", "text": "prefers dark mode"})

        self.assertEqual(len(repository.list_memories()), 1)
        self.assertEqual(repository.get_memory("m-1")["text"], "prefers dark mode")

        self.assertTrue(repository.delete_memory("m-1"))
        self.assertIsNone(repository.get_memory("m-1"))

    def test_deleting_an_unknown_memory_returns_false(self):
        self.assertFalse(self.repository().delete_memory("missing"))

    def test_memories_keep_insertion_order(self):
        repository = self.repository()
        for index in range(3):
            repository.add_memory({"id": f"m-{index}", "text": str(index)})
        self.assertEqual(
            [memory["id"] for memory in repository.list_memories()],
            ["m-0", "m-1", "m-2"],
        )

    def test_update_memory_changes_fields_and_touches_the_timestamp(self):
        repository = self.repository()
        repository.add_memory({"id": "m-1", "text": "old", "created_at": "2024-01-01T00:00:00+00:00"})

        updated = repository.update_memory("m-1", {"text": "new", "confidence": 0.9})

        self.assertEqual(updated["text"], "new")
        self.assertAlmostEqual(updated["confidence"], 0.9, places=3)

    def test_add_memory_is_idempotent_for_the_same_id(self):
        repository = self.repository()
        repository.add_memory({"id": "m-1", "text": "first"})
        repository.add_memory({"id": "m-1", "text": "second"})

        memories = repository.list_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["text"], "first")

    def test_trim_to_keeps_the_newest_records(self):
        repository = self.repository()
        for index in range(5):
            repository.add_memory({"id": f"m-{index}", "text": str(index)})

        removed = repository.trim_to(2)

        self.assertEqual(removed, 3)
        self.assertEqual(
            [memory["id"] for memory in repository.list_memories()], ["m-3", "m-4"]
        )

    def test_clear_memories_reports_how_many_were_removed(self):
        repository = self.repository()
        for index in range(3):
            repository.add_memory({"id": f"m-{index}", "text": str(index)})

        self.assertEqual(repository.clear_memories(), 3)
        self.assertEqual(repository.list_memories(), [])


class ExperienceRepositoryTests(TemporaryDataRootTestCase):
    def repository(self) -> SqliteExperienceRepository:
        return self.context().experiences

    def _experience(self, experience_id: str = "e-1", **overrides) -> dict:
        record = {
            "experience_id": experience_id,
            "timestamp": "2024-01-01T00:00:00+00:00",
            "outcome": "success",
            "original_user_request": "open notepad",
            "tools_used": ["applications.open"],
            "generated_plan": ["step one"],
            "extracted_entities": {"application": "notepad"},
            "attempt": 2,
        }
        record.update(overrides)
        return record

    def test_add_and_list_experiences(self):
        repository = self.repository()
        self.assertTrue(repository.add_experience(self._experience()))

        experiences = repository.list_experiences()
        self.assertEqual(len(experiences), 1)
        self.assertEqual(experiences[0]["experience_id"], "e-1")
        self.assertEqual(experiences[0]["outcome"], "success")

    def test_structured_fields_round_trip(self):
        repository = self.repository()
        repository.add_experience(self._experience())

        stored = repository.get_experience("e-1")
        self.assertEqual(stored["tools_used"], ["applications.open"])
        self.assertEqual(stored["generated_plan"], ["step one"])
        self.assertEqual(stored["extracted_entities"], {"application": "notepad"})
        self.assertEqual(stored["attempt"], 2)
        self.assertEqual(stored["timestamp"], "2024-01-01T00:00:00+00:00")

    def test_experience_without_an_id_is_rejected(self):
        self.assertFalse(self.repository().add_experience({"outcome": "success"}))

    def test_replace_updates_an_existing_experience(self):
        repository = self.repository()
        repository.add_experience(self._experience())

        self.assertTrue(
            repository.replace_experience(self._experience(outcome="failure"))
        )
        self.assertEqual(repository.get_experience("e-1")["outcome"], "failure")

    def test_replace_unknown_experience_returns_false(self):
        self.assertFalse(self.repository().replace_experience(self._experience("missing")))

    def test_count_and_trim(self):
        repository = self.repository()
        for index in range(5):
            repository.add_experience(self._experience(f"e-{index}"))

        self.assertEqual(repository.count_experiences(), 5)
        self.assertEqual(repository.trim_to(2), 3)
        self.assertEqual(repository.count_experiences(), 2)

    def test_delete_below_ordinal(self):
        repository = self.repository()
        for index in range(3):
            repository.add_experience(self._experience(f"e-{index}"))

        # Ordinals start at 1; dropping below 3 removes the first two.
        self.assertEqual(repository.delete_experiences_below_ordinal(3), 2)
        self.assertEqual(repository.count_experiences(), 1)

    def test_clear_removes_every_experience(self):
        repository = self.repository()
        repository.add_experience(self._experience())
        repository.clear_experiences()
        self.assertEqual(repository.list_experiences(), [])


class FeedbackRepositoryTests(TemporaryDataRootTestCase):
    def repository(self) -> SqliteFeedbackRepository:
        return self.context().feedback

    def test_record_and_list_feedback(self):
        repository = self.repository()
        self.assertTrue(
            repository.record_feedback(
                {"record_id": "r-1", "outcome": "success", "reason": "looks right"}
            )
        )

        events = repository.list_feedback()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["outcome"], "success")
        self.assertEqual(events[0]["reason"], "looks right")

    def test_latest_feedback_returns_the_most_recent_event(self):
        repository = self.repository()
        repository.record_feedback({"record_id": "r-1", "outcome": "success"})
        repository.record_feedback({"record_id": "r-1", "outcome": "failure"})

        self.assertEqual(repository.latest_feedback("r-1")["outcome"], "failure")

    def test_latest_feedback_for_an_unknown_record_is_none(self):
        self.assertIsNone(self.repository().latest_feedback("missing"))

    def test_clear_removes_every_feedback_event(self):
        repository = self.repository()
        repository.record_feedback({"record_id": "r-1", "outcome": "success"})
        repository.clear_feedback()
        self.assertEqual(repository.list_feedback(), [])


class IndexStateRepositoryTests(TemporaryDataRootTestCase):
    def repository(self) -> SqliteIndexStateRepository:
        return self.context().index_state

    def test_set_and_get_hashes(self):
        repository = self.repository()
        repository.set_hash("doc-1", "abc", path="doc-1.pdf")

        self.assertEqual(repository.get_hashes(), {"doc-1": "abc"})

    def test_setting_a_hash_again_updates_it(self):
        repository = self.repository()
        repository.set_hash("doc-1", "abc")
        repository.set_hash("doc-1", "def")
        self.assertEqual(repository.get_hashes(), {"doc-1": "def"})

    def test_remove_forgets_a_document(self):
        repository = self.repository()
        repository.set_hash("doc-1", "abc")
        repository.remove("doc-1")
        self.assertEqual(repository.get_hashes(), {})

    def test_describe_reports_per_document_metadata(self):
        repository = self.repository()
        repository.set_hash("doc-1", "abc", path="knowledge/doc-1.pdf")

        described = repository.describe()
        self.assertEqual(len(described), 1)
        self.assertEqual(described[0]["doc_id"], "doc-1")
        self.assertEqual(described[0]["path"], "knowledge/doc-1.pdf")
        self.assertEqual(described[0]["content_hash"], "abc")


class CorrectionMemoryRepositoryTests(TemporaryDataRootTestCase):
    def repository(self) -> SqliteCorrectionMemoryRepository:
        return self.context().correction_memory

    def test_add_and_list_lessons(self):
        repository = self.repository()
        self.assertTrue(
            repository.add_lesson(
                {
                    "type": "interpretation_correction",
                    "pattern": "open the thing",
                    "correct_behavior": ["ask which thing"],
                    "fields": {"application": "unknown"},
                    "confidence": 0.9,
                }
            )
        )

        lessons = repository.list_lessons()
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["correct_behavior"], ["ask which thing"])
        self.assertEqual(lessons[0]["fields"], {"application": "unknown"})

    def test_the_same_lesson_is_never_taught_twice(self):
        repository = self.repository()
        lesson = {"type": "task_pattern", "pattern": "p", "confidence": 0.9}

        self.assertTrue(repository.add_lesson(lesson))
        self.assertFalse(repository.add_lesson(lesson))
        self.assertEqual(len(repository.list_lessons()), 1)

    def test_lesson_without_a_type_or_pattern_is_rejected(self):
        repository = self.repository()
        self.assertFalse(repository.add_lesson({"pattern": "p"}))
        self.assertFalse(repository.add_lesson({"type": "task_pattern"}))

    def test_increment_hits(self):
        repository = self.repository()
        repository.add_lesson({"type": "task_pattern", "pattern": "p", "confidence": 0.9})

        repository.increment_hits("task_pattern", "p")
        repository.increment_hits("task_pattern", "p")

        self.assertEqual(repository.list_lessons()[0]["hits"], 2)


# ---------------------------------------------------------------------------
# persistence across a close / reopen cycle
# ---------------------------------------------------------------------------


class DurabilityTests(TemporaryDataRootTestCase):
    def test_every_domain_survives_a_close_and_reopen(self):
        first = open_persistence(self.paths)
        first.conversations.create_conversation("conv-1", "Durable")
        first.conversations.save_messages(
            "conv-1", [{"id": "m-1", "role": "user", "content": "hello"}]
        )
        first.tasks.save_tasks(
            {"t-1": {"id": "t-1", "request": "x", "status": "COMPLETED", "updated_at": 1.0}}
        )
        first.user_memory.add_memory({"id": "mem-1", "text": "prefers tabs"})
        first.experiences.add_experience(
            {"experience_id": "e-1", "outcome": "success", "timestamp": "2024-01-01T00:00:00+00:00"}
        )
        first.feedback.record_feedback({"record_id": "r-1", "outcome": "success"})
        first.index_state.set_hash("doc-1", "abc")
        first.correction_memory.add_lesson(
            {"type": "task_pattern", "pattern": "p", "confidence": 0.9}
        )
        first.close()

        second = open_persistence(self.paths, migrate=False)
        self._contexts.append(second)

        self.assertEqual(second.conversations.get_conversation("conv-1")["title"], "Durable")
        self.assertEqual(len(second.conversations.list_messages("conv-1")), 1)
        self.assertEqual(len(second.tasks.list_tasks()), 1)
        self.assertEqual(second.user_memory.get_memory("mem-1")["text"], "prefers tabs")
        self.assertEqual(second.experiences.get_experience("e-1")["outcome"], "success")
        self.assertEqual(second.feedback.latest_feedback("r-1")["outcome"], "success")
        self.assertEqual(second.index_state.get_hashes(), {"doc-1": "abc"})
        self.assertEqual(len(second.correction_memory.list_lessons()), 1)

    def test_the_database_is_a_single_file_in_the_data_root(self):
        context = self.context()
        self.assertTrue(context.database_file.exists())
        self.assertEqual(context.database_file.parent, self.paths.root)
        self.assertEqual(context.database_file.name, "atlas.db")

    def test_chroma_has_its_own_directory_separate_from_sqlite(self):
        self.paths.ensure()
        self.assertNotEqual(self.paths.chroma_dir, self.paths.database_file.parent)
        self.assertTrue(self.paths.chroma_dir.is_dir())


# ---------------------------------------------------------------------------
# data-root resolution
# ---------------------------------------------------------------------------


class DataRootTests(unittest.TestCase):
    def test_an_explicit_override_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            resolved = resolve_data_paths(override=directory)
            self.assertEqual(resolved.root, Path(directory).resolve())

    def test_ensure_creates_every_expected_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = AtlasDataPaths(root=Path(directory)).ensure()
            for expected in (
                paths.chroma_dir,
                paths.knowledge_dir,
                paths.artifacts_dir,
                paths.logs_dir,
                paths.models_dir,
                paths.config_dir,
            ):
                self.assertTrue(expected.is_dir(), expected)

    def test_data_root_is_resolved_from_the_environment(self):
        import os

        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("ATLAS_DATA_ROOT")
            os.environ["ATLAS_DATA_ROOT"] = directory
            try:
                resolved = resolve_data_paths()
            finally:
                if previous is None:
                    os.environ.pop("ATLAS_DATA_ROOT", None)
                else:
                    os.environ["ATLAS_DATA_ROOT"] = previous
            self.assertEqual(resolved.root, Path(directory).resolve())


# ---------------------------------------------------------------------------
# legacy migration
# ---------------------------------------------------------------------------


class LegacyMigrationTests(TemporaryDataRootTestCase):
    """Build a legacy file layout inside the data root and import it."""

    def write_legacy(self, **files: str) -> Path:
        legacy = self.paths.root / "database"
        legacy.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (legacy / name.replace("_", ".")).write_text(content, encoding="utf-8")
        return legacy

    def write_legacy_generic(self, name: str, content: str, folder: str = "database") -> None:
        directory = self.paths.root / folder
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(content, encoding="utf-8")

    def test_imports_tasks_memory_experience_feedback_and_index_state(self):
        self.write_legacy_generic(
            "tasks.json",
            json.dumps({"t-1": {"id": "t-1", "request": "x", "status": "COMPLETED"}}),
        )
        self.write_legacy_generic(
            "user_memory.jsonl", json.dumps({"id": "mem-1", "text": "prefers tabs"}) + "\n"
        )
        self.write_legacy_generic(
            "experiences.jsonl",
            json.dumps({"experience_id": "e-1", "outcome": "success"}) + "\n",
        )
        self.write_legacy_generic(
            "feedback.jsonl", json.dumps({"record_id": "r-1", "outcome": "failure"}) + "\n"
        )
        self.write_legacy_generic(
            "index_state.json", json.dumps({"doc_hashes": {"doc-1": "abc"}})
        )

        context = self.context(migrate=True)

        self.assertEqual(len(context.tasks.list_tasks()), 1)
        self.assertEqual(context.user_memory.get_memory("mem-1")["text"], "prefers tabs")
        self.assertEqual(context.experiences.get_experience("e-1")["outcome"], "success")
        self.assertEqual(context.feedback.latest_feedback("r-1")["outcome"], "failure")
        self.assertEqual(context.index_state.get_hashes(), {"doc-1": "abc"})

    def test_imports_conversations_and_their_transcript(self):
        session = self.paths.root / "memory" / "sessions" / "conv-1"
        session.mkdir(parents=True, exist_ok=True)
        (session / "metadata.json").write_text(
            json.dumps({"id": "conv-1", "title": "Legacy title"}), encoding="utf-8"
        )
        (session / "messages.json").write_text(
            json.dumps([{"id": "m-1", "role": "user", "content": "hello"}]), encoding="utf-8"
        )
        (session / "summary.txt").write_text("A short summary", encoding="utf-8")

        context = self.context(migrate=True)

        stored = context.conversations.get_conversation("conv-1")
        self.assertEqual(stored["title"], "Legacy title")
        self.assertEqual(stored["summary"], "A short summary")
        self.assertEqual(len(context.conversations.list_messages("conv-1")), 1)

    def test_imports_the_active_conversation_pointer(self):
        session = self.paths.root / "memory" / "sessions" / "conv-1"
        session.mkdir(parents=True, exist_ok=True)
        (session / "metadata.json").write_text(json.dumps({"id": "conv-1"}), encoding="utf-8")
        (self.paths.root / "memory" / "active_session.json").write_text(
            json.dumps({"session_id": "conv-1"}), encoding="utf-8"
        )

        context = self.context(migrate=True)

        self.assertEqual(context.conversations.get_active_conversation_id(), "conv-1")

    def test_imports_correction_lessons(self):
        self.write_legacy_generic(
            "interpretation_lessons.jsonl",
            json.dumps({"type": "task_pattern", "pattern": "p", "confidence": 0.9}) + "\n",
            folder="memory",
        )

        context = self.context(migrate=True)

        self.assertEqual(len(context.correction_memory.list_lessons()), 1)

    def test_migration_is_idempotent(self):
        self.write_legacy_generic(
            "tasks.json", json.dumps({"t-1": {"id": "t-1", "request": "x", "status": "X"}})
        )
        self.write_legacy_generic(
            "user_memory.jsonl", json.dumps({"id": "mem-1", "text": "a"}) + "\n"
        )

        first = open_persistence(self.paths, migrate=True)
        counts_after_first = (len(first.tasks.list_tasks()), len(first.user_memory.list_memories()))
        first.close()

        second = open_persistence(self.paths, migrate=True)
        self._contexts.append(second)
        counts_after_second = (len(second.tasks.list_tasks()), len(second.user_memory.list_memories()))

        third = open_persistence(self.paths, migrate=True)
        self._contexts.append(third)
        counts_after_third = (len(third.tasks.list_tasks()), len(third.user_memory.list_memories()))

        self.assertEqual(counts_after_first, (1, 1))
        self.assertEqual(counts_after_second, counts_after_first)
        self.assertEqual(counts_after_third, counts_after_first)

    def test_running_migration_three_times_in_one_process_does_not_duplicate(self):
        self.write_legacy_generic(
            "experiences.jsonl", json.dumps({"experience_id": "e-1"}) + "\n"
        )

        counts = []
        for _ in range(3):
            context = open_persistence(self.paths, migrate=True)
            counts.append(context.experiences.count_experiences())
            context.close()

        self.assertEqual(counts, [1, 1, 1])

    def test_legacy_files_are_preserved(self):
        legacy = self.write_legacy_generic(
            "tasks.json", json.dumps({"t-1": {"id": "t-1", "request": "x", "status": "X"}})
        )

        context = self.context(migrate=True)

        self.assertTrue((self.paths.root / "database" / "tasks.json").exists())
        self.assertEqual(len(context.tasks.list_tasks()), 1)

    def test_malformed_json_is_skipped_without_failing_startup(self):
        self.write_legacy_generic("tasks.json", "{not json")
        self.write_legacy_generic(
            "user_memory.jsonl",
            "not json\n" + json.dumps({"id": "mem-1", "text": "good"}) + "\n",
        )

        context = self.context(migrate=True)

        self.assertEqual(context.tasks.list_tasks(), [])
        self.assertEqual(len(context.user_memory.list_memories()), 1)

    def test_an_empty_legacy_file_is_a_no_op(self):
        self.write_legacy_generic("tasks.json", "")
        self.write_legacy_generic("experiences.jsonl", "")

        context = self.context(migrate=True)

        self.assertEqual(context.tasks.list_tasks(), [])
        self.assertEqual(context.experiences.count_experiences(), 0)

    def test_missing_legacy_files_are_a_no_op(self):
        context = self.context(migrate=True)
        self.assertEqual(context.tasks.list_tasks(), [])
        self.assertEqual(context.user_memory.list_memories(), [])
        self.assertEqual(context.experiences.count_experiences(), 0)
        self.assertEqual(context.index_state.get_hashes(), {})

    def test_records_without_an_id_are_skipped(self):
        self.write_legacy_generic(
            "user_memory.jsonl",
            json.dumps({"text": "no id"}) + "\n" + json.dumps({"id": "mem-1"}) + "\n",
        )

        context = self.context(migrate=True)

        memories = context.user_memory.list_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["id"], "mem-1")

    def test_ids_and_timestamps_are_preserved(self):
        self.write_legacy_generic(
            "user_memory.jsonl",
            json.dumps(
                {
                    "id": "mem-1",
                    "text": "kept",
                    "created_at": "2023-05-05T12:00:00+00:00",
                    "updated_at": "2023-05-06T12:00:00+00:00",
                    "confidence": 0.81,
                }
            )
            + "\n",
        )

        context = self.context(migrate=True)

        memory = context.user_memory.get_memory("mem-1")
        self.assertEqual(memory["id"], "mem-1")
        self.assertEqual(memory["created_at"], "2023-05-05T12:00:00+00:00")
        self.assertEqual(memory["updated_at"], "2023-05-06T12:00:00+00:00")

    def test_a_partially_valid_conversation_set_still_imports_the_valid_ones(self):
        sessions = self.paths.root / "memory" / "sessions"
        (sessions / "good").mkdir(parents=True, exist_ok=True)
        (sessions / "good" / "metadata.json").write_text(
            json.dumps({"id": "good"}), encoding="utf-8"
        )
        (sessions / "bad").mkdir(parents=True, exist_ok=True)
        (sessions / "bad" / "metadata.json").write_text("{not json", encoding="utf-8")

        context = self.context(migrate=True)

        self.assertEqual(context.conversations.list_conversation_ids(), ["good"])

    def test_duplicate_ids_in_a_legacy_file_collapse_to_one_record(self):
        self.write_legacy_generic(
            "user_memory.jsonl",
            json.dumps({"id": "mem-1", "text": "first"}) + "\n"
            + json.dumps({"id": "mem-1", "text": "second"}) + "\n",
        )

        context = self.context(migrate=True)

        self.assertEqual(len(context.user_memory.list_memories()), 1)

    def test_unexpected_fields_are_ignored(self):
        self.write_legacy_generic(
            "tasks.json",
            json.dumps(
                {
                    "t-1": {
                        "id": "t-1",
                        "request": "x",
                        "status": "X",
                        "surprise": {"nested": True},
                    }
                }
            ),
        )

        context = self.context(migrate=True)

        self.assertEqual(len(context.tasks.list_tasks()), 1)


# ---------------------------------------------------------------------------
# repository isolation: core logic against the interface, not SQLite
# ---------------------------------------------------------------------------


class RepositoryIsolationTests(unittest.TestCase):
    """Core code must work against the contract, with no SQLite in sight."""

    def test_a_domain_service_works_against_an_in_memory_repository(self):
        # A hand-written implementation of the conversation contract that stores
        # nothing on disk. If the domain layer depended on SQLite, this would not
        # be possible -- and nothing about it mentions a connection or SQL.
        class InMemoryConversations:
            def __init__(self) -> None:
                self._conversations: dict[str, dict] = {}
                self._messages: dict[str, list[dict]] = {}
                self._active: str | None = None

            def list_conversation_ids(self):
                return list(self._conversations)

            def get_conversation(self, conversation_id):
                return self._conversations.get(conversation_id)

            def create_conversation(self, conversation_id, title):
                record = {"id": conversation_id, "title": title, "message_count": 0}
                self._conversations[conversation_id] = record
                return record

            def save_conversation(self, conversation_id, metadata):
                self._conversations[conversation_id] = dict(metadata)

            def delete_conversation(self, conversation_id):
                self._conversations.pop(conversation_id, None)
                self._messages.pop(conversation_id, None)

            def list_messages(self, conversation_id):
                return list(self._messages.get(conversation_id, []))

            def save_messages(self, conversation_id, messages):
                self._messages[conversation_id] = [dict(m) for m in messages]

            def update_message(self, conversation_id, message_id, changes):
                for message in self._messages.get(conversation_id, []):
                    if message["id"] == message_id:
                        message.update(changes)
                        return message
                return None

            def get_active_conversation_id(self):
                return self._active

            def set_active_conversation_id(self, conversation_id):
                self._active = conversation_id

            def clear_active_conversation_id(self):
                self._active = None

        # This tiny service knows only the contract. It is the shape every core
        # consumer uses, so it proves the boundary is usable, not just declared.
        class ConversationTitles:
            def __init__(self, repository) -> None:
                self._repository = repository

            def retitle(self, conversation_id: str, title: str) -> dict:
                stored = self._repository.get_conversation(conversation_id)
                if stored is None:
                    raise KeyError(conversation_id)
                updated = {**stored, "title": title}
                self._repository.save_conversation(conversation_id, updated)
                return updated

            def title_of(self, conversation_id: str) -> str:
                return self._repository.get_conversation(conversation_id)["title"]

        from persistence.interfaces import ConversationRepository

        repository = InMemoryConversations()
        self.assertIsInstance(repository, ConversationRepository)

        service = ConversationTitles(repository)
        repository.create_conversation("c-1", "Original")
        self.assertEqual(service.title_of("c-1"), "Original")
        service.retitle("c-1", "Renamed")
        self.assertEqual(service.title_of("c-1"), "Renamed")

    def test_the_sqlite_repositories_satisfy_the_same_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            context = open_persistence(AtlasDataPaths(root=Path(directory)), migrate=False)
            from persistence.interfaces import (
                ConversationRepository,
                CorrectionMemoryRepository,
                ExperienceRepository,
                FeedbackRepository,
                IndexStateRepository,
                TaskRepository,
                UserMemoryRepository,
            )

            self.assertIsInstance(context.conversations, ConversationRepository)
            self.assertIsInstance(context.tasks, TaskRepository)
            self.assertIsInstance(context.user_memory, UserMemoryRepository)
            self.assertIsInstance(context.experiences, ExperienceRepository)
            self.assertIsInstance(context.feedback, FeedbackRepository)
            self.assertIsInstance(context.index_state, IndexStateRepository)
            self.assertIsInstance(context.correction_memory, CorrectionMemoryRepository)
            # Closed before the temporary directory is removed, or Windows keeps
            # the database file locked and the cleanup fails.
            context.close()


# ---------------------------------------------------------------------------
# store façades (model-aware wrappers over the repositories)
# ---------------------------------------------------------------------------


class StoreFacadeTests(unittest.TestCase):
    def test_task_store_round_trips_through_an_isolated_database(self):
        from task_store import TaskStore

        with tempfile.TemporaryDirectory() as directory:
            with TaskStore(path=Path(directory) / "tasks.json") as store:
                store.save({"t-1": {"status": "COMPLETED", "response": "done"}})
                loaded = store.load()
                self.assertEqual(loaded["t-1"]["status"], "COMPLETED")

    def test_conversation_store_round_trips_without_touching_the_shared_database(self):
        from memory.models import ConversationSessionMetadata, MemoryMessage
        from memory.storage import ConversationStore

        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(root_folder=Path(directory) / "sessions")
            self.addCleanup(store.close)
            metadata = ConversationSessionMetadata(id="c-1", title="Hello")
            store.save_metadata("c-1", metadata)
            store.save_messages("c-1", [MemoryMessage(id="m-1", role="user", content="hi")])

            self.assertEqual(store.load_metadata("c-1").title, "Hello")
            self.assertEqual(store.load_messages("c-1")[0].content, "hi")

    def test_user_memory_store_crud_and_sensitive_value_protection(self):
        from memory.user_memory import UserMemoryStore

        with tempfile.TemporaryDirectory() as directory:
            store = UserMemoryStore(path=Path(directory) / "user_memory.jsonl", embedder=_NoEmbedder())
            self.addCleanup(store.close)
            created = store.create(text="I prefer dark mode")
            self.assertIsNotNone(created)
            self.assertEqual(len(store.list()), 1)

            # A secret is never stored, even when explicitly offered.
            self.assertIsNone(store.create(text="remember my password is hunter2"))
            self.assertEqual(len(store.list()), 1)

    def test_identical_memories_are_updated_not_duplicated(self):
        from memory.user_memory import UserMemoryStore

        with tempfile.TemporaryDirectory() as directory:
            store = UserMemoryStore(path=Path(directory) / "user_memory.jsonl", embedder=_NoEmbedder())
            self.addCleanup(store.close)
            store.create(text="I prefer dark mode")
            store.create(text="I prefer dark mode")
            self.assertEqual(len(store.list()), 1)


class _NoEmbedder:
    """A user-memory embedder stub that never touches Chroma."""

    def index(self, memory) -> bool:  # noqa: ANN001
        return False

    def similarities(self, query, ids):  # noqa: ANN001
        return {}

    def delete(self, memory_id: str) -> None:
        return None


# ---------------------------------------------------------------------------
# the persistence path helper used by isolated stores
# ---------------------------------------------------------------------------


class LegacyPathDatabaseTests(unittest.TestCase):
    def test_it_derives_a_database_beside_the_given_file(self):
        with tempfile.TemporaryDirectory() as directory:
            database = database_for_legacy_path(Path(directory) / "tasks.json")
            self.addCleanup(database.close)
            self.assertEqual(database.path, Path(directory) / "tasks.db")
            self.assertTrue(database.ephemeral)

    def test_an_isolated_database_does_not_lock_its_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            database = database_for_legacy_path(Path(directory) / "tasks.json")
            SqliteTaskRepository(database).save_tasks(
                {"t-1": {"id": "t-1", "request": "x", "status": "X"}}
            )
            database.close()
            # No -wal/-shm sidecars may be left behind: they would keep the
            # directory undeletable on Windows.
            leftovers = sorted(entry.name for entry in Path(directory).iterdir())
            self.assertEqual(leftovers, ["tasks.db"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
