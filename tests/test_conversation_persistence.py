"""Milestone 1: conversations are first-class and survive a restart.

These tests exercise the *real* store on a temporary root, so what is asserted
is genuine persistence (files written, re-read by a fresh manager), not a fake.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory.memory_manager import MemoryManager
from memory.models import EXECUTION_STATES, MemoryMessage
from memory.storage import ConversationStore


class _Manager:
    """Build a MemoryManager rooted at a temporary folder (restart = new one)."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def open(self) -> MemoryManager:
        manager = MemoryManager()
        manager._store = ConversationStore(root_folder=self.root)
        manager._sessions._store = manager._store  # noqa: SLF001 - same object, one store
        manager._sessions._active_path = self.root / "active_session.json"  # noqa: SLF001
        return manager

    def cleanup(self) -> None:
        self._temporary.cleanup()


class ConversationPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = _Manager()

    def tearDown(self) -> None:
        self.sandbox.cleanup()

    def test_create_list_rename_delete(self) -> None:
        memory = self.sandbox.open()
        first = memory.create_session(title="First")
        second = memory.create_session(title="Second")

        listed = memory.list_sessions()
        self.assertEqual({item.id for item in listed}, {first.id, second.id})

        memory.rename_session(first.id, new_title="Renamed")
        self.assertEqual(memory.open_session(first.id).title, "Renamed")

        memory.delete_session(second.id)
        self.assertEqual([item.id for item in memory.list_sessions()], [first.id])

    def test_messages_survive_a_restart_with_full_shape(self) -> None:
        memory = self.sandbox.open()
        session = memory.create_session(title="Docker")
        memory.append_message(role="user", content="Tell me about Docker.")
        memory.append_message(
            role="assistant",
            content="Docker packages software into containers.",
            attachments=[{"name": "notes.txt", "kind": "text"}],
            tool_calls=[{"tool": "web.search", "status": "completed"}],
            tool_results=[{"tool": "web.search", "output": {"results": []}}],
            citations=["https://docs.docker.com/"],
            execution_state="completed",
            metadata={"response_kind": "research"},
        )

        # A fresh manager over the same root == an Atlas restart.
        reopened = self.sandbox.open()
        reopened.open_session(session.id)
        messages = reopened.get_recent_messages()

        self.assertEqual(len(messages), 2)
        assistant = messages[1]
        self.assertEqual(assistant.content, "Docker packages software into containers.")
        self.assertEqual(assistant.attachments[0]["name"], "notes.txt")
        self.assertEqual(assistant.tool_calls[0]["tool"], "web.search")
        self.assertEqual(assistant.citations, ["https://docs.docker.com/"])
        self.assertEqual(assistant.execution_state, "completed")
        self.assertEqual(assistant.response_kind, "research")

    def test_active_session_pointer_survives_restart(self) -> None:
        memory = self.sandbox.open()
        session = memory.create_session(title="Persisted")
        memory.append_message(role="user", content="hello")

        reopened = self.sandbox.open()
        self.assertEqual(reopened.get_active_session_id(), session.id)
        self.assertEqual(reopened.get_recent_messages()[0].content, "hello")

    def test_message_update_records_a_real_state_change(self) -> None:
        memory = self.sandbox.open()
        memory.create_session(title="Streaming")
        message = memory.append_message(
            role="assistant", content="partial", execution_state="running"
        )
        updated = memory.update_message(
            message.id, content="partial then cancelled", execution_state="cancelled"
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated.execution_state, "cancelled")  # type: ignore[union-attr]
        self.assertEqual(self.sandbox.open().get_recent_messages()[0].execution_state, "cancelled")

    def test_message_update_ignores_unknown_fields(self) -> None:
        memory = self.sandbox.open()
        memory.create_session(title="Guarded")
        message = memory.append_message(role="user", content="hi")
        memory.update_message(message.id, nonsense="nope")
        self.assertFalse(hasattr(self.sandbox.open().get_recent_messages()[0], "nonsense"))

    def test_unknown_execution_state_is_normalized_not_trusted(self) -> None:
        message = MemoryMessage.from_dict({"role": "assistant", "execution_state": "made_up"})
        self.assertEqual(message.execution_state, "unknown")
        for state in EXECUTION_STATES:
            self.assertEqual(MemoryMessage.from_dict({"execution_state": state}).execution_state, state)

    def test_conversation_search_finds_a_topic_across_sessions(self) -> None:
        memory = self.sandbox.open()
        docker = memory.create_session(title="Docker")
        memory.append_message(role="user", content="How do containers share the kernel?")
        memory.append_message(role="assistant", content="Containers share the host kernel.")
        memory.create_session(title="Godot")
        memory.append_message(role="user", content="Which node type renders sprites?")

        hits = memory.search_conversations("containers")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["conversation_id"], docker.id)
        self.assertIn("containers", hits[0]["excerpt"].casefold())

    def test_a_conversation_is_not_a_task(self) -> None:
        """One conversation holds questions, research, tasks and follow-ups."""

        memory = self.sandbox.open()
        conversation = memory.create_session(title="Mixed")
        memory.append_message(role="user", content="What is Docker?", metadata={"response_kind": "conversation"})
        memory.append_message(role="assistant", content="Docker is ...", metadata={"response_kind": "conversation"})
        memory.append_message(role="user", content="Research the latest Docker changes.")
        memory.append_message(
            role="assistant", content="Here is what changed.", metadata={"response_kind": "research"}
        )
        memory.append_message(role="user", content="Put the important points into Notepad.")
        memory.append_message(
            role="assistant",
            content="Written to Notepad.",
            tool_calls=[{"tool": "applications.write_text", "status": "completed"}],
            metadata={"response_kind": "computer"},
        )

        messages = memory.list_messages(conversation.id)
        self.assertEqual(len(messages), 6)
        kinds = [message.response_kind for message in messages if message.role == "assistant"]
        self.assertEqual(kinds, ["conversation", "research", "computer"])

    def test_list_messages_limit_returns_the_tail(self) -> None:
        memory = self.sandbox.open()
        session = memory.create_session(title="Tail")
        for index in range(5):
            memory.append_message(role="user", content=f"turn {index}")
        self.assertEqual([m.content for m in memory.list_messages(session.id, limit=2)], ["turn 3", "turn 4"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
