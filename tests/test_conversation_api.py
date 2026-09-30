"""Milestones 6-7: the conversational API, streaming, cancellation, memory API.

These tests drive the real FastAPI app with a *fake* Brain and a *fake* memory
root, so the HTTP contract, the persisted turn shape and the cancellation
semantics are exercised directly instead of being asserted about.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

import api as api_module
from api import AtlasService, app
from memory.conversation_runtime import CancellationToken
from memory.memory_manager import MemoryManager
from memory.storage import ConversationStore
from memory.user_memory import UserMemoryStore


class _NullIndex:
    available = False

    def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def index_message(self, **_kwargs: Any) -> bool:
        return False

    def forget_conversation(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _EmbedderStub:
    def index(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def similarities(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
        return {}

    def delete(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _FakeContext:
    def __init__(self, tools: list[tuple[str, str]] | None = None, status: str = "COMPLETED") -> None:
        from models import TaskStatus

        self.task_id = "task-1"
        self.status = TaskStatus[status]
        self.tool_calls = [
            {"tool": name, "status": state, "success": True, "output": {}, "parameters": {}}
            for name, state in (tools or [])
        ]
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.web_sources: list[str] = []
        self.execution_plan = None
        self.metadata: dict[str, Any] = {"reasoning_answer": {"mode": "direct_answer"}}


class _FakeBrain:
    """Delegation stand-in: what ran is what the API must report."""

    def __init__(self, tools: list[tuple[str, str]] | None = None) -> None:
        self.requests: list[str] = []
        self._tools = tools or []
        self.last_context = None

    def process(self, user_input: str) -> str:
        self.requests.append(user_input)
        self.last_context = _FakeContext(tools=self._tools)
        return f"done: {user_input}"

    def approve_pending(self, *_args: Any, **_kwargs: Any) -> str:
        return "approved"

    def deny_pending(self, *_args: Any, **_kwargs: Any) -> str:
        return "denied"


class _StreamingAsk:
    """A real streaming boundary over fake tokens (no network, no model).

    It is *only* a streamer: it is passed to the runtime's ``stream`` argument,
    never used as the non-streaming askable, because the two contracts differ.
    """

    def __init__(self, pieces: list[str], *, cancel_after: CancellationToken | None = None) -> None:
        self.pieces = pieces
        self._cancel_after = cancel_after

    def __call__(self, *, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
        raise AssertionError("the streaming boundary must not be used as the askable")

    def stream(self, *, system_prompt, user_prompt, on_token, should_stop):  # noqa: ARG002
        produced = ""
        for index, piece in enumerate(self.pieces):
            if should_stop is not None and should_stop():
                break
            on_token(piece)
            produced += piece
            if self._cancel_after is not None and index == 0:
                self._cancel_after.cancel()
        return produced


class _ServiceHarness:
    """Wire an AtlasService onto a temporary memory root and a fake Brain."""

    def __init__(self, *, tools: list[tuple[str, str]] | None = None) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.service = AtlasService()
        self.brain = _FakeBrain(tools=tools)

    def _memory(self) -> MemoryManager:
        manager = MemoryManager(ask=None, index=_NullIndex())
        manager._store = ConversationStore(root_folder=self.root)  # noqa: SLF001
        manager._sessions._store = manager._store  # noqa: SLF001
        manager._sessions._active_path = self.root / "active_session.json"  # noqa: SLF001
        return manager

    def install(self) -> AtlasService:
        from memory.conversation_runtime import ConversationRuntime

        service = self.service
        service.brain = self.brain  # type: ignore[assignment]
        service.user_memory = UserMemoryStore(
            path=self.root / "user_memory.jsonl", embedder=_EmbedderStub()
        )
        runtime = ConversationRuntime(
            memory_manager=self._memory(),
            brain=self.brain,
            user_memory=service.user_memory,
            ask=None,  # deterministic: no live model call from a unit test
            index=_NullIndex(),
        )
        service.runtime = runtime
        # ensure_runtime() is a no-op once brain/runtime exist, so no vector
        # store, embedder, or knowledge indexing is touched by these tests.
        return service

    def cleanup(self) -> None:
        self._temporary.cleanup()


class ConversationalApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = _ServiceHarness()
        self.service = self.harness.install()
        self._original = api_module.service
        api_module.service = self.service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        api_module.service = self._original
        self.harness.cleanup()

    # -- conversations -------------------------------------------------------

    def test_create_list_rename_delete_conversation(self) -> None:
        created = self.client.post("/api/conversations", json={"title": "First"})
        self.assertEqual(created.status_code, 201)
        conversation_id = created.json()["id"]

        listed = self.client.get("/api/conversations").json()["conversations"]
        self.assertEqual([item["id"] for item in listed], [conversation_id])

        renamed = self.client.patch(f"/api/conversations/{conversation_id}", json={"title": "Renamed"})
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["title"], "Renamed")

        self.assertEqual(self.client.delete(f"/api/conversations/{conversation_id}").status_code, 204)
        self.assertEqual(self.client.get("/api/conversations").json()["conversations"], [])

    def test_post_message_answers_conversationally_without_touching_the_brain(self) -> None:
        conversation = self.client.post("/api/conversations", json={}).json()
        response = self.client.post(
            "/api/messages", json={"message": "What is RAM?", "conversation_id": conversation["id"]}
        )
        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["kind"], "conversation")
        self.assertEqual(self.harness.brain.requests, [])
        self.assertEqual(payload["conversation_id"], conversation["id"])

        messages = self.client.get(
            f"/api/conversations/{conversation['id']}/messages"
        ).json()["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["execution_state"], "completed")

    def test_post_message_delegates_an_action(self) -> None:
        response = self.client.post("/api/messages", json={"message": "Open Chrome."})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.harness.brain.requests, ["Open Chrome."])
        self.assertEqual(response.json()["kind"], "computer")

    def test_messages_survive_a_service_restart(self) -> None:
        conversation = self.client.post("/api/conversations", json={}).json()
        self.client.post(
            "/api/messages",
            json={"message": "Remember that I prefer local-first architecture.", "conversation_id": conversation["id"]},
        )
        # A new runtime over the same store == a restart.
        restarted = self.harness._memory()  # noqa: SLF001
        self.assertTrue(restarted.list_messages(conversation["id"]))

    def test_conversation_search_endpoint(self) -> None:
        # Two conversations, so the search must attribute the hit correctly
        # rather than returning whatever session happens to be first.
        other = self.client.post("/api/conversations", json={}).json()
        self.client.post(
            "/api/messages",
            json={"message": "Tell me about Godot nodes.", "conversation_id": other["id"]},
        )
        target = self.client.post("/api/conversations", json={}).json()
        self.client.post(
            "/api/messages",
            json={"message": "We use SQLite for local storage.", "conversation_id": target["id"]},
        )

        results = self.client.get("/api/conversations/search", params={"query": "SQLite"}).json()["results"]
        self.assertTrue(results)
        self.assertIn("SQLite", results[0]["excerpt"])
        self.assertEqual(results[0]["conversation_id"], target["id"])
        self.assertFalse(any("Godot" in hit["excerpt"] for hit in results))

    def test_empty_search_query_returns_nothing(self) -> None:
        self.assertEqual(self.client.get("/api/conversations/search", params={"query": " "}).json()["results"], [])

    # -- streaming -----------------------------------------------------------

    def test_stream_endpoint_emits_real_token_events_and_a_terminal_event(self) -> None:
        from memory.conversation_runtime import EVENT_TOKEN

        streaming = _StreamingAsk(["Docker ", "packages ", "software."])
        with self._patched_send(streaming):
            with self.client.stream("POST", "/api/messages/stream", json={"message": "What is Docker?"}) as response:
                body = "".join(response.iter_text())

        self.assertIn("event: open", body)
        self.assertIn(f"event: {EVENT_TOKEN}", body)
        self.assertIn("event: assistant_completed", body)
        # Only lines emitted with the token event carry token data; the terminal
        # event carries the whole turn, so the event name is matched explicitly
        # rather than inferred from the presence of a "text" field.
        token_payloads = []
        current_event = ""
        for line in body.splitlines():
            if line.startswith("event: "):
                current_event = line[len("event: "):]
            elif line.startswith("data: ") and current_event == EVENT_TOKEN:
                token_payloads.append(json.loads(line[len("data: "):]))
        self.assertEqual([payload["text"] for payload in token_payloads], ["Docker ", "packages ", "software."])

    def test_stream_endpoint_forwards_events_before_the_turn_ends(self) -> None:
        """The stream must be incremental, not a batch replayed at the end.

        A turn whose answer arrives in two parts with a pause between them must
        let the client read the first token *while the turn is still running*.
        The in-process test client buffers a whole response, so the response's
        own iterator is driven directly here.
        """

        import asyncio
        import threading

        released = threading.Event()
        first_seen = threading.Event()
        parts = ["First part. ", "Second part."]

        class SlowStream(_StreamingAsk):
            def stream(self, *, system_prompt, user_prompt, on_token, should_stop):  # noqa: ARG002
                produced = ""
                for index, piece in enumerate(parts):
                    on_token(piece)
                    produced += piece
                    if index == 0:
                        # The turn is still running here; the first token must
                        # already be readable by the client.
                        first_seen.set()
                        released.wait(timeout=10)
                    if should_stop is not None and should_stop():
                        break
                return produced

        def send_message(**kwargs: Any) -> dict[str, Any]:
            kwargs["stream"] = SlowStream(parts)
            return original_send(**kwargs)

        original_send = AtlasService.send_message.__get__(self.service, AtlasService)
        frames: list[str] = []

        async def drain(iterator) -> None:
            opening = await iterator.__anext__()
            self.assertIn("event: open", opening)
            frame = ""
            for _ in range(6):
                frame += await iterator.__anext__()
                if "event: token" in frame:
                    break
            frames.append(frame)
            # The turn has not finished, yet the token was already delivered.
            self.assertFalse(released.is_set())
            released.set()
            while True:
                try:
                    frames.append(await iterator.__anext__())
                except StopAsyncIteration:
                    return

        with patch.object(self.service, "send_message", side_effect=send_message):
            response = api_module.stream_message(api_module.MessagePayload(message="Tell me slowly."))
            asyncio.run(drain(response.body_iterator))
            released.set()

        body = "".join(frames)
        self.assertIn("event: token", body)
        self.assertIn("event: assistant_completed", body)
        self.assertTrue(first_seen.is_set())

    def _patched_send(self, streaming: Any, token: CancellationToken | None = None):
        """Patch the *instance's* send_message, capturing the real one first.

        Patching the class leaves the captured reference pointing at whatever a
        previous test left behind; binding to the instance keeps the original
        real implementation and the injected streamer in one place.
        """

        original = AtlasService.send_message.__get__(self.service, AtlasService)

        def send_message(**kwargs: Any) -> dict[str, Any]:
            kwargs["stream"] = streaming
            if token is not None:
                kwargs["token"] = token
            return original(**kwargs)

        return patch.object(self.service, "send_message", side_effect=send_message)

    def test_stream_endpoint_reports_a_cancelled_turn(self) -> None:
        token = CancellationToken()
        streaming = _StreamingAsk(["Partial ", "answer"], cancel_after=token)
        with self._patched_send(streaming, token):
            with self.client.stream("POST", "/api/messages/stream", json={"message": "Tell me a long story."}) as response:
                body = "".join(response.iter_text())

        self.assertIn("event: cancelled", body)
        self.assertIn('"execution_state": "cancelled"', body)

    def test_streaming_can_be_disabled_honestly(self) -> None:
        with patch.object(api_module, "ENABLE_CONVERSATION_STREAMING", False):
            response = self.client.post("/api/messages/stream", json={"message": "hi"})
        self.assertEqual(response.status_code, 503)

    # -- cancellation ---------------------------------------------------------

    def test_cancel_unknown_turn_is_a_404(self) -> None:
        self.assertEqual(self.client.post("/api/turns/nope/cancel").status_code, 404)

    def test_cancel_running_turn_returns_true(self) -> None:
        token = CancellationToken()
        self.service._turn_tokens["turn-1"] = token  # noqa: SLF001 - simulating an in-flight turn
        response = self.client.post("/api/turns/turn-1/cancel")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(token.is_cancelled())

    # -- memory API -----------------------------------------------------------

    def test_memory_create_read_update_delete_clear(self) -> None:
        created = self.client.post(
            "/api/memory", json={"text": "User prefers tabs", "kind": "preference"}
        )
        self.assertEqual(created.status_code, 201)
        memory_id = created.json()["id"]

        listed = self.client.get("/api/memory").json()
        self.assertEqual(listed["count"], 1)
        self.assertTrue(listed["enabled"])

        updated = self.client.patch(f"/api/memory/{memory_id}", json={"text": "User prefers spaces"})
        self.assertEqual(updated.json()["text"], "User prefers spaces")

        self.assertEqual(self.client.delete(f"/api/memory/{memory_id}").status_code, 204)
        self.assertEqual(self.client.get("/api/memory").json()["count"], 0)

        self.client.post("/api/memory", json={"text": "User works on Atlas", "kind": "project"})
        cleared = self.client.post("/api/memory/clear").json()
        self.assertEqual(cleared["removed"], 1)

    def test_memory_rejects_an_unrecognized_kind(self) -> None:
        response = self.client.post("/api/memory", json={"text": "x", "kind": "made_up"})
        self.assertEqual(response.status_code, 422)

    def test_memory_search_endpoint(self) -> None:
        self.client.post("/api/memory", json={"text": "User works on the Atlas project", "kind": "project"})
        self.client.post("/api/memory", json={"text": "User prefers Vim", "kind": "preference"})
        results = self.client.get("/api/memory", params={"query": "Atlas"}).json()["memories"]
        self.assertEqual(len(results), 1)

    def test_memory_can_be_disabled_and_re_enabled(self) -> None:
        disabled = self.client.post("/api/memory/enabled", json={"enabled": False}).json()
        self.assertFalse(disabled["enabled"])
        self.assertEqual(
            self.client.post("/api/memory", json={"text": "should not store"}).status_code, 400
        )
        re_enabled = self.client.post("/api/memory/enabled", json={"enabled": True}).json()
        self.assertTrue(re_enabled["enabled"])

    def test_unknown_memory_update_is_a_404(self) -> None:
        self.assertEqual(self.client.patch("/api/memory/nope", json={"text": "x"}).status_code, 404)

    def test_a_user_message_writes_durable_memory_through_the_turn_path(self) -> None:
        self.client.post("/api/messages", json={"message": "Remember that I prefer local-first architecture."})
        memories = self.client.get("/api/memory").json()["memories"]
        self.assertTrue(memories)
        self.assertIn("local-first", memories[0]["text"])

    def test_ordinary_conversation_creates_no_memory(self) -> None:
        self.client.post("/api/messages", json={"message": "What is Docker?"})
        self.assertEqual(self.client.get("/api/memory").json()["count"], 0)


class ExistingTaskApiTests(unittest.TestCase):
    """The pre-existing task API must keep working unchanged."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self._original = api_module.service
        self.service = AtlasService()
        self.service.brain = _FakeBrain()  # type: ignore[assignment]
        # A fresh task store so this test observes only what it creates. The
        # persisted store is real production data and is not this test's subject.
        from task_store import TaskStore

        self.service._task_store = TaskStore(path=Path(self._temporary.name) / "tasks.json")  # noqa: SLF001
        self.service._tasks.clear()  # noqa: SLF001
        api_module.service = self.service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        api_module.service = self._original
        self._temporary.cleanup()

    def test_health_and_tools_and_tasks_still_work(self) -> None:
        self.assertEqual(self.client.get("/api/health").json()["status"], "online")
        self.assertEqual(self.client.post("/api/tasks", json={"request": "do a thing"}).status_code, 202)
        tasks = self.client.get("/api/tasks").json()["tasks"]
        self.assertEqual([task["request"] for task in tasks], ["do a thing"])

    def test_queue_endpoint_still_reports_in_flight_state(self) -> None:
        self.assertIn("running", self.client.get("/api/queue").json())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
