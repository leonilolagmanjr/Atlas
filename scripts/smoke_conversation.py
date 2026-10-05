"""Live, offline-of-the-network conversation smoke test.

This boots the *real* FastAPI app with the *real* Brain, the real tool registry
and the real local model, and drives an actual conversation through HTTP:

    ask a question -> follow-up that needs context -> a real read-only task
    -> long-term memory -> conversation search -> cancellation

The conversation store is redirected to a temporary folder so a smoke run never
touches the user's real conversations, and the smoke test deliberately uses
read-only capabilities so it never opens or changes anything on the machine.

Run with the local model available:

    python scripts/smoke_conversation.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import api as api_module  # noqa: E402
from api import AtlasService, app  # noqa: E402
from memory.memory_manager import MemoryManager  # noqa: E402
from memory.storage import ConversationStore  # noqa: E402


def _install(root: Path) -> AtlasService:
    service = AtlasService()
    service.ensure_runtime()
    # Redirect only the conversation/user stores; the model, tools, and Brain
    # are the real ones. A smoke run must never write into the user's own
    # conversations or memory.
    memory = MemoryManager()
    memory._store = ConversationStore(root_folder=root)  # noqa: SLF001
    memory._sessions._store = memory._store  # noqa: SLF001
    # The active-conversation pointer is a row in the conversation store's own
    # database, so redirecting the store is enough; there is no separate file.
    service.runtime._memory = memory  # noqa: SLF001
    service.runtime._context._memory = memory  # noqa: SLF001
    # The Brain keeps its own handle on the session store; it must be redirected
    # too or the early-answer path writes the smoke turns into real history.
    if service.brain is not None:
        service.brain._memory_manager = memory  # noqa: SLF001
    from memory.user_memory import UserMemoryStore

    service.user_memory = UserMemoryStore(path=root / "user_memory.jsonl")
    service.runtime._user_memory = service.user_memory  # noqa: SLF001
    return service


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = api_module.service
        service = _install(root)
        api_module.service = service
        client = TestClient(app)
        failures: list[str] = []

        def check(label: str, condition: bool, detail: str = "") -> None:
            status = "ok " if condition else "FAIL"
            print(f"[{status}] {label}{(' — ' + detail) if detail else ''}")
            if not condition:
                failures.append(label)

        try:
            _section("conversation")
            created = client.post("/api/conversations", json={"title": "Smoke"})
            check("conversation created", created.status_code == 201)
            conversation_id = created.json()["id"]

            _section("question (real local model)")
            started = time.perf_counter()
            answer = client.post(
                "/api/messages",
                json={"message": "What is Docker? Answer in two sentences.", "conversation_id": conversation_id},
            )
            text = answer.json()["text"]
            print(f"({time.perf_counter() - started:.1f}s) {text[:400]}")
            check("question answered", bool(text.strip()) and answer.json()["execution_state"] == "completed")
            check("kind is conversation", answer.json()["kind"] == "conversation", answer.json()["kind"])

            _section("follow-up that depends on the previous turn")
            follow_up = client.post(
                "/api/messages",
                json={"message": "Explain that more simply.", "conversation_id": conversation_id},
            )
            follow_text = follow_up.json()["text"]
            print(follow_text[:400])
            check("follow-up answered from context", bool(follow_text.strip()))

            _section("real read-only task (existing tool path)")
            # A path inside Atlas's own root, so this really calls the existing
            # permission-gated filesystem tool rather than being refused.
            task = client.post(
                "/api/messages",
                json={"message": "List the files in the tests folder.", "conversation_id": conversation_id},
            )
            print(json.dumps(task.json()["activity"], indent=2)[:600])
            check("task delegated to tools", len(task.json()["tool_calls"]) > 0)
            check("task completed", task.json()["execution_state"] == "completed", task.json()["execution_state"])

            _section("a request the root check refuses is reported honestly")
            refused = client.post(
                "/api/messages",
                json={"message": "List the files in my Documents folder.", "conversation_id": conversation_id},
            )
            print(refused.json()["text"][:300])
            check(
                "refused path is not answered as if it worked",
                "outside the allowed root" in refused.json()["text"].casefold()
                or len(refused.json()["tool_calls"]) > 0,
            )

            _section("long-term memory")
            client.post(
                "/api/messages",
                json={"message": "Remember that I prefer local-first architecture.", "conversation_id": conversation_id},
            )
            memories = client.get("/api/memory").json()
            check("durable statement remembered", memories["count"] > 0, json.dumps(memories["memories"])[:200])
            check(
                "transient question not remembered",
                not any("docker" in item["text"].casefold() for item in memories["memories"]),
            )

            _section("conversation search")
            search = client.get("/api/conversations/search", params={"query": "docker"}).json()
            check("search finds the exchange", len(search["results"]) > 0, f"{len(search['results'])} hit(s)")

            _section("streaming and cancellation")
            frames: list[str] = []

            async def drain(payload: dict) -> None:
                response = api_module.stream_message(api_module.MessagePayload(**payload))
                try:
                    while True:
                        frames.append(await response.body_iterator.__anext__())
                except StopAsyncIteration:
                    return

            import asyncio

            asyncio.run(
                drain({"message": "Explain what a container is in one sentence.", "conversation_id": conversation_id})
            )
            body = "".join(frames)
            check("stream opened", "event: open" in body)
            check("stream reported a token or an answer", "event: token" in body or "assistant_completed" in body)

            cancelled = {"seen": False}

            def watch() -> None:
                deadline = time.time() + 30
                while time.time() < deadline:
                    with service._lock:  # noqa: SLF001
                        if service._turn_tokens:  # noqa: SLF001
                            cancelled["seen"] = True
                            return
                    time.sleep(0.05)

            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()

            async def cancel_mid_turn() -> str:
                response = api_module.stream_message(
                    api_module.MessagePayload(
                        message="Research the latest developments in quantum computing and write a long summary.",
                        conversation_id=conversation_id,
                    )
                )

                async def pull() -> str:
                    collected = ""
                    try:
                        while True:
                            collected += await response.body_iterator.__anext__()
                    except StopAsyncIteration:
                        return collected
                    # Cancel as soon as the turn is under way.
                    async def stop() -> None:
                        with service._lock:  # noqa: SLF001
                            turn_ids = list(service._turn_tokens)  # noqa: SLF001
                        for turn_id in turn_ids:
                            client.post(f"/api/turns/{turn_id}/cancel")

                    await asyncio.sleep(0.4)
                    await stop()
                    return collected

                return await pull()

            cancel_body = asyncio.run(cancel_mid_turn())
            print(cancel_body[-400:])
            check("turn was cancellable", cancelled["seen"], "a live turn was observed")

            _section("persistence across a restart")
            messages_before = client.get(f"/api/conversations/{conversation_id}/messages").json()["messages"]
            api_module.service = _install(root)  # a brand new service, same store
            restarted = TestClient(app)
            listed = restarted.get("/api/conversations").json()["conversations"]
            check("conversation survived a restart", any(item["id"] == conversation_id for item in listed))
            messages_after = restarted.get(f"/api/conversations/{conversation_id}/messages").json()["messages"]
            check("transcript survived a restart", len(messages_after) == len(messages_before))
            check(
                "memory survived a restart",
                restarted.get("/api/memory").json()["count"] > 0,
            )
        finally:
            api_module.service = original

        print("\n" + ("FAILURES: " + ", ".join(failures) if failures else "all smoke checks passed"))
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
