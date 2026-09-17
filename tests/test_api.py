import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api import AtlasService, TaskRecord, _reasoning_snapshot, app
from models import ExecutionContext, TaskStatus
from task_store import TaskStore


class ApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_reports_local_runtime(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["local"])
        self.assertEqual(response.json()["status"], "online")

    def test_system_reports_real_machine_data(self):
        response = self.client.get("/api/system")

        self.assertEqual(response.status_code, 200)
        self.assertIn("system", response.json())
        self.assertIn("disk", response.json()["system"])

    def test_tool_knowledge_exposes_powerShell_records(self):
        response = self.client.get("/api/tool-knowledge")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(item["name"] == "Get-Process" for item in response.json()["tools"]))

    def test_tool_discovery_returns_candidates_without_execution(self):
        response = self.client.get("/api/tool-discovery", params={"query": "inspect memory usage"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["candidates"][0]["tool"], "Get-Process")


class ReasoningObservabilityTests(unittest.TestCase):
    def test_task_endpoint_serializes_early_answer_contract(self):
        context = ExecutionContext(user_input="Inspect sources")
        context.status = TaskStatus.COMPLETED
        context.metadata = {
            "reasoning": [{"stage": "EVALUATING", "detail": "evidence assessed", "iteration": 1, "metadata": {"history": "PRIVATE_HISTORY"}}],
            "reasoning_answer": {
                "mode": "grounded_answer",
                "provenance": ["knowledge"],
                "citations": [r"C:\private\report.pdf", "https://user:password@example.com/report?token=SECRET#private"],
                "metadata": {"decision": {"goal": "PRIVATE_PROMPT"}, "evidence": {"count": 1, "sources": ["knowledge"], "items": [{"content": "PRIVATE_CONTENT"}]}},
            },
        }
        with TemporaryDirectory() as directory:
            store = TaskStore(path=Path(directory) / "tasks.json")
            runtime = AtlasService(brain=SimpleNamespace(process=lambda request: "Answer", last_context=context), _task_store=store)
            self.addCleanup(runtime._executor.shutdown, wait=True)
            record = TaskRecord(id="record", request="Inspect sources", status="PENDING", created_at=1, updated_at=1)
            runtime._tasks[record.id] = record
            with self.assertLogs("api", level="INFO") as logs:
                runtime._run(record.id)
            with patch("api.service", runtime):
                response = TestClient(app).get("/api/tasks/record")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "COMPLETED")
            self.assertIsNone(payload["plan"])
            self.assertEqual(payload["reasoning"], [{"stage": "EVALUATING", "detail": "evidence assessed", "iteration": 1}])
            self.assertEqual(payload["response_mode"], "grounded_answer")
            self.assertEqual(payload["provenance"], ["knowledge"])
            self.assertEqual(payload["citations"], ["report.pdf", "https://example.com/report"])
            self.assertEqual(payload["evidence"], {"count": 1, "sources": ["knowledge"]})
            serialized = json.dumps(payload) + json.dumps(store.load()) + " ".join(logs.output)
            for private in ("PRIVATE_HISTORY", "PRIVATE_PROMPT", "PRIVATE_CONTENT", "password", "SECRET"):
                self.assertNotIn(private, serialized)

    def test_delegated_action_and_approval_preserve_trace(self):
        context = ExecutionContext(user_input="Open application")
        context.status = TaskStatus.WAITING_FOR_CONFIRMATION
        context.metadata["task"] = {
            "response_mode": "action_report",
            "original_prompt": "PRIVATE_PROMPT",
            "context": {"reasoning": [{"stage": "PLANNING", "detail": "delegated to task pipeline", "iteration": 1}]},
        }
        brain = SimpleNamespace(process=lambda request: "Approval needed", last_context=context)
        store = Mock()
        store.load.return_value = {}
        runtime = AtlasService(brain=brain, _task_store=store)
        self.addCleanup(runtime._executor.shutdown, wait=True)
        record = TaskRecord(id="action", request="Open application", status="PENDING", created_at=1, updated_at=1)
        runtime._tasks[record.id] = record
        runtime._run(record.id)
        self.assertEqual(record.response_mode, "action_report")
        self.assertEqual(record.reasoning[0]["stage"], "PLANNING")
        self.assertEqual(record.provenance, [])
        def approve(task_id):
            self.assertEqual(task_id, context.task_id)
            context.status = TaskStatus.COMPLETED
            return "Done"
        brain.approve_pending = approve
        resolved = runtime.approve(record.id)
        self.assertEqual(resolved.status, "COMPLETED")
        self.assertEqual(resolved.reasoning, record.reasoning)
        self.assertEqual(resolved.response_mode, "action_report")

    def test_unrelated_approval_context_cannot_replace_reasoning(self):
        store = Mock()
        store.load.return_value = {}
        runtime = AtlasService(_task_store=store)
        self.addCleanup(runtime._executor.shutdown, wait=True)
        record = TaskRecord(id="action", task_id="expected", request="Open application", status="RUNNING", created_at=1, updated_at=1, reasoning=[{"stage": "PLANNING", "detail": "delegated", "iteration": 1}])
        other = ExecutionContext(user_input="other")
        other.metadata["reasoning"] = [{"stage": "ANSWERING", "detail": "other task", "iteration": 1}]
        runtime._capture_reasoning(record, other)
        self.assertEqual(record.reasoning[0]["detail"], "delegated")

    def test_reasoning_fields_are_bounded_and_drop_nested_data(self):
        record = TaskRecord(
            id="bounded", request="test", status="COMPLETED", created_at=1, updated_at=1,
            reasoning=[{"stage": "RETRIEVING", "detail": "x" * 1000, "iteration": 1000000, "content": "PRIVATE", "metadata": {"prompt": "PRIVATE"}}] * 100,
            provenance=["web", "web", "invalid", {"content": "PRIVATE"}],
            citations=[f"https://example.com/{index}?token=PRIVATE" for index in range(50)],
            response_mode="unrecognized",
            evidence={"count": 999999999, "sources": ["web", "invalid"], "items": [{"content": "PRIVATE"}]},
        )
        self.assertEqual(len(record.reasoning), 64)
        self.assertEqual(len(record.reasoning[0]["detail"]), 240)
        self.assertEqual(record.reasoning[0]["iteration"], 100)
        self.assertEqual(set(record.reasoning[0]), {"stage", "detail", "iteration"})
        self.assertEqual(len(record.citations), 16)
        self.assertEqual(record.provenance, ["web"])
        self.assertIsNone(record.response_mode)
        self.assertEqual(record.evidence, {"count": 1000000, "sources": ["web"]})
        self.assertNotIn("PRIVATE", record.model_dump_json())

    def test_malformed_and_legacy_metadata_remain_serializable(self):
        for metadata in (None, [], {"reasoning": "PRIVATE", "reasoning_answer": ["PRIVATE"]}, {"task": {"context": ["PRIVATE"]}}):
            with self.subTest(metadata=metadata):
                snapshot = _reasoning_snapshot(SimpleNamespace(metadata=metadata))
                self.assertEqual(snapshot, {"reasoning": [], "provenance": [], "citations": [], "response_mode": None, "evidence": None})
        record = TaskRecord(id="legacy", request="test", status="COMPLETED", created_at=1, updated_at=1)
        self.assertEqual(record.reasoning, [])
        malformed = TaskRecord(**{**record.model_dump(), "reasoning": [None, {"stage": []}, {"stage": "UNKNOWN"}, {"stage": "ANSWERING", "detail": {}, "iteration": True}], "citations": ["javascript:alert(1)", "data:text/html,PRIVATE", "https://[invalid", "line\ncontent"]})
        self.assertEqual(malformed.reasoning, [{"stage": "ANSWERING", "detail": "", "iteration": 0}])
        self.assertEqual(malformed.citations, [])

    def test_reloaded_records_sanitize_observability(self):
        store = Mock()
        store.load.return_value = {"persisted": {"id": "persisted", "request": "test", "status": "COMPLETED", "created_at": 1, "updated_at": 1, "evidence": {"count": 1, "sources": ["files"], "items": [{"content": "PRIVATE"}]}, "reasoning": [{"stage": "ANSWERING", "detail": "file_lookup", "metadata": {"history": "PRIVATE"}}]}}
        runtime = AtlasService(_task_store=store)
        self.addCleanup(runtime._executor.shutdown, wait=True)
        payload = runtime._tasks["persisted"].model_dump_json()
        self.assertNotIn("PRIVATE", payload)
        self.assertNotIn("items", payload)


if __name__ == "__main__":
    unittest.main()
