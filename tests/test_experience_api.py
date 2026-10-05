"""API-level tests for the feedback endpoint and its experience linkage.

These exercise the real FastAPI routes and the real ``AtlasService`` bookkeeping,
with a temporary experience/task store so nothing touches developer data.
"""

from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api import AtlasService, app
from experience.service import ExperienceService
from experience.store import ExperienceStore
from models import (
    ExecutionContext,
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    StepStatus,
    TaskStatus,
)
from models_task import Task, TaskAction
from task_store import TaskStore


def build_context(*, status: TaskStatus = TaskStatus.COMPLETED, request: str = "Research resumes and put it in Notepad.") -> ExecutionContext:
    task = Task(
        goal="research_and_deliver",
        task_type="multi_step",
        original_prompt=request,
        entities={"application": "Notepad"},
        request_type="hybrid",
        actions=[
            TaskAction(action_id="a1", capability="web.research", parameters={}),
            TaskAction(action_id="a2", capability="applications.write_text", parameters={}, depends_on=["a1"]),
        ],
    )
    context = ExecutionContext(user_input=request)
    context.status = status
    context.final_response = "Done."
    context.execution_plan = ExecutionPlan(
        user_question=request,
        steps=[
            ExecutionStep(id="a1", name="Research", action="invoke_tool", description="",
                          status=StepStatus.COMPLETED, metadata={"tool": "web.research", "parameters": {}}),
            ExecutionStep(id="a2", name="Deliver", action="invoke_tool", description="",
                          status=StepStatus.COMPLETED, metadata={"tool": "applications.write_text", "parameters": {}}),
        ],
        status=PlanStatus.COMPLETED,
    )
    context.tool_calls = [
        {"tool": "web.research", "output": {"content": "research"}, "success": True, "status": "completed", "parameters": {}, "error": None},
        {"tool": "applications.write_text", "output": {"application": "Notepad", "characters": 10, "observed": True},
         "success": True, "status": "completed", "parameters": {}, "error": None},
    ]
    context.verification_results = [
        {"capability": "applications.write_text", "verified": True, "status": "verified", "detail": "read back"}
    ]
    context.metadata["task_object"] = task
    return context


class FeedbackEndpointTests(unittest.TestCase):
    @contextmanager
    def _runtime(self, directory: str, context: ExecutionContext):
        """Build a service over an isolated database and release it on exit.

        The service owns SQLite files inside ``directory``; closing them before
        the temp folder is removed is required on Windows, so the fixture is a
        context manager rather than a bare factory.
        """

        service = ExperienceService(
            store=ExperienceStore(
                experience_path=Path(directory) / "experiences.jsonl",
                feedback_path=Path(directory) / "feedback.jsonl",
            )
        )
        brain = SimpleNamespace(process=lambda request: "Answer", last_context=context)
        runtime = AtlasService(
            brain=brain,
            _task_store=TaskStore(path=Path(directory) / "tasks.json"),
            experience=service,
        )
        self.addCleanup(runtime._executor.shutdown, wait=True)
        try:
            yield runtime
        finally:
            runtime.close()

    def test_feedback_available_is_captured_for_a_meaningful_task(self):
        with tempfile.TemporaryDirectory() as directory:
            context = build_context()
            with self._runtime(directory, context) as runtime:
                from api import TaskRecord

                record = TaskRecord(id="rec", request=context.user_input, status="PENDING", created_at=1, updated_at=1)
                runtime._tasks[record.id] = record
                runtime._run(record.id)

                self.assertTrue(record.feedback_available)
                self.assertTrue(record.experience_id)

    def test_plain_conversation_has_no_feedback_available(self):
        with tempfile.TemporaryDirectory() as directory:
            context = ExecutionContext(user_input="What is Python?")
            context.status = TaskStatus.COMPLETED
            context.final_response = "A programming language."
            with self._runtime(directory, context) as runtime:
                from api import TaskRecord

                record = TaskRecord(id="rec", request="What is Python?", status="PENDING", created_at=1, updated_at=1)
                runtime._tasks[record.id] = record
                runtime._run(record.id)

                self.assertFalse(record.feedback_available)
                self.assertEqual(record.feedback_outcome, "unknown")

    def test_feedback_endpoint_stores_success_and_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            context = build_context()
            with self._runtime(directory, context) as runtime:
                from api import TaskRecord

                record = TaskRecord(id="rec", request=context.user_input, status="PENDING", created_at=1, updated_at=1)
                runtime._tasks[record.id] = record
                runtime._run(record.id)

                with patch("api.service", runtime):
                    response = TestClient(app).post(
                    "/api/tasks/rec/feedback", json={"outcome": "success"}
                    )
                    self.assertEqual(response.status_code, 200)
                    payload = response.json()
                    self.assertEqual(payload["feedback_outcome"], "success")
                    self.assertTrue(payload["feedback_available"])
                    # Persisted to the task snapshot too.
                    self.assertEqual(runtime._tasks["rec"].feedback_outcome, "success")

    def test_feedback_endpoint_stores_failure_category_and_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            context = build_context()
            with self._runtime(directory, context) as runtime:
                from api import TaskRecord

                record = TaskRecord(id="rec", request=context.user_input, status="PENDING", created_at=1, updated_at=1)
                runtime._tasks[record.id] = record
                runtime._run(record.id)

                with patch("api.service", runtime):
                    response = TestClient(app).post(
                    "/api/tasks/rec/feedback",
                    json={
                    "outcome": "failure",
                    "failure_category": "did_not_follow_instruction",
                    "correction": "I asked you to put it in Notepad.",
                    },
                    )
                    self.assertEqual(response.status_code, 200)
                    payload = response.json()
                    self.assertEqual(payload["feedback_outcome"], "failure")
                    self.assertEqual(payload["feedback_category"], "did_not_follow_instruction")
                    self.assertEqual(payload["feedback_category_label"], "Did not follow instruction")
                    self.assertEqual(payload["feedback_correction"], "I asked you to put it in Notepad.")

    def test_feedback_endpoint_rejects_unknown_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            context = build_context()
            with self._runtime(directory, context) as runtime:
                from api import TaskRecord

                record = TaskRecord(id="rec", request=context.user_input, status="COMPLETED", created_at=1, updated_at=1)
                runtime._tasks[record.id] = record

                with patch("api.service", runtime):
                    response = TestClient(app).post("/api/tasks/rec/feedback", json={"outcome": "maybe"})
                    self.assertEqual(response.status_code, 422)

    def test_feedback_endpoint_rejects_unknown_category(self):
        with tempfile.TemporaryDirectory() as directory:
            context = build_context()
            with self._runtime(directory, context) as runtime:
                from api import TaskRecord

                record = TaskRecord(id="rec", request=context.user_input, status="COMPLETED", created_at=1, updated_at=1)
                runtime._tasks[record.id] = record

                with patch("api.service", runtime):
                    response = TestClient(app).post(
                    "/api/tasks/rec/feedback",
                    json={"outcome": "failure", "failure_category": "invented_category"},
                    )
                    self.assertEqual(response.status_code, 422)

    def test_feedback_endpoint_404s_for_unknown_task(self):
        with tempfile.TemporaryDirectory() as directory:
            with self._runtime(directory, build_context()) as runtime:
                with patch("api.service", runtime):
                    response = TestClient(app).post("/api/tasks/missing/feedback", json={"outcome": "success"})
                    self.assertEqual(response.status_code, 404)

    def test_feedback_endpoint_conflicts_while_task_is_running(self):
        with tempfile.TemporaryDirectory() as directory:
            with self._runtime(directory, build_context()) as runtime:
                from api import TaskRecord

                record = TaskRecord(id="rec", request="x", status="RUNNING", created_at=1, updated_at=1)
                runtime._tasks[record.id] = record
                with patch("api.service", runtime):
                    response = TestClient(app).post("/api/tasks/rec/feedback", json={"outcome": "success"})
                    self.assertEqual(response.status_code, 409)

    def test_experience_status_endpoint_is_bounded_and_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            context = build_context()
            with self._runtime(directory, context) as runtime:
                from api import TaskRecord

                record = TaskRecord(id="rec", request=context.user_input, status="PENDING", created_at=1, updated_at=1)
                runtime._tasks[record.id] = record
                runtime._run(record.id)

                with patch("api.service", runtime):
                    response = TestClient(app).get("/api/experience")
                    self.assertEqual(response.status_code, 200)
                    payload = response.json()
                    self.assertTrue(payload["enabled"])
                    self.assertEqual(payload["counts"]["total"], 1)
                    # The task text must not leak into the status surface.
                    self.assertNotIn(context.user_input, response.text)


if __name__ == "__main__":
    unittest.main()
