"""Tests for the single-worker task queue and authorization safeguards.

These verify that:

* the queue reports which task is running and which are waiting,
* tasks run one at a time in submission order,
* a second approve/deny for the same task is rejected while one is in flight.
"""

from __future__ import annotations

import threading
import time
import unittest

from fastapi import HTTPException

from api import AtlasService
from models import TaskStatus


class _FakeContext:
    def __init__(self) -> None:
        self.task_id = "internal-task"
        self.status = TaskStatus.WAITING_FOR_CONFIRMATION
        self.execution_plan = None
        self.tool_calls: list = []
        self.errors: list = []
        self.warnings: list = []
        self.web_sources: list = []


class _FakeBrain:
    """A brain whose completion timing the test controls."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.last_context = None
        self._gate = gate
        self.approve_calls = 0
        self.deny_calls = 0

    def process(self, request: str) -> str:
        if self._gate is not None:
            self._gate.wait(timeout=5)
        context = _FakeContext()
        context.status = TaskStatus.COMPLETED
        self.last_context = context
        return f"done: {request}"

    def approve_pending(self, task_id: str | None = None) -> str:
        self.approve_calls += 1
        context = _FakeContext()
        context.status = TaskStatus.COMPLETED
        self.last_context = context
        return "approved"

    def deny_pending(self, task_id: str | None = None) -> str:
        self.deny_calls += 1
        context = _FakeContext()
        context.status = TaskStatus.CANCELLED
        self.last_context = context
        return "denied"


class QueueTests(unittest.TestCase):
    def _service(self, brain) -> AtlasService:
        # Bypass real runtime construction (no model, no vector store).
        svc = AtlasService()
        svc.brain = brain
        svc.registry = object()
        return svc

    def test_queue_reports_running_and_pending(self):
        gate = threading.Event()
        brain = _FakeBrain(gate)
        svc = self._service(brain)

        first = svc.submit("first task")
        second = svc.submit("second task")
        # Wait until the worker has picked up the first task.
        deadline = time.time() + 5
        while svc.queue_snapshot()["running"] is None and time.time() < deadline:
            time.sleep(0.02)

        snapshot = svc.queue_snapshot()
        self.assertEqual(snapshot["running"], first.id)
        self.assertIn(second.id, snapshot["pending"])

        gate.set()
        # Let the worker drain both tasks.
        deadline = time.time() + 5
        while svc.queue_snapshot()["running"] is not None and time.time() < deadline:
            time.sleep(0.02)

    def test_tasks_run_one_at_a_time(self):
        active = 0
        max_active = 0
        lock = threading.Lock()

        class SequentialBrain(_FakeBrain):
            def process(self, request: str) -> str:
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)
                with lock:
                    active -= 1
                context = _FakeContext()
                context.status = TaskStatus.COMPLETED
                self.last_context = context
                return f"done: {request}"

        svc = self._service(SequentialBrain())
        for index in range(4):
            svc.submit(f"task {index}")

        deadline = time.time() + 5
        while time.time() < deadline:
            with lock:
                if max_active and active == 0 and svc.queue_snapshot()["running"] is None:
                    break
            time.sleep(0.02)

        self.assertEqual(max_active, 1)


class AuthorizationGuardTests(unittest.TestCase):
    def _awaiting_service(self, brain) -> tuple[AtlasService, str]:
        svc = AtlasService()
        svc.brain = brain
        svc.registry = object()
        record = svc.submit("needs approval")
        # Mark it as awaiting confirmation with a linked task id.
        with svc._lock:
            record.status = TaskStatus.WAITING_FOR_CONFIRMATION.value
            record.task_id = "internal-task"
        return svc, record.id

    def test_approve_runs_once_and_completes(self):
        brain = _FakeBrain()
        svc, record_id = self._awaiting_service(brain)

        record = svc.approve(record_id)

        self.assertEqual(brain.approve_calls, 1)
        self.assertEqual(record.status, TaskStatus.COMPLETED.value)

    def test_second_approve_is_rejected(self):
        brain = _FakeBrain()
        svc, record_id = self._awaiting_service(brain)
        svc.approve(record_id)

        # The task is no longer awaiting confirmation, so a second call must fail.
        with self.assertRaises(HTTPException) as context:
            svc.approve(record_id)

        self.assertEqual(context.exception.status_code, 409)
        self.assertEqual(brain.approve_calls, 1)

    def test_deny_after_approve_is_rejected(self):
        brain = _FakeBrain()
        svc, record_id = self._awaiting_service(brain)
        svc.approve(record_id)

        with self.assertRaises(HTTPException):
            svc.deny(record_id)

    def test_concurrent_approvals_only_admit_one(self):
        gate = threading.Event()
        brain = _FakeBrain(gate)

        # Make approve block so a concurrent second call overlaps.
        original_approve = brain.approve_pending

        def blocking_approve(task_id=None):
            gate.wait(timeout=5)
            return original_approve(task_id)

        brain.approve_pending = blocking_approve  # type: ignore[assignment]
        svc, record_id = self._awaiting_service(brain)

        results: list[object] = []

        def call():
            try:
                results.append(svc.approve(record_id))
            except HTTPException as exc:
                results.append(exc.status_code)

        first = threading.Thread(target=call)
        first.start()
        # Give the first call time to claim the record.
        time.sleep(0.1)
        second = threading.Thread(target=call)
        second.start()

        gate.set()
        first.join(timeout=5)
        second.join(timeout=5)

        # Exactly one success and one rejection.
        statuses = [result for result in results if isinstance(result, int)]
        self.assertEqual(statuses, [409])


if __name__ == "__main__":
    unittest.main()
