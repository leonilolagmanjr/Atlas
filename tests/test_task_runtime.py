import tempfile
import unittest
from pathlib import Path

from api import AtlasService, TaskRecord
from config import TASK_STORE_FILE
from models import TaskStatus
from task_store import TaskStore


class TaskRuntimeTests(unittest.TestCase):
    def test_service_restores_completed_task_history(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tasks.json"
            record = TaskRecord(
                id="api-task-1",
                request="Inspect the system",
                status=TaskStatus.COMPLETED.value,
                response="done",
                created_at=1.0,
                updated_at=2.0,
            )
            store = TaskStore(path=path)
            store.save({record.id: record})

            service = AtlasService(_task_store=store)

            restored = service._get(record.id)
            self.assertEqual(restored.response, "done")
            self.assertEqual(restored.status, TaskStatus.COMPLETED.value)

    def test_service_marks_interrupted_task_as_failed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tasks.json"
            record = TaskRecord(
                id="api-task-2",
                request="Run a task",
                status=TaskStatus.WAITING_FOR_CONFIRMATION.value,
                created_at=1.0,
                updated_at=2.0,
            )
            store = TaskStore(path=path)
            store.save({record.id: record})

            service = AtlasService(_task_store=store)

            restored = service._get(record.id)
            self.assertEqual(restored.status, TaskStatus.FAILED.value)
            self.assertIn("could not be resumed", restored.errors[0])
            self.assertNotEqual(service._task_store._path, TASK_STORE_FILE)


if __name__ == "__main__":
    unittest.main()
