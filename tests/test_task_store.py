import tempfile
import unittest
from pathlib import Path

from task_store import TaskStore


class TaskStoreTests(unittest.TestCase):
    def test_task_records_round_trip_as_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = TaskStore(path=Path(temporary_directory) / "tasks.json")
            records = {"task-1": {"status": "COMPLETED", "response": "done"}}

            store.save(records)

            self.assertEqual(store.load(), records)

    def test_corrupt_task_file_is_treated_as_empty(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tasks.json"
            path.write_text("not json", encoding="utf-8")

            self.assertEqual(TaskStore(path=path).load(), {})


if __name__ == "__main__":
    unittest.main()
