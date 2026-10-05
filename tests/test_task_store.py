import tempfile
import unittest
from pathlib import Path

from task_store import TaskStore


class TaskStoreTests(unittest.TestCase):
    def test_task_records_round_trip_through_sqlite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with TaskStore(path=Path(temporary_directory) / "tasks.json") as store:
                records = {"task-1": {"status": "COMPLETED", "response": "done"}}

                store.save(records)

                loaded = store.load()
                self.assertEqual(loaded["task-1"]["status"], "COMPLETED")
                self.assertEqual(loaded["task-1"]["response"], "done")

    def test_corrupt_legacy_task_file_is_treated_as_empty(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tasks.json"
            path.write_text("not json", encoding="utf-8")

            with TaskStore(path=path) as store:
                self.assertEqual(store.load(), {})


if __name__ == "__main__":
    unittest.main()
