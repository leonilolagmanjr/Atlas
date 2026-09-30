"""Milestone 9/10: a folder listing is a filesystem question, not a writing task.

"List the files in the tests folder" must call the existing read-only
`filesystem.list` capability. If it were treated as content generation the model
would happily invent file names, which is the one thing a local assistant must
not do about the user's own disk.

These tests use the real interpreter with a fake model (so nothing is generated)
and the real tool schema, and assert the deterministic routing.
"""

from __future__ import annotations

import unittest

from reasoning.task_interpreter import SemanticTaskInterpreter


def _interpreter() -> SemanticTaskInterpreter:
    # The model boundary is never consulted for these deterministic routes; a
    # stub keeps the test offline and asserts the rules rather than the model.
    return SemanticTaskInterpreter(ask=lambda **_kwargs: "{}")


class FolderListingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.interpreter = _interpreter()

    def _capabilities(self, text: str) -> list[str]:
        return [action.capability for action in self.interpreter.interpret(text).actions]

    def test_listing_a_named_folder_uses_filesystem_list(self) -> None:
        cases = {
            "List the files in the tests folder.": "tests",
            "what files are in the tests folder": "tests",
            "list everything in the tests folder": "tests",
            "show me the contents of the .kilo folder": ".kilo",
            "list the files in the Program Files folder": "Program Files",
        }
        for text, expected_path in cases.items():
            with self.subTest(text=text):
                task = self.interpreter.interpret(text)
                self.assertEqual([action.capability for action in task.actions], ["filesystem.list"])
                self.assertEqual(task.actions[0].parameters["path"], expected_path)

    def test_an_explicit_path_is_used_verbatim(self) -> None:
        path = "C:\\Users\\someone\\Documents\\GitHub\\Atlas\\tests"
        task = self.interpreter.interpret(f"List the files in {path}")
        self.assertEqual(task.actions[0].capability, "filesystem.list")
        self.assertEqual(task.actions[0].parameters["path"], path)

    def test_a_known_user_folder_resolves(self) -> None:
        task = self.interpreter.interpret("list the files in Downloads")
        self.assertEqual(task.actions[0].capability, "filesystem.list")
        self.assertIn("ownloads", str(task.actions[0].parameters["path"]))

    def test_writing_a_list_is_still_content_generation(self) -> None:
        # "Write a list of fruits" is a writing task, not a directory listing.
        capabilities = self._capabilities("write a list of fruits in notepad")
        self.assertIn("content.generate", capabilities)
        self.assertNotIn("filesystem.list", capabilities)

    def test_folder_creation_is_unaffected(self) -> None:
        task = self.interpreter.interpret("create a folder called Projects")
        self.assertEqual([action.capability for action in task.actions], ["filesystem.create_folder"])

    def test_a_listing_without_a_resolvable_folder_is_left_alone(self) -> None:
        # Nothing is guessed: without a name or path there is no folder to read,
        # so the request is not silently turned into a read of some other place.
        task = self.interpreter.interpret("list the files")
        self.assertNotIn("filesystem.list", [action.capability for action in task.actions])

    def test_a_delete_request_is_not_turned_into_a_file_write(self) -> None:
        # Atlas has no delete capability. Falling through to content generation
        # would write a junk file named after the request instead of saying the
        # thing cannot be done, so no action is planned at all.
        for text in ("delete the file report.pdf", "Delete the file report.pdf."):
            with self.subTest(text=text):
                task = self.interpreter.interpret(text)
                self.assertEqual(task.actions, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
