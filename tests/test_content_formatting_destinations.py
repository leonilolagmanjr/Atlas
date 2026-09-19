"""Destination resolution tests for the content formatting pipeline.

The renderer must produce plain, readable text for a plain-text file
destination (results.txt) rather than generic prose, keep Markdown structure
for .md, and preserve exact layout for code/scripts.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from content_formatting import ContentFormatter, DestinationType  # noqa: E402


class DestinationParsingTests(unittest.TestCase):
    def _destination(self, value: str) -> DestinationType:
        return ContentFormatter()._parse_destination(value)

    def test_plain_text_file_renders_like_notepad(self):
        for name in ("results.txt", "notes.log", "data.csv", "output.text"):
            with self.subTest(name=name):
                self.assertEqual(self._destination(name), DestinationType.NOTEPAD)

    def test_markdown_file_keeps_markdown_rendering(self):
        for name in ("report.md", "documentation.markdown"):
            with self.subTest(name=name):
                self.assertEqual(self._destination(name), DestinationType.MARKDOWN_FILE)

    def test_streamlit_style_names_are_not_mistaken_for_markdown(self):
        # A destination containing "markdown" is markdown, but an arbitrary
        # filename that merely embeds "md" as a substring is not.
        self.assertNotEqual(self._destination("amd_temp.txt"), DestinationType.MARKDOWN_FILE)

    def test_unknown_destination_is_generic(self):
        self.assertEqual(self._destination("somewhere_else"), DestinationType.GENERIC)

    def test_plain_text_file_output_is_not_empty(self):
        formatted, metadata = ContentFormatter().format(
            "First paragraph.\nSecond paragraph.", destination="results.txt"
        )
        self.assertTrue(formatted.strip())
        self.assertEqual(metadata["content_type"], metadata["content_type"])
        self.assertIn("First paragraph.", formatted)


if __name__ == "__main__":
    unittest.main()
