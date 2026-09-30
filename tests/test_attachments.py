"""Milestone 9: attachments go through the existing document path.

Attachments must not create a second PDF/text pipeline. These tests verify that
reading delegates to the registered file tools, that a refused path stays
refused, and that an image is described honestly rather than reported as
"no text".
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from memory.attachments import (
    Attachment,
    AttachmentReader,
    MAX_ATTACHMENT_CHARS,
    attachment_kind,
    normalize_attachments,
)


class _ToolOutcome:
    def __init__(self, *, success: bool = True, output: Any = None, error: str = "") -> None:
        self.success = success
        self.output = output
        self.error = error


class _RecordingTools:
    """Stands in for ``ToolRouter.execute`` and records the calls it received."""

    def __init__(self, outcomes: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._outcomes = outcomes or {}

    def __call__(self, name: str, parameters: dict[str, Any]) -> Any:
        self.calls.append((name, parameters))
        result = self._outcomes.get(parameters.get("path"), None)
        if isinstance(result, Exception):
            raise result
        if result is not None:
            return result
        return _ToolOutcome(output={"path": parameters.get("path"), "text": "file contents"})


class AttachmentClassificationTests(unittest.TestCase):
    def test_extensions_map_to_kinds(self) -> None:
        self.assertEqual(attachment_kind("notes.txt"), "text")
        self.assertEqual(attachment_kind("report.pdf"), "pdf")
        self.assertEqual(attachment_kind("screenshot.PNG"), "image")
        self.assertEqual(attachment_kind("archive.zip"), "file")
        self.assertEqual(attachment_kind("noextension"), "file")


class NormalizePayloadTests(unittest.TestCase):
    def test_unknown_fields_are_dropped_and_kind_is_derived(self) -> None:
        normalized = normalize_attachments([
            {"name": "notes.txt", "path": "notes.txt", "text": "injected!", "kind": "made_up"},
        ])
        self.assertEqual(len(normalized), 1)
        self.assertNotIn("text", normalized[0])
        self.assertEqual(normalized[0]["kind"], "text")

    def test_empty_and_non_dict_entries_are_ignored(self) -> None:
        self.assertEqual(normalize_attachments([{"nonsense": 1}, "x", None]), [])
        self.assertEqual(normalize_attachments(None), [])

    def test_attachment_count_is_bounded(self) -> None:
        payload = [{"name": f"f{i}.txt", "path": f"f{i}.txt"} for i in range(50)]
        self.assertLessEqual(len(normalize_attachments(payload)), 5)


class AttachmentReaderTests(unittest.TestCase):
    def test_reading_delegates_to_the_existing_file_tool(self) -> None:
        tools = _RecordingTools()
        reader = AttachmentReader(tool_runner=tools)
        results = reader.read([{"name": "notes.txt", "path": "notes.txt", "kind": "text"}])
        self.assertEqual(tools.calls[0][0], "filesystem.read")
        self.assertEqual(tools.calls[0][1]["path"], "notes.txt")
        self.assertEqual(results[0].text, "file contents")

    def test_a_refused_path_is_reported_not_worked_around(self) -> None:
        # The file tool refuses; the reader must report the refusal and must not
        # attempt any other access path.
        tools = _RecordingTools({
            "../secret.txt": _ToolOutcome(success=False, error="Path is outside the allowed root"),
        })
        reader = AttachmentReader(tool_runner=tools)
        results = reader.read([{"name": "secret.txt", "path": "../secret.txt"}])
        self.assertIn("outside the allowed root", results[0].error)
        self.assertEqual(results[0].text, "")
        self.assertEqual(len(tools.calls), 1)

    def test_a_raising_tool_does_not_break_the_turn(self) -> None:
        tools = _RecordingTools({"bad.pdf": RuntimeError("boom")})
        reader = AttachmentReader(tool_runner=tools)
        results = reader.read([{"name": "bad.pdf", "path": "bad.pdf"}])
        self.assertTrue(results[0].error)
        self.assertEqual(results[0].text, "")

    def test_an_image_is_described_honestly(self) -> None:
        tools = _RecordingTools()
        reader = AttachmentReader(tool_runner=tools)
        results = reader.read([{"name": "shot.png", "path": "shot.png", "kind": "image"}])
        self.assertEqual(results[0].kind, "image")
        self.assertEqual(results[0].text, "")
        self.assertIn("vision_observe", results[0].metadata.get("observed_by", ""))
        self.assertIn("image", results[0].as_prompt())
        # No file read was attempted for an image.
        self.assertEqual(tools.calls, [])

    def test_no_reader_available_degrades_honestly(self) -> None:
        reader = AttachmentReader(tool_runner=None)
        results = reader.read([{"name": "notes.txt", "path": "notes.txt"}])
        self.assertIn("no file reader", results[0].error)

    def test_missing_path_is_reported(self) -> None:
        reader = AttachmentReader(tool_runner=_RecordingTools())
        results = reader.read([{"name": "notes.txt"}])
        self.assertIn("no path", results[0].error)

    def test_attachment_text_is_bounded(self) -> None:
        huge = _ToolOutcome(output={"text": "x" * (MAX_ATTACHMENT_CHARS * 3)})
        reader = AttachmentReader(tool_runner=_RecordingTools({"big.txt": huge}))
        results = reader.read([{"name": "big.txt", "path": "big.txt"}])
        self.assertEqual(len(results[0].text), MAX_ATTACHMENT_CHARS)

    def test_empty_output_is_reported_as_such(self) -> None:
        reader = AttachmentReader(tool_runner=_RecordingTools({"empty.txt": _ToolOutcome(output={"text": ""})}))
        results = reader.read([{"name": "empty.txt", "path": "empty.txt"}])
        self.assertEqual(results[0].error, "no readable text")

    def test_document_shape_is_stored_without_raw_text(self) -> None:
        attachment = Attachment(name="notes.txt", path="notes.txt", kind="text", text="secret body")
        record = attachment.to_dict()
        self.assertNotIn("text", record)
        self.assertEqual(record["name"], "notes.txt")

    def test_context_text_renders_readable_attachments(self) -> None:
        attachments = [
            Attachment(name="a.txt", kind="text", text="alpha"),
            Attachment(name="b.pdf", kind="pdf", error="unreadable"),
        ]
        rendered = AttachmentReader.context_text(attachments)
        self.assertIn("alpha", rendered)
        self.assertIn("unreadable", rendered)


class AttachmentContextTests(unittest.TestCase):
    def test_attachment_text_reaches_the_model_prompt(self) -> None:
        from memory.context_manager import ContextBundle

        bundle = ContextBundle(question="Summarize this.", attachments="[notes.txt]\nalpha beta")
        prompt = bundle.as_prompt()
        self.assertIn("ATTACHED FILES", prompt)
        self.assertIn("alpha beta", prompt)

    def test_no_attachments_leaves_the_prompt_unchanged(self) -> None:
        from memory.context_manager import ContextBundle

        prompt = ContextBundle(question="hello").as_prompt()
        self.assertNotIn("ATTACHED FILES", prompt)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
