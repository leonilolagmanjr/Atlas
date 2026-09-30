"""Attachments: what a user handed Atlas alongside a message.

An attachment is *input*, not a new pipeline. This module does one bounded job:
turn an attachment payload into the same evidence the existing document path
already produces, by delegating to the registered tools (``filesystem.read``
for permitted local files, ``filesystem.search_content`` for in-root search)
rather than re-implementing PDF/text/OCR handling.

Two rules matter:

* **Attachments are untrusted content, like web evidence.** Their text is data
  about the file, never an instruction Atlas follows. It is read the same way
  fetched pages already are.
* **Attachment reading goes through the existing permission path.** A path the
  filesystem tools refuse is refused here too; nothing bypasses the root check,
  the size bounds, or the permission engine.

Images are described honestly: their pixels are available to the vision layer,
but this module does not pretend to have extracted text from them. Only the
metadata and the path are reported, and the caller may observe the screen with
``computer.vision_observe`` if a visual read is genuinely needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: File extensions Atlas can read as text through the existing file tools.
_TEXT_SUFFIXES: frozenset[str] = frozenset(
    {".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".log", ".py", ".js",
     ".ts", ".tsx", ".jsx", ".html", ".htm", ".css", ".xml", ".ini", ".cfg", ".toml", ".sql"}
)
#: File extensions the existing PDF extraction path understands.
_PDF_SUFFIXES: frozenset[str] = frozenset({".pdf"})
#: Image extensions: observable by the vision layer, not text-extractable here.
_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
)

#: Hard cap on attachment text folded into one context, so a large PDF cannot
#: silently consume the whole prompt budget.
MAX_ATTACHMENT_CHARS: int = 6000
#: Maximum number of attachments read per turn.
MAX_ATTACHMENTS: int = 5


def attachment_kind(name: str) -> str:
    """Classify an attachment by extension for the stored record and the UI."""

    suffix = Path(name).suffix.casefold()
    if suffix in _PDF_SUFFIXES:
        return "pdf"
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _TEXT_SUFFIXES:
        return "text"
    return "file"


@dataclass
class Attachment:
    """One attachment as stored on a conversation message."""

    name: str
    path: str = ""
    kind: str = "file"
    size: int | None = None
    text: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The stored shape. Raw text is not persisted on the message: the file
        itself (and the tool result) is the record, and the text lives in the
        evidence used for the turn."""
        record: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.path:
            record["path"] = self.path
        if self.size is not None:
            record["size"] = self.size
        if self.error:
            record["error"] = self.error
        if self.metadata:
            record["metadata"] = self.metadata
        return record

    def as_prompt(self) -> str:
        if self.error:
            return f"[{self.name}] could not be read: {self.error}"
        if self.text:
            return f"[{self.name}]\n{self.text}"
        if self.kind == "image":
            return f"[{self.name}] is an image; its pixels are available to screen observation, not read as text."
        return f"[{self.name}] contained no readable text."


def normalize_attachments(payload: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Validate and normalize an untrusted attachment payload from the client.

    The client can only name paths and human labels; it cannot inject text as if
    the file had produced it. Unknown fields are dropped.
    """

    if not payload:
        return []
    normalized: list[dict[str, Any]] = []
    for item in payload[:MAX_ATTACHMENTS]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()[:260]
        path = str(item.get("path") or "").strip()[:1000]
        if not name and not path:
            continue
        kind = str(item.get("kind") or "").strip()
        if kind not in {"text", "pdf", "image", "file"}:
            kind = attachment_kind(path or name)
        record: dict[str, Any] = {"name": name or Path(path).name, "path": path, "kind": kind}
        size = item.get("size")
        if isinstance(size, int) and size >= 0:
            record["size"] = size
        normalized.append(record)
    return normalized


class AttachmentReader:
    """Read attachments through the existing, permission-gated file tools."""

    def __init__(
        self,
        *,
        tool_runner: Callable[[str, dict[str, Any]], Any] | None = None,
        max_chars: int = MAX_ATTACHMENT_CHARS,
    ) -> None:
        self._run = tool_runner
        self._max_chars = max(200, int(max_chars))

    def read(self, attachments: list[dict[str, Any]]) -> list[Attachment]:
        """Read each attachment, degrading honestly per file."""

        results: list[Attachment] = []
        for record in attachments[:MAX_ATTACHMENTS]:
            path = str(record.get("path") or "")
            name = str(record.get("name") or Path(path).name or "attachment")
            kind = str(record.get("kind") or attachment_kind(path or name))
            attachment = Attachment(name=name, path=path, kind=kind)
            size = record.get("size")
            if isinstance(size, int):
                attachment.size = size
            if not path:
                attachment.error = "no path was given"
                results.append(attachment)
                continue
            if kind == "image":
                # Images are observed, not text-extracted. Saying so is more
                # useful than returning an empty string as if it were read.
                attachment.metadata["observed_by"] = "computer.vision_observe"
                results.append(attachment)
                continue
            if self._run is None:
                attachment.error = "no file reader is available in this runtime"
                results.append(attachment)
                continue
            try:
                outcome = self._run("filesystem.read", {"path": path, "max_bytes": self._max_chars})
            except Exception as exc:  # noqa: BLE001 - a bad attachment must not break a turn
                logger.exception("Attachment read failed")
                attachment.error = f"{type(exc).__name__}"
                results.append(attachment)
                continue
            text, error = _tool_text(outcome)
            if error:
                attachment.error = error
            elif text:
                attachment.text = text[: self._max_chars]
            else:
                attachment.error = "no readable text"
            results.append(attachment)
        return results

    @staticmethod
    def context_text(attachments: list[Attachment]) -> str:
        """Render the readable attachments as bounded context."""

        rendered = [item.as_prompt() for item in attachments if item.text or item.error]
        return "\n\n".join(rendered)[:MAX_ATTACHMENT_CHARS * 2]


def _tool_text(outcome: Any) -> tuple[str, str]:
    """Extract text from a tool outcome without assuming one shape."""

    if outcome is None:
        return "", "the file reader returned nothing"
    success = getattr(outcome, "success", None)
    if success is False:
        return "", str(getattr(outcome, "error", "") or "the file reader failed")
    output = getattr(outcome, "output", outcome)
    if isinstance(output, dict):
        for key in ("text", "content", "page_text"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value, ""
        return "", "no readable text"
    if isinstance(output, str):
        return output, ""
    error = getattr(outcome, "error", "")
    return ("", str(error)) if error else ("", "")
