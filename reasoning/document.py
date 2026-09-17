"""Structured document representation between content and delivery.

Retrieved or generated text is never treated as an opaque string. It is
recovered into a :class:`Document` (title, headings, paragraphs, lists, code)
and only then rendered for a destination. This is what keeps paragraph
boundaries, headings, list structure, and line breaks intact all the way into
Notepad instead of collapsing a whole page into one extremely long line.

The module is fully deterministic and model-free:

* :func:`document_from_text` recovers structure from text (paragraph breaks,
  headings, bulleted/numbered lists, fenced code) without inventing content.
* :func:`reflow` guarantees readable paragraphs for prose that arrives as a
  single unbroken run, splitting only at existing sentence/word boundaries.
* :class:`Document` renders for a destination (plain text, Windows controls)
  and reports structural facts (paragraph count, line count, word count) that
  verification can use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

#: Block kinds. Kept small and semantic; the renderer owns presentation.
PARAGRAPH = "paragraph"
HEADING = "heading"
LIST_ITEM = "list_item"
CODE = "code"

#: Longest run of text that may appear as a single visual line before the
#: formatter re-flows it at a sentence boundary.
MAX_LINE_CHARS = 600
#: A line this short with no sentence punctuation can be a heading.
MAX_HEADING_CHARS = 80
MAX_HEADING_WORDS = 12

_BLANK_LINE_RE = re.compile(r"\n\s*\n")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*\u2022\u2023\u2013]|\d{1,3}[.)])\s+(?P<body>\S.*)$")
_ORDERED_MARKER_RE = re.compile(r"^\s*(?P<number>\d{1,3})[.)]\s+")
_FENCE_RE = re.compile(r"^\s*```")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_HEADING_PREFIX_RE = re.compile(r"^\s*#{1,6}\s+(?P<title>.+?)\s*$")
_WS_RUN_RE = re.compile(r"[ \t\f\v]+")


@dataclass
class Block:
    """One structural unit of a document."""

    kind: str
    text: str
    #: Heading level (1..6) or 1-based list position; 0 when not applicable.
    level: int = 0
    ordered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "level": self.level,
            "ordered": self.ordered,
        }


@dataclass
class Document:
    """An ordered, structured body of text destined for some output."""

    blocks: list[Block] = field(default_factory=list)
    title: str = ""
    subtitle: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- structural facts -------------------------------------------------------

    @property
    def paragraphs(self) -> list[str]:
        return [block.text for block in self.blocks if block.kind == PARAGRAPH]

    @property
    def headings(self) -> list[str]:
        return [block.text for block in self.blocks if block.kind == HEADING]

    @property
    def list_items(self) -> list[str]:
        return [block.text for block in self.blocks if block.kind == LIST_ITEM]

    @property
    def word_count(self) -> int:
        return len(self.to_text().split())

    @property
    def character_count(self) -> int:
        return len(self.to_text())

    def is_empty(self) -> bool:
        return not any(block.text.strip() for block in self.blocks) and not self.title.strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "blocks": [block.to_dict() for block in self.blocks],
            "paragraph_count": len(self.paragraphs),
            "heading_count": len(self.headings),
            "list_item_count": len(self.list_items),
            "word_count": self.word_count,
            "metadata": dict(self.metadata),
        }

    # -- rendering --------------------------------------------------------------

    def to_text(self, *, line_ending: str = "\n") -> str:
        """Render the document, preserving block boundaries and blank lines."""

        pieces: list[str] = []
        if self.title.strip():
            pieces.append(self.title.strip())
        if self.subtitle.strip():
            pieces.append(self.subtitle.strip())
        for block in self.blocks:
            text = block.text.strip()
            if not text:
                continue
            if block.kind == LIST_ITEM:
                marker = f"{block.level}." if block.ordered else "-"
                pieces.append(f"{marker} {text}")
            else:
                pieces.append(text)
        return line_ending.join(_join_blocks(pieces))


def _join_blocks(pieces: list[str]) -> list[str]:
    """Join block strings with one blank line between them."""

    if not pieces:
        return []
    body = "\n\n".join(pieces)
    return body.replace("\r\n", "\n").replace("\r", "\n").split("\n")


# ---------------------------------------------------------------------------
# Structure recovery
# ---------------------------------------------------------------------------


def normalize_line_endings(text: str) -> str:
    """Return ``text`` with CRLF/CR line endings normalized to LF."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def _is_heading_candidate(line: str, *, previous_blank: bool, next_blank: bool) -> bool:
    text = line.strip()
    if not text or len(text) > MAX_HEADING_CHARS:
        return False
    if len(text.split()) > MAX_HEADING_WORDS:
        return False
    if text.endswith((".", ",", ";", ":", "?", "!")):
        return False
    if not (previous_blank or next_blank):
        return False
    words = [word for word in text.split() if any(ch.isalpha() for ch in word)]
    if not words:
        return False
    if text.isupper():
        return True
    capitalised = sum(1 for word in words if word[:1].isupper())
    return capitalised >= max(1, len(words) - 1)


def document_from_text(
    text: str,
    *,
    title: str = "",
    subtitle: str = "",
    metadata: dict[str, Any] | None = None,
    reflow_long_lines: bool = True,
) -> Document:
    """Recover a :class:`Document` from text without inventing content.

    Blank lines separate blocks, list markers become list items, and short
    unpunctuated lines surrounded by breaks become headings. A single unbroken
    run of prose is re-flowed at existing sentence boundaries so it can never
    render as one extremely long line.
    """

    document = Document(
        title=title.strip(),
        subtitle=subtitle.strip(),
        metadata=dict(metadata or {}),
    )
    body = normalize_line_endings(text or "")
    if not body.strip():
        return document

    segments = _split_fences(body) if _FENCE_RE.search(body) else [(False, body)]
    for is_code, segment in segments:
        if is_code:
            document.blocks.append(Block(kind=CODE, text=segment.strip("\n")))
            continue
        document.blocks.extend(_blocks_from_prose(segment, reflow_long_lines=reflow_long_lines))
    return document


def _split_fences(text: str) -> list[tuple[bool, str]]:
    """Split text into (is_code, segment) pairs using ``` fences."""

    segments: list[tuple[bool, str]] = []
    buffer: list[str] = []
    in_code = False
    for line in normalize_line_endings(text).split("\n"):
        if _FENCE_RE.match(line):
            if buffer or segments:
                segments.append((in_code, "\n".join(buffer)))
            buffer = []
            in_code = not in_code
            continue
        buffer.append(line)
    if buffer:
        segments.append((in_code, "\n".join(buffer)))
    return segments


def _blocks_from_prose(text: str, *, reflow_long_lines: bool) -> list[Block]:
    blocks: list[Block] = []
    for raw_block in _BLANK_LINE_RE.split(normalize_line_endings(text)):
        if not raw_block.strip():
            continue
        lines = raw_block.split("\n")
        heading_prefix = _HEADING_PREFIX_RE.match(lines[0])
        if heading_prefix and len(lines) == 1:
            blocks.append(
                Block(
                    kind=HEADING,
                    text=heading_prefix.group("title").strip(),
                    level=min(6, lines[0].count("#") or 1),
                )
            )
            continue

        paragraph_lines: list[str] = []
        for index, line in enumerate(lines):
            marker = _LIST_MARKER_RE.match(line)
            if marker:
                _flush_paragraph(blocks, paragraph_lines, reflow_long_lines=reflow_long_lines)
                paragraph_lines = []
                ordered_match = _ORDERED_MARKER_RE.match(line)
                blocks.append(
                    Block(
                        kind=LIST_ITEM,
                        text=marker.group("body").strip(),
                        level=int(ordered_match.group("number")) if ordered_match else 0,
                        ordered=bool(ordered_match),
                    )
                )
                continue
            stripped = line.strip()
            if not stripped:
                continue
            previous_blank = index == 0 or not lines[index - 1].strip()
            next_blank = index == len(lines) - 1 or not lines[index + 1].strip()
            if _is_heading_candidate(stripped, previous_blank=previous_blank, next_blank=next_blank):
                _flush_paragraph(blocks, paragraph_lines, reflow_long_lines=reflow_long_lines)
                paragraph_lines = []
                blocks.append(Block(kind=HEADING, text=stripped, level=2))
                continue
            paragraph_lines.append(stripped)
        _flush_paragraph(blocks, paragraph_lines, reflow_long_lines=reflow_long_lines)
    return blocks


def _flush_paragraph(blocks: list[Block], lines: list[str], *, reflow_long_lines: bool) -> None:
    if not lines:
        return
    paragraph = _WS_RUN_RE.sub(" ", " ".join(lines)).strip()
    if not paragraph:
        return
    pieces = reflow(paragraph) if reflow_long_lines else [paragraph]
    for piece in pieces:
        if piece:
            blocks.append(Block(kind=PARAGRAPH, text=piece))


def reflow(paragraph: str, *, max_chars: int = MAX_LINE_CHARS) -> list[str]:
    """Split an over-long paragraph at sentence boundaries, never mid-word.

    Only whitespace changes: the words, their order, and their punctuation are
    preserved exactly.
    """

    text = _WS_RUN_RE.sub(" ", paragraph).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in _SENTENCE_END_RE.split(text):
        if not sentence:
            continue
        if current and length + len(sentence) + 1 > max_chars:
            pieces.append(" ".join(current))
            current = []
            length = 0
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        pieces.append(" ".join(current))
    return [piece for piece in pieces if piece]


def render_for_delivery(text: str, *, title: str = "", line_ending: str = "\n") -> str:
    """Deterministically render arbitrary text for a destination.

    Used at the delivery boundary so a structured body keeps its paragraph
    boundaries and a single unbroken blob is never delivered as one line.
    """

    rendered = document_from_text(text, title=title).to_text(line_ending=line_ending)
    if rendered.strip():
        return rendered
    return normalize_line_endings(text or "").strip()


def append_documents(documents: Iterable[Document]) -> Document:
    """Combine several documents into one, preserving order."""

    combined = Document()
    for document in documents:
        if document is None:
            continue
        if not combined.title and document.title:
            combined.title = document.title
        if not combined.subtitle and document.subtitle:
            combined.subtitle = document.subtitle
        combined.blocks.extend(document.blocks)
    return combined