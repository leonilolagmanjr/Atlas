"""Content formatting pipeline: normalization, structuring, and destination-aware rendering.

This module transforms raw retrieved/generated content into human-readable,
destination-appropriate text before it is written to an application.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Detected content types for formatting decisions."""
    ARTICLE = "article"
    ESSAY = "essay"
    STORY = "story"
    SCRIPT = "script"
    TRANSCRIPT = "transcript"
    LYRICS = "lyrics"
    POEM = "poem"
    DOCUMENTATION = "documentation"
    CODE = "code"
    LIST = "list"
    TABLE = "table"
    INSTRUCTIONS = "instructions"
    RECIPE = "recipe"
    NEWS = "news"
    REVIEW = "review"
    REFERENCE = "reference"
    CONVERSATION = "conversation"
    GENERAL_PROSE = "general_prose"
    GENERIC = "generic"


class DestinationType(str, Enum):
    """Destination types for rendering."""
    NOTEPAD = "notepad"
    WORDPAD = "wordpad"
    WORD = "word"
    MARKDOWN_FILE = "markdown"
    CODE_EDITOR = "code_editor"
    TERMINAL = "terminal"
    GENERIC = "generic"


@dataclass
class ContentDocument:
    """Structured document representation for formatting pipeline."""
    content: str = ""
    content_type: ContentType = ContentType.GENERIC
    title: str = ""
    sections: list[dict[str, Any]] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    formatting_hints: dict[str, Any] = field(default_factory=dict)
    destination: DestinationType = DestinationType.GENERIC

    def to_plain_text(self) -> str:
        """Render as plain text with preserved structure."""
        parts = []
        if self.title:
            parts.append(self.title)
            parts.append("")
        
        for section in self.sections:
            heading = section.get("heading", "")
            content = section.get("content", "")
            level = section.get("level", 1)
            list_type = section.get("list_type")
            
            if heading:
                prefix = "#" * min(level, 3)
                parts.append(f"{prefix} {heading}")
                parts.append("")
            
            if list_type:
                parts.append(content)
            else:
                parts.append(content)
            parts.append("")
        
        if not self.sections and self.content:
            parts.append(self.content)
        
        return "\n".join(parts).strip()


class ContentNormalizer:
    """Normalize raw web/retrieved content: remove artifacts, fix whitespace, preserve structure."""

    _NAV_PATTERNS = [
        r"(?i)\b(?:cookie|consent|privacy policy|terms of service|accept all|reject all)\b",
        r"(?i)\b(?:sign in|log in|register|subscribe|newsletter)\b",
        r"(?i)\b(?:share|tweet|facebook|linkedin|pinterest|reddit)\b",
        r"(?i)\b(?:advertisement|sponsored|promoted)\b",
        r"(?i)\b(?:related articles?|you may also like|recommended)\b",
        r"(?i)\b(?:skip to|main content|navigation|menu)\b",
        r"(?i)\b(?:found this document useful|ratings?|views?|uploaded by)\b",
        r"(?i)\b(?:download to read|download now|save for later|embed|share)\b",
        r"(?i)\b(?:report this document|document description|original title)\b",
    ]

    _HTML_ARTIFACTS = [
        r"&nbsp;",
        r"&",
        r"<",
        r">",
        r"&#\d+;",
        r"&[a-z]+;",
        r"<[^>]+>",
    ]

    def __init__(self) -> None:
        self._nav_regex = [re.compile(p) for p in self._NAV_PATTERNS]
        self._html_regex = [re.compile(p) for p in self._HTML_ARTIFACTS]

    def normalize(self, raw_content: str, source: str = "", preserve_indentation: bool = False) -> str:
        """Normalize raw content: clean artifacts, fix whitespace, preserve paragraphs."""
        if not raw_content or not raw_content.strip():
            return ""

        text = raw_content

        # Auto-detect if content looks like code (to preserve indentation)
        if not preserve_indentation:
            preserve_indentation = self._looks_like_code(text)

        # Remove HTML entities and tags
        for regex in self._html_regex:
            text = regex.sub(" ", text)

        # Remove navigation/boilerplate patterns
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            # Don't strip if preserving indentation (for code)
            if preserve_indentation:
                line_stripped = line.rstrip()
            else:
                line_stripped = line.strip()
            if not line_stripped:
                cleaned_lines.append("")
                continue
            is_boilerplate = False
            for regex in self._nav_regex:
                if regex.search(line_stripped):
                    is_boilerplate = True
                    break
            if not is_boilerplate:
                cleaned_lines.append(line_stripped)

        # Rejoin and normalize whitespace
        text = "\n".join(cleaned_lines)
        text = self._normalize_whitespace(text, preserve_indentation)
        text = self._deduplicate_lines(text)
        text = self._normalize_line_endings(text)

        return text.strip()

    def _looks_like_code(self, text: str) -> bool:
        """Quick heuristic to detect code-like content."""
        lines = text.splitlines()
        if not lines:
            return False
        # Check for common code patterns
        code_indicators = 0
        for line in lines[:20]:  # Check first 20 lines
            stripped = line.lstrip()
            if not stripped:
                continue
            if stripped.startswith(("def ", "function ", "class ", "import ", "const ", "let ", "var ", "if ", "for ", "while ", "try:", "catch ", "finally:")):
                code_indicators += 1
            if stripped.startswith(("{", "}", "[", "]", "();")):
                code_indicators += 1
            # Lines with significant leading whitespace (indentation)
            if len(line) - len(stripped) >= 4:
                code_indicators += 1
        return code_indicators >= 3

    def _normalize_whitespace(self, text: str, preserve_indentation: bool = False) -> str:
        """Normalize whitespace while preserving paragraph structure."""
        if preserve_indentation:
            # Only normalize multiple spaces within a line, not leading indentation
            lines = text.splitlines()
            normalized_lines = []
            for line in lines:
                # Replace multiple spaces/tabs with single space, but keep leading whitespace
                leading = line[:len(line) - len(line.lstrip())]
                rest = line.lstrip()
                rest = re.sub(r"[ \t]+", " ", rest)
                normalized_lines.append(leading + rest)
            text = "\n".join(normalized_lines)
            # Normalize multiple newlines to at most 2 (paragraph break)
            text = re.sub(r"\n{3,}", "\n\n", text)
        else:
            # Replace multiple spaces with single space (but not newlines)
            text = re.sub(r"[ \t]+", " ", text)
            # Normalize multiple newlines to at most 2 (paragraph break)
            text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _deduplicate_lines(self, text: str) -> str:
        """Remove consecutive duplicate lines (common in scraped content)."""
        lines = text.splitlines()
        if not lines:
            return text
        deduped = [lines[0]]
        for line in lines[1:]:
            if line.strip() != deduped[-1].strip():
                deduped.append(line)
        return "\n".join(deduped)

    def _normalize_line_endings(self, text: str) -> str:
        """Ensure consistent line endings."""
        return text.replace("\r\n", "\n").replace("\r", "\n")


class ContentTypeDetector:
    """Detect content type from text using deterministic signals."""

    _SCRIPT_INDICATORS = [
        (r"(?m)^[A-Z][A-Z0-9 .]{1,30}:\s", 2.0),
        (r"(?i)\b(?:scene|act|int\.|ext\.)\s+\d", 1.5),
        (r"(?i)\b(?:screenplay|script)\b", 1.0),
    ]

    _TRANSCRIPT_INDICATORS = [
        (r"(?m)^\d{1,2}:\d{2}(?::\d{2})?\s", 2.0),
        (r"(?m)^[A-Z][a-z]+:\s", 1.5),
        (r"(?i)\btranscript\b", 1.0),
    ]

    _LYRICS_INDICATORS = [
        (r"(?i)\b(?:verse|chorus|bridge|hook)\b", 1.5),
        (r"(?i)\blyrics?\b", 1.0),
        (r"(?m)^\[(?:verse|chorus|bridge)\]", 1.5),
    ]

    _POEM_INDICATORS = [
        (r"(?m)^\s*[A-Z][a-z]+(?:\s+[a-z]+)*\s*$", 0.5),
        (r"(?i)\bpoem\b", 1.0),
    ]

    _CODE_INDICATORS = [
        (r"(?m)^\s*(?:def|function|class|import|const|let|var)\s", 2.0),
        (r"(?m)^\s*(?:if|for|while|try|catch|finally)\s", 1.5),
        (r"(?m)^\s*[{}()\[\];]", 1.0),
        (r"(?i)\bcode\b", 0.5),
    ]

    _LIST_INDICATORS = [
        (r"(?m)^\s*[\d]+\.\s", 1.5),
        (r"(?m)^\s*[-*]\s", 1.5),
        (r"(?i)\b(?:list of|top \d+|best \d+)\b", 1.0),
    ]

    _ARTICLE_INDICATORS = [
        (r"(?m)^#{1,3}\s", 1.5),
        (r"(?m)^[A-Z][A-Za-z\s]{10,}:$", 1.0),
        (r"(?i)\b(?:article|essay|news|report)\b", 0.5),
    ]

    def detect(self, text: str, title: str = "", url: str = "") -> ContentType:
        """Detect content type using deterministic scoring."""
        if not text or not text.strip():
            return ContentType.GENERIC

        lowered = text.casefold()
        title_lower = title.casefold()
        url_lower = url.casefold()

        scores = {ctype: 0.0 for ctype in ContentType}

        for pattern, weight in self._SCRIPT_INDICATORS:
            matches = len(re.findall(pattern, text))
            if matches:
                scores[ContentType.SCRIPT] += min(matches * weight, 5.0)

        for pattern, weight in self._TRANSCRIPT_INDICATORS:
            matches = len(re.findall(pattern, text))
            if matches:
                scores[ContentType.TRANSCRIPT] += min(matches * weight, 5.0)

        for pattern, weight in self._LYRICS_INDICATORS:
            matches = len(re.findall(pattern, text))
            if matches:
                scores[ContentType.LYRICS] += min(matches * weight, 5.0)

        for pattern, weight in self._POEM_INDICATORS:
            matches = len(re.findall(pattern, text))
            if matches:
                scores[ContentType.POEM] += min(matches * weight, 3.0)

        for pattern, weight in self._CODE_INDICATORS:
            matches = len(re.findall(pattern, text))
            if matches:
                scores[ContentType.CODE] += min(matches * weight, 5.0)

        for pattern, weight in self._LIST_INDICATORS:
            matches = len(re.findall(pattern, text))
            if matches:
                scores[ContentType.LIST] += min(matches * weight, 5.0)

        for pattern, weight in self._ARTICLE_INDICATORS:
            matches = len(re.findall(pattern, text))
            if matches:
                scores[ContentType.ARTICLE] += min(matches * weight, 3.0)

        if "script" in title_lower or "screenplay" in title_lower:
            scores[ContentType.SCRIPT] += 2.0
        if "transcript" in title_lower:
            scores[ContentType.TRANSCRIPT] += 2.0
        if "lyrics" in title_lower:
            scores[ContentType.LYRICS] += 2.0
        if "github.com" in url_lower or "gitlab.com" in url_lower:
            scores[ContentType.CODE] += 2.0
        if "wikipedia.org" in url_lower or "wiki" in url_lower:
            scores[ContentType.REFERENCE] += 2.0

        best_type = max(scores, key=scores.get)
        if scores[best_type] >= 1.0:
            return best_type

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) >= 3:
            return ContentType.GENERAL_PROSE

        return ContentType.GENERIC


class ParagraphReconstructor:
    """Intelligently reconstruct paragraphs from broken text."""

    def reconstruct(self, text: str, content_type: ContentType) -> list[str]:
        """Reconstruct paragraphs based on content type."""
        if not text or not text.strip():
            return []

        if content_type in (ContentType.CODE, ContentType.SCRIPT, ContentType.TRANSCRIPT, ContentType.LYRICS):
            return self._preserve_structure(text)

        return self._reconstruct_prose(text)

    def _preserve_structure(self, text: str) -> list[str]:
        """Preserve existing line/paragraph structure."""
        parts = text.split("\n\n")
        result = []
        for part in parts:
            stripped = part.strip()
            if stripped:
                result.append(stripped)
        return result if result else [text.strip()]

    def _reconstruct_prose(self, text: str) -> list[str]:
        """Reconstruct paragraphs for prose content."""
        lines = [line.strip() for line in text.splitlines()]
        
        rejoined = []
        buffer = ""
        for line in lines:
            if not line:
                if buffer:
                    rejoined.append(buffer)
                    buffer = ""
                rejoined.append("")
                continue
            
            if buffer:
                if re.search(r"[.!?]\s*$", buffer):
                    rejoined.append(buffer)
                    buffer = line
                else:
                    buffer += " " + line
            else:
                buffer = line
        
        if buffer:
            rejoined.append(buffer)

        joined = "\n".join(rejoined)
        paragraphs = [p.strip() for p in joined.split("\n\n") if p.strip()]
        
        return paragraphs if paragraphs else [text.strip()]


class DestinationRenderer:
    """Render ContentDocument for specific destinations."""

    def render(self, document: ContentDocument) -> str:
        """Render document for its destination."""
        destination = document.destination
        
        if destination == DestinationType.NOTEPAD:
            return self._render_notepad(document)
        elif destination in (DestinationType.WORDPAD, DestinationType.WORD):
            return self._render_word(document)
        elif destination == DestinationType.MARKDOWN_FILE:
            return self._render_markdown(document)
        elif destination == DestinationType.CODE_EDITOR:
            return self._render_code_editor(document)
        elif destination == DestinationType.TERMINAL:
            return self._render_terminal(document)
        else:
            return self._render_generic(document)

    def _render_notepad(self, document: ContentDocument) -> str:
        parts = []
        if document.title:
            parts.append(document.title)
            parts.append("")
        
        if document.content_type == ContentType.SCRIPT:
            return self._render_script_notepad(document)
        elif document.content_type == ContentType.TRANSCRIPT:
            return self._render_transcript_notepad(document)
        elif document.content_type == ContentType.LYRICS:
            return self._render_lyrics_notepad(document)
        elif document.content_type == ContentType.LIST:
            return self._render_list_notepad(document)
        elif document.content_type == ContentType.CODE:
            return self._render_code_notepad(document)
        
        for para in document.paragraphs:
            parts.append(para)
            parts.append("")
        
        return "\n".join(parts).strip()

    def _render_script_notepad(self, document: ContentDocument) -> str:
        parts = []
        if document.title:
            parts.append(document.title)
            parts.append("")
        
        content = document.content
        content = re.sub(r"([A-Z][A-Z0-9 .]{1,30}:.*?)(?=\n[A-Z][A-Z0-9 .]{1,30}:|\n\n|$)", r"\1\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        parts.append(content)
        return "\n".join(parts).strip()

    def _render_transcript_notepad(self, document: ContentDocument) -> str:
        parts = []
        if document.title:
            parts.append(document.title)
            parts.append("")
        
        content = document.content
        content = re.sub(r"(\d{1,2}:\d{2}(?::\d{2})?\s+[^:]+:.*?)(?=\n\d{1,2}:\d{2}|\n\n|$)", r"\1\n", content)
        content = re.sub(r"([A-Z][a-z]+:.*?)(?=\n[A-Z][a-z]+:|\n\n|$)", r"\1\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        parts.append(content)
        return "\n".join(parts).strip()

    def _render_lyrics_notepad(self, document: ContentDocument) -> str:
        parts = []
        if document.title:
            parts.append(document.title)
            parts.append("")
        parts.append(document.content)
        return "\n".join(parts).strip()

    def _render_list_notepad(self, document: ContentDocument) -> str:
        parts = []
        if document.title:
            parts.append(document.title)
            parts.append("")
        parts.append(document.content)
        return "\n".join(parts).strip()

    def _render_code_notepad(self, document: ContentDocument) -> str:
        parts = []
        if document.title:
            parts.append(document.title)
            parts.append("")
        parts.append(document.content)
        return "\n".join(parts).strip()

    def _render_word(self, document: ContentDocument) -> str:
        return self._render_notepad(document)

    def _render_markdown(self, document: ContentDocument) -> str:
        parts = []
        if document.title:
            parts.append(f"# {document.title}")
            parts.append("")
        
        if document.content_type == ContentType.CODE:
            parts.append("```")
            parts.append(document.content)
            parts.append("```")
            return "\n".join(parts)
        
        for section in document.sections:
            heading = section.get("heading", "")
            content = section.get("content", "")
            level = section.get("level", 1)
            if heading:
                parts.append(f"{'#' * level} {heading}")
                parts.append("")
            parts.append(content)
            parts.append("")
        
        if not document.sections and document.content:
            parts.append(document.content)
        
        return "\n".join(parts).strip()

    def _render_code_editor(self, document: ContentDocument) -> str:
        return document.content

    def _render_terminal(self, document: ContentDocument) -> str:
        parts = []
        if document.title:
            parts.append(document.title)
        parts.append(document.content)
        return "\n".join(parts).strip()

    def _render_generic(self, document: ContentDocument) -> str:
        return document.to_plain_text()


class FormattingVerifier:
    """Verify formatted output quality before writing."""

    def verify(self, original: str, formatted: str, content_type: ContentType, destination: DestinationType) -> tuple[bool, list[str]]:
        """Verify formatting quality. Returns (passed, issues)."""
        issues = []
        
        if not formatted or not formatted.strip():
            issues.append("Formatted output is empty")
            return False, issues
        
        if len(formatted) > 1000 and "\n" not in formatted:
            issues.append("Large content has no line breaks - likely collapsed")
        
        if re.search(r"\n{4,}", formatted):
            issues.append("Excessive blank lines (4+ consecutive)")
        
        if len(formatted) < len(original) * 0.3 and len(original) > 500:
            issues.append("Formatted content significantly shorter than original")
        
        if content_type == ContentType.CODE:
            if not self._check_code_indentation(original, formatted):
                issues.append("Code indentation may be damaged")
        
        if self._has_excessive_duplication(formatted):
            issues.append("Excessive duplicate content detected")
        
        return len(issues) == 0, issues

    def _check_code_indentation(self, original: str, formatted: str) -> bool:
        orig_lines = [l for l in original.splitlines() if l.strip()]
        fmt_lines = [l for l in formatted.splitlines() if l.strip()]
        
        if not orig_lines or not fmt_lines:
            return True
        
        orig_indented = [len(l) - len(l.lstrip()) for l in orig_lines[:10] if l.startswith((" ", "\t"))]
        fmt_indented = [len(l) - len(l.lstrip()) for l in fmt_lines[:10] if l.startswith((" ", "\t"))]
        
        if orig_indented and fmt_indented:
            return abs(sum(orig_indented) - sum(fmt_indented)) < 10
        
        return True

    def _has_excessive_duplication(self, text: str) -> bool:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 10:
            return False
        unique = set(lines)
        return len(unique) / len(lines) < 0.5


class ContentFormatter:
    """Main formatting pipeline orchestrator."""

    def __init__(self) -> None:
        self.normalizer = ContentNormalizer()
        self.detector = ContentTypeDetector()
        self.reconstructor = ParagraphReconstructor()
        self.renderer = DestinationRenderer()
        self.verifier = FormattingVerifier()

    def format(self, raw_content: str, destination: str = "notepad", title: str = "", source: str = "", url: str = "") -> tuple[str, dict[str, Any]]:
        """Run full formatting pipeline."""
        dest_type = self._parse_destination(destination)
        
        normalized = self.normalizer.normalize(raw_content, source)
        
        content_type = self.detector.detect(normalized, title, url)
        
        paragraphs = self.reconstructor.reconstruct(normalized, content_type)
        
        document = ContentDocument(
            content=normalized,
            content_type=content_type,
            title=title,
            paragraphs=paragraphs,
            source=source,
            destination=dest_type,
        )
        
        formatted = self.renderer.render(document)
        
        passed, issues = self.verifier.verify(raw_content, formatted, content_type, dest_type)
        
        metadata = {
            "content_type": content_type.value,
            "destination": dest_type.value,
            "original_length": len(raw_content),
            "normalized_length": len(normalized),
            "formatted_length": len(formatted),
            "paragraphs_detected": len(paragraphs),
            "verification_passed": passed,
            "verification_issues": issues,
        }
        
        if not passed:
            logger.warning(f"Formatting verification issues: {issues}")
        
        return formatted, metadata

    def _parse_destination(self, destination: str) -> DestinationType:
        lowered = destination.casefold()
        if "notepad" in lowered:
            return DestinationType.NOTEPAD
        elif "wordpad" in lowered:
            return DestinationType.WORDPAD
        elif "word" in lowered and "notepad" not in lowered:
            return DestinationType.WORD
        elif "markdown" in lowered or ".md" in lowered:
            return DestinationType.MARKDOWN_FILE
        elif any(editor in lowered for editor in ["code", "vscode", "editor", "sublime", "vim"]):
            return DestinationType.CODE_EDITOR
        elif any(term in lowered for term in ["terminal", "cmd", "powershell", "console"]):
            return DestinationType.TERMINAL
        # A plain-text file destination (results.txt, notes.log, output.csv)
        # renders as plain text, the same as Notepad, rather than as generic
        # prose; a Markdown/HTML file keeps its own structure.
        if re.search(r"\.(?:txt|log|csv|tsv|text)$", lowered.strip()):
            return DestinationType.NOTEPAD
        return DestinationType.GENERIC

    def format_with_repair(self, raw_content: str, destination: str = "notepad", title: str = "", source: str = "", url: str = "", max_repairs: int = 2) -> tuple[str, dict[str, Any]]:
        """Format with automatic repair on verification failure."""
        formatted, metadata = self.format(raw_content, destination, title, source, url)
        
        repairs = 0
        while not metadata["verification_passed"] and repairs < max_repairs:
            logger.info(f"Formatting repair attempt {repairs + 1}/{max_repairs}")
            repairs += 1
            
            for issue in metadata["verification_issues"]:
                if "collapsed" in issue or "no line breaks" in issue:
                    formatted = self._repair_collapsed(formatted, metadata["content_type"])
                elif "Excessive blank lines" in issue:
                    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
                elif "shorter than original" in issue:
                    formatted = self.normalizer.normalize(raw_content, source)
            
            content_type = ContentType(metadata["content_type"])
            dest_type = DestinationType(metadata["destination"])
            passed, issues = self.verifier.verify(raw_content, formatted, content_type, dest_type)
            metadata["verification_passed"] = passed
            metadata["verification_issues"] = issues
            metadata["repairs_attempted"] = repairs
        
        metadata["final_verification_passed"] = metadata["verification_passed"]
        return formatted, metadata

    def _repair_collapsed(self, text: str, content_type: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        paragraphs = []
        current = []
        for sent in sentences:
            current.append(sent)
            if len(current) >= 3:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
        return "\n\n".join(paragraphs)


def format_content(raw_content: str, destination: str = "notepad", title: str = "", source: str = "", url: str = "") -> tuple[str, dict[str, Any]]:
    """Convenience function for one-shot formatting."""
    formatter = ContentFormatter()
    return formatter.format(raw_content, destination, title, source, url)


def format_content_with_repair(raw_content: str, destination: str = "notepad", title: str = "", source: str = "", url: str = "", max_repairs: int = 2) -> tuple[str, dict[str, Any]]:
    """Convenience function for formatting with repair."""
    formatter = ContentFormatter()
    return formatter.format_with_repair(raw_content, destination, title, source, url, max_repairs)