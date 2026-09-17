"""Synthesis and output formatting for Atlas.

This module handles the transformation of raw retrieved information into
structured, well-formatted output suitable for the requested destination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from models_task import EvidenceSource, EvidenceState


@dataclass
class DocumentStructure:
    """Structured document representation preserving formatting."""
    title: str = ""
    subtitle: str = ""
    sections: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_section(self, heading: str, content: str, level: int = 1, list_type: str | None = None) -> None:
        """Add a section to the document."""
        self.sections.append({
            "heading": heading,
            "content": content,
            "level": level,
            "list_type": list_type,  # "bullet", "numbered", or None
        })

    def to_plain_text(self) -> str:
        """Render as plain text with preserved structure."""
        parts = []
        if self.title:
            parts.append(self.title.upper())
            parts.append("=" * len(self.title))
            parts.append("")
        if self.subtitle:
            parts.append(self.subtitle)
            parts.append("-" * len(self.subtitle))
            parts.append("")
        
        for section in self.sections:
            heading = section["heading"]
            content = section["content"]
            level = section["level"]
            list_type = section["list_type"]
            
            if heading:
                prefix = "#" * level
                parts.append(f"{prefix} {heading}")
                parts.append("")
            
            if list_type:
                # Content is already formatted as list items
                parts.append(content)
            else:
                parts.append(content)
            parts.append("")
        
        return "\n".join(parts).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "sections": self.sections,
            "metadata": self.metadata,
        }


class ContentSynthesizer:
    """Synthesize retrieved evidence into structured content."""

    def __init__(self) -> None:
        pass

    def synthesize(self, evidence_state: EvidenceState, task_goal: str, task_type: str) -> DocumentStructure:
        """Synthesize evidence into a structured document based on the task goal."""
        
        if not evidence_state.relevant_sources:
            return DocumentStructure(
                title="No Relevant Information Found",
                subtitle=f"Could not retrieve sufficient information for: {evidence_state.target}",
            )

        # Determine synthesis strategy based on goal and content type
        if evidence_state.goal == "retrieve_document" and evidence_state.must_be_artifact:
            return self._synthesize_artifact(evidence_state)
        elif evidence_state.goal == "find_review":
            return self._synthesize_review_summary(evidence_state)
        elif evidence_state.goal == "find_reference":
            return self._synthesize_reference_summary(evidence_state)
        elif evidence_state.goal == "find_information":
            return self._synthesize_information_summary(evidence_state)
        else:
            return self._synthesize_generic_summary(evidence_state)

    def _synthesize_artifact(self, evidence_state: EvidenceState) -> DocumentStructure:
        """Synthesize when the user wants the actual artifact (script, transcript, etc.)."""
        # Find the best artifact source
        artifact_sources = [s for s in evidence_state.relevant_sources if s.is_artifact]
        if not artifact_sources:
            artifact_sources = evidence_state.relevant_sources
        
        # Use the highest relevance artifact source
        best_source = max(artifact_sources, key=lambda s: s.relevance_score)
        
        doc = DocumentStructure(
            title=best_source.title or evidence_state.target,
            subtitle=f"Source: {best_source.url}" if best_source.url else "",
            metadata={"source_type": "artifact", "source_url": best_source.url},
        )
        
        # For artifacts, preserve the content as-is with minimal formatting
        content = best_source.content.strip()
        if content:
            doc.add_section("", content)
        
        return doc

    def _synthesize_review_summary(self, evidence_state: EvidenceState) -> DocumentStructure:
        """Synthesize review/opinion information."""
        doc = DocumentStructure(
            title=f"Reviews: {evidence_state.target}",
            subtitle=f"Based on {len(evidence_state.relevant_sources)} sources",
        )
        
        # Group by source type
        review_sources = [s for s in evidence_state.relevant_sources if s.source_type in {"reference", "secondary"}]
        
        for i, source in enumerate(review_sources[:5], 1):
            if source.content:
                # Extract key points from review content
                summary = self._extract_key_points(source.content, max_points=3)
                doc.add_section(
                    f"{source.title or f'Source {i}'}",
                    summary,
                    level=2,
                )
        
        return doc

    def _synthesize_reference_summary(self, evidence_state: EvidenceState) -> DocumentStructure:
        """Synthesize encyclopedic/reference information."""
        doc = DocumentStructure(
            title=f"Reference: {evidence_state.target}",
            subtitle=f"Based on {len(evidence_state.relevant_sources)} sources",
        )
        
        # Prefer reference-type sources
        ref_sources = [s for s in evidence_state.relevant_sources if s.source_type == "reference"]
        if not ref_sources:
            ref_sources = evidence_state.relevant_sources
        
        for source in ref_sources[:3]:
            if source.content:
                summary = self._extract_key_points(source.content, max_points=5)
                doc.add_section(
                    source.title or "Overview",
                    summary,
                    level=2,
                )
        
        return doc

    def _synthesize_information_summary(self, evidence_state: EvidenceState) -> DocumentStructure:
        """Synthesize general information request."""
        doc = DocumentStructure(
            title=f"Information: {evidence_state.target}",
            subtitle=f"Based on {len(evidence_state.relevant_sources)} sources",
        )
        
        # Group sources by type
        for source in evidence_state.relevant_sources[:5]:
            if source.content:
                summary = self._extract_key_points(source.content, max_points=3)
                source_label = f"[{source.source_type}] {source.title or 'Source'}"
                doc.add_section(source_label, summary, level=2)
        
        return doc

    def _synthesize_generic_summary(self, evidence_state: EvidenceState) -> DocumentStructure:
        """Generic synthesis fallback."""
        doc = DocumentStructure(
            title=f"Summary: {evidence_state.target}",
            subtitle=f"Based on {len(evidence_state.relevant_sources)} sources",
        )
        
        for source in evidence_state.relevant_sources[:5]:
            if source.content:
                summary = self._extract_key_points(source.content, max_points=3)
                doc.add_section(source.title or "Source", summary, level=2)
        
        return doc

    def _extract_key_points(self, text: str, max_points: int = 5) -> str:
        """Extract key points from text content."""
        # Split into sentences/paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences.extend([s.strip() for s in re.split(r'[.!?]+', p) if s.strip()])
        
        # Filter out very short/long sentences and boilerplate
        filtered = []
        for s in sentences:
            if 20 <= len(s) <= 300:
                # Skip boilerplate patterns
                if not any(b in s.casefold() for b in ["cookie", "consent", "privacy policy", "terms of service", "javascript", "enable javascript"]):
                    filtered.append(s)
        
        # Take top sentences
        selected = filtered[:max_points]
        if not selected and paragraphs:
            selected = paragraphs[:max_points]
        
        if not selected:
            return text[:500] + ("..." if len(text) > 500 else "")
        
        return "\n".join(f"• {point}" for point in selected)


class OutputFormatter:
    """Format synthesized content for specific destinations."""

    def __init__(self) -> None:
        pass

    def format_for_destination(self, document: DocumentStructure, destination: str, content_type: str = "text") -> str:
        """Format document for a specific destination application."""
        destination_lower = destination.casefold()
        
        if "notepad" in destination_lower:
            return self._format_for_notepad(document)
        elif "word" in destination_lower or "wordpad" in destination_lower:
            return self._format_for_word(document)
        elif "code" in destination_lower or "vscode" in destination_lower or "editor" in destination_lower:
            return self._format_for_code_editor(document)
        else:
            return self._format_generic(document)

    def _format_for_notepad(self, document: DocumentStructure) -> str:
        """Format for Notepad - plain text with clear structure."""
        return document.to_plain_text()

    def _format_for_word(self, document: DocumentStructure) -> str:
        """Format for Word/WordPad - richer text with headings."""
        return document.to_plain_text()

    def _format_for_code_editor(self, document: DocumentStructure) -> str:
        """Format for code editors - preserve code structure."""
        return document.to_plain_text()

    def _format_generic(self, document: DocumentStructure) -> str:
        """Generic formatting fallback."""
        return document.to_plain_text()


def synthesize_and_format(evidence_state: EvidenceState, task_goal: str, task_type: str, destination: str) -> tuple[DocumentStructure, str]:
    """Convenience function to synthesize and format in one call."""
    synthesizer = ContentSynthesizer()
    formatter = OutputFormatter()
    
    document = synthesizer.synthesize(evidence_state, task_goal, task_type)
    formatted = formatter.format_for_destination(document, destination)
    
    return document, formatted