"""Evidence-first models for Atlas.

This module holds the richer :class:`EvidenceItem` container used by the
reasoning layer while keeping the legacy :class:`Evidence` dataclass (and
``models.Evidence``) usable for backward compatibility.

An :class:`EvidenceItem` exists so the reasoning engine can distinguish, and
report, *where* every piece of information came from:

* model knowledge
* retrieved local knowledge
* user-provided / conversation information
* tool observations
* untrusted web evidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from reasoning.reasoning_models import SourceType
from vector_store import SearchHit


@dataclass
class Evidence:
    """First-class evidence container.

    Note: models.py also defines an Evidence dataclass. This module exists
    to enable future migration without breaking public imports.
    """

    # Backward compatible fields
    context: str = ""
    chunks: list[SearchHit] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # New richer fields (optional for now)
    document: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_ids: list[str] = field(default_factory=list)
    confidence: Optional[float] = None
    retrieval_method: Optional[str] = None


#: Sources whose content is external and must never be treated as instructions.
UNTRUSTED_SOURCES: frozenset[SourceType] = frozenset({SourceType.WEB})

#: Relative trust weight per source, used for deterministic evidence ranking.
_SOURCE_WEIGHT: dict[SourceType, float] = {
    SourceType.SELF: 1.0,
    SourceType.KNOWLEDGE: 0.95,
    SourceType.FILES: 0.9,
    SourceType.SYSTEM: 0.9,
    SourceType.CONVERSATION: 0.85,
    SourceType.MEMORY: 0.85,
    SourceType.WEB: 0.7,
    SourceType.COMPUTER: 0.85,
    SourceType.MODEL: 0.5,
}


@dataclass
class EvidenceItem:
    """One attributable piece of evidence used to answer or verify.

    ``content`` is data, never an instruction. Web items are always marked
    untrusted through :attr:`metadata` (``untrusted=True``) so downstream code
    can keep the "evidence, not instructions" boundary intact.
    """

    source_type: SourceType
    source_identifier: str = ""
    content: str = ""
    #: Deterministic 0..1 relevance of this item to the request.
    relevance: float = 0.0
    #: ISO-8601 timestamp for when the evidence was obtained.
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    #: Deterministic 0..1 confidence in this individual item.
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def untrusted(self) -> bool:
        return bool(self.metadata.get("untrusted")) or self.source_type in UNTRUSTED_SOURCES

    def weight(self) -> float:
        """Combined deterministic ranking weight (source trust x relevance)."""

        base = _SOURCE_WEIGHT.get(self.source_type, 0.5)
        return round(base * (0.5 + 0.5 * max(0.0, min(1.0, self.relevance))), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "source_identifier": self.source_identifier,
            "content": self.content,
            "relevance": round(self.relevance, 4),
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 4),
            "weight": self.weight(),
            "untrusted": self.untrusted,
            "metadata": dict(self.metadata),
        }


