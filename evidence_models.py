"""Evidence-first models for Atlas V3.2.

This file holds richer Evidence dataclasses while keeping the existing
models.py Evidence type usable for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

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

