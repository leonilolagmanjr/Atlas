"""Text chunking.

This module has ONE responsibility: split text into fixed-size chunks.
It does NOT do embeddings, indexing, or retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass(frozen=True)
class Chunk:
    """A chunk of text derived from a source document."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    page_number: Optional[int]
    text: str


def chunk_text(*, doc_id: str, text: str, page_number: Optional[int] = None) -> List[Chunk]:
    """Split text into chunks.

    Args:
        doc_id: Stable identifier for the document (e.g., hash).
        text: Raw text to chunk.
        page_number: Optional page number for future citations.

    Returns:
        List of Chunk objects.
    """

    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0

    if not text:
        return chunks

    # Character-based chunking (kept for minimal behavior change).
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunk_text_str = text[start:end]
        chunk_id = f"{doc_id}:{page_number}:{chunk_index}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                page_number=page_number,
                text=chunk_text_str,
            )
        )

        chunk_index += 1
        start += max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    return chunks
