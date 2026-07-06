"""Document loading.

ONE responsibility: read documents from disk.
This module supports easy extension to TXT, DOCX, Markdown.

It does NOT chunk, embed, index, or retrieve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader

from config import KNOWLEDGE_FOLDER, KNOWLEDGE_GLOB


@dataclass(frozen=True)
class LoadedPage:
    """A single page of a loaded document."""

    page_number: int
    text: str


@dataclass(frozen=True)
class LoadedDocument:
    """Loaded document content with page boundaries."""

    source_path: Path
    pages: List[LoadedPage]

    @property
    def full_text(self) -> str:
        """Concatenate page texts."""

        return "\n".join(p.text for p in self.pages if p.text)


def get_documents(
    knowledge_folder: Path = KNOWLEDGE_FOLDER,
    pattern: str = KNOWLEDGE_GLOB,
) -> List[Path]:
    """Return candidate document paths from the knowledge folder."""

    if not knowledge_folder.exists():
        return []
    return list(knowledge_folder.glob(pattern))


def read_pdf_document(pdf_path: Path) -> LoadedDocument:
    """Read a PDF and return loaded pages.

    Errors are raised to the caller so indexer can handle them gracefully.
    """

    reader = PdfReader(str(pdf_path))
    pages: List[LoadedPage] = []

    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(LoadedPage(page_number=idx, text=page_text))

    return LoadedDocument(source_path=pdf_path, pages=pages)


def read_document(path: Path) -> LoadedDocument:
    """Read a document by extension.

    Currently supports PDFs.

    Future extension point:
    - TXT
    - DOCX
    - Markdown
    """

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_document(path)

    raise ValueError(f"Unsupported document type: {path}")

