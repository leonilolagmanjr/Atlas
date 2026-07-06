"""Automatic document indexing.

ONE responsibility: detect new/modified documents and update the vector store
incrementally using SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from chunker import chunk_text
from config import INDEX_STATE_FILE, KNOWLEDGE_FOLDER
from document_loader import LoadedDocument, get_documents, read_document
from vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexState:
    """Persistent index state (doc_id -> sha256)."""

    doc_hashes: Dict[str, str]

    @staticmethod
    def load(path: Path) -> "IndexState":
        if not path.exists():
            return IndexState(doc_hashes={})
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return IndexState(doc_hashes={})
            return IndexState(doc_hashes=dict(raw.get("doc_hashes", {})))
        except Exception:
            logger.exception("Failed to load index state from %s", path)
            return IndexState(doc_hashes={})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"doc_hashes": self.doc_hashes}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Compute SHA-256 for a file."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_id_for_path(path: Path) -> str:
    """Stable document identifier.

    Uses filename stem for readability, while hashing is used for change detection.
    """

    return path.stem


def index_knowledge_base(*, vector_store: VectorStore, knowledge_folder: Path = KNOWLEDGE_FOLDER) -> None:
    """Index/refresh documents in the knowledge folder incrementally."""

    if not knowledge_folder.exists():
        logger.warning("Knowledge folder missing: %s", knowledge_folder)
        return

    state = IndexState.load(INDEX_STATE_FILE)
    updated_state = dict(state.doc_hashes)

    paths = get_documents(knowledge_folder)

    for path in paths:
        try:
            file_hash = sha256_file(path)
            doc_id = doc_id_for_path(path)

            prev_hash = state.doc_hashes.get(doc_id)
            if prev_hash == file_hash:
                logger.info("Index: unchanged; skipping %s", path.name)
                continue

            logger.info("Index: indexing %s", path.name)

            vector_store.delete_by_doc_id(doc_id=doc_id)

            loaded: LoadedDocument = read_document(path)

            chunks = []
            for page in loaded.pages:
                page_chunks = chunk_text(
                    doc_id=doc_id,
                    text=page.text,
                    page_number=page.page_number,
                )
                chunks.extend(page_chunks)

            if not chunks:
                logger.warning("Index: no text extracted; skipping ingestion for %s", path.name)
                continue

            vector_store.add_chunks(doc_id=doc_id, source=path.name, chunks=chunks)
            updated_state[doc_id] = file_hash

        except Exception:
            logger.exception("Indexing failed for %s", path)
            continue

    IndexState(doc_hashes=updated_state).save(INDEX_STATE_FILE)

