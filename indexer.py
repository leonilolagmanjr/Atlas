"""Automatic document indexing.

ONE responsibility: detect new/modified documents and update the vector store
incrementally using SHA-256 hashes.

The document itself stays on the filesystem and its embeddings stay in Chroma;
the "is this document current?" question is answered by the ``index_documents``
table in the authoritative ``atlas.db``. ``IndexState`` is a thin façade over
:class:`persistence.repositories.SqliteIndexStateRepository` and holds no file
format of its own. ``database/index_state.json`` is imported once by
:mod:`persistence.migration` and is no longer read at runtime.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from chunker import chunk_text
from config import KNOWLEDGE_FOLDER
from document_loader import LoadedDocument, get_documents, read_document
from persistence.repositories import SqliteIndexStateRepository
from vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexState:
    """Persistent index state (doc_id -> sha256), backed by ``atlas.db``.

    ``store`` is an optional repository; when omitted the process-wide
    persistence context is used, so callers keep the simple ``IndexState.load()``
    shape while the storage lives in SQLite.
    """

    doc_hashes: Dict[str, str]

    @staticmethod
    def repository() -> SqliteIndexStateRepository:
        from persistence.context import default_persistence

        return default_persistence().index_state

    @staticmethod
    def load(store: SqliteIndexStateRepository | None = None) -> "IndexState":
        """Read the stored hashes. Never raises; a failure means 'index fresh'."""

        try:
            repository = store if store is not None else IndexState.repository()
            return IndexState(doc_hashes=dict(repository.get_hashes()))
        except Exception:  # noqa: BLE001 - indexing must not fail on a read error
            logger.exception("Failed to load index state")
            return IndexState(doc_hashes={})

    def save(
        self, store: SqliteIndexStateRepository | None = None, *, previous: Dict[str, str] | None = None
    ) -> None:
        """Persist the hash of every document and forget deleted ones.

        ``previous`` is the state this object replaced; ids present there but
        absent here were removed from the knowledge folder and are deleted from
        the index metadata so a stale row cannot keep a deleted PDF "indexed".
        """

        repository = store if store is not None else IndexState.repository()
        for doc_id, content_hash in self.doc_hashes.items():
            repository.set_hash(doc_id, content_hash)
        for doc_id in set(previous or {}) - set(self.doc_hashes):
            repository.remove(doc_id)


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


def index_knowledge_base(
    *,
    vector_store: VectorStore,
    knowledge_folder: Path = KNOWLEDGE_FOLDER,
    index_state: SqliteIndexStateRepository | None = None,
) -> None:
    """Index/refresh documents in the knowledge folder incrementally."""

    if not knowledge_folder.exists():
        logger.warning("Knowledge folder missing: %s", knowledge_folder)
        return

    repository = index_state if index_state is not None else IndexState.repository()
    state = IndexState.load(repository)
    updated_state = dict(state.doc_hashes)

    paths = get_documents(knowledge_folder)

    # Remove documents that were indexed previously but no longer exist, so a
    # deleted PDF stops being served from the vector store and its stale hash is
    # dropped instead of lingering in the state file forever.
    present_ids = {doc_id_for_path(path) for path in paths}
    for doc_id in set(state.doc_hashes) - present_ids:
        logger.info("Index: removed; dropping deleted document %s", doc_id)
        vector_store.delete_by_doc_id(doc_id=doc_id)
        updated_state.pop(doc_id, None)

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

    IndexState(doc_hashes=updated_state).save(repository, previous=state.doc_hashes)

