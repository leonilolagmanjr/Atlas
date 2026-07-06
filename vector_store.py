"""Vector store (ChromaDB) management.

ONE responsibility: manage ChromaDB (collections, add/delete/query).
This module does NOT read PDFs and does NOT perform indexing decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, DATABASE_FOLDER, EMBEDDING_MODEL_NAME
from chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    """A retrieved chunk from the vector store."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    page_number: Optional[int]
    text: str
    source: str
    distance: float


class VectorStore:
    """Encapsulates Chroma operations and embedding model."""

    def __init__(self, *, persist_dir: str = str(DATABASE_FOLDER), collection_name: str = COLLECTION_NAME):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(collection_name)
        self._embedding_model_name = EMBEDDING_MODEL_NAME
        self._embedding_model: Optional[SentenceTransformer] = None

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        model = self._get_embedding_model()
        # sentence-transformers returns numpy arrays; cast to python lists.
        vectors = model.encode(list(texts))
        return [v.tolist() for v in vectors]

    def delete_by_doc_id(self, *, doc_id: str) -> int:
        """Delete all chunks for a given doc_id."""

        # We store doc_id in metadata.
        try:
            self._collection.delete(where={"doc_id": doc_id})
        except Exception:
            logger.exception("Failed to delete chunks for doc_id=%s", doc_id)
            return 0
        return 1

    def add_chunks(self, *, doc_id: str, source: str, chunks: Sequence[Chunk]) -> int:
        """Add chunk embeddings + metadata to the collection."""

        if not chunks:
            return 0

        ids: List[str] = []
        texts: List[str] = []
        embeddings_texts: List[str] = []
        metadatas: List[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            texts.append(chunk.text)
            embeddings_texts.append(chunk.text)
            metadatas.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source": source,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                }
            )

        embeddings = self._embed_texts(embeddings_texts)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return len(chunks)

    def query(self, *, query_text: str, top_k: int) -> List[SearchHit]:
        """Query for top_k most similar chunks."""

        if top_k <= 0:
            return []

        embedding = self._embed_texts([query_text])[0]

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[SearchHit] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc_text, meta, dist in zip(docs, metas, dists):
            chunk_id = meta.get("chunk_id") or ""
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    page_number=meta.get("page_number"),
                    text=doc_text,
                    source=str(meta.get("source", "")),
                    distance=float(dist),
                )
            )

        return hits

