"""Conversation retrieval index (Milestone 4).

Conversation history is a *different* kind of memory from Atlas's other stores,
and it must not contaminate them:

* KNOWLEDGE   — what is true (``vector_store`` / ``atlas_knowledge``)
* EXPERIENCE  — what worked or failed (``atlas_experience``)
* USER MEMORY — what this user prefers (``atlas_user_memory``)
* CONVERSATION— what was said (this module, ``atlas_conversation``)

This index reuses the embedding infrastructure Atlas already has (the same
ChromaDB persistence directory and the same embedding model), in a **separate
collection** tagged ``memory_type="conversation"``. There is deliberately no
second vector database and no second embedding model.

Like the experience embeddings, this layer is *optional by design*: when Chroma
or the embedder is unavailable, ``available`` is ``False`` and callers fall back
to the deterministic lexical search that already exists in ``MemoryManager``.
Nothing here calls a model or executes anything.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from config import (
    CHROMA_FOLDER,
    CONVERSATION_COLLECTION_NAME,
    CONVERSATION_INDEX_ENABLED,
)
from memory.models import MemoryMessage

logger = logging.getLogger(__name__)

#: Chunks shorter than this are not worth their own embedding.
_MIN_CHARS = 12
#: Chunks longer than this are split so a long answer is retrievable by part.
_MAX_CHARS = 1200


def _chunks(text: str) -> list[str]:
    """Split message text into embeddable pieces (paragraph-aware, bounded)."""

    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= _MAX_CHARS:
        return [cleaned]
    pieces: list[str] = []
    buffer = ""
    for paragraph in cleaned.split("\n\n"):
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) > _MAX_CHARS and buffer:
            pieces.append(buffer)
            buffer = paragraph
        else:
            buffer = candidate
    if buffer:
        pieces.append(buffer)
    # Hard-split anything still oversized rather than silently dropping it.
    final: list[str] = []
    for piece in pieces:
        while len(piece) > _MAX_CHARS:
            final.append(piece[:_MAX_CHARS])
            piece = piece[_MAX_CHARS:]
        if piece:
            final.append(piece)
    return final


class ConversationIndex:
    """Semantic index over stored conversation messages."""

    def __init__(
        self,
        *,
        persist_dir: str | None = None,
        collection_name: str = CONVERSATION_COLLECTION_NAME,
        enabled: bool = CONVERSATION_INDEX_ENABLED,
    ) -> None:
        self._persist_dir = persist_dir or str(CHROMA_FOLDER)
        self._collection_name = collection_name
        self._enabled = bool(enabled)
        self._client = None
        self._collection: Any = None

    # -- availability ------------------------------------------------------------

    @property
    def available(self) -> bool:
        return bool(self._enabled) and bool(self._ensure_collection())

    def _ensure_collection(self):
        if not self._enabled:
            return None
        if self._collection is not None:
            return self._collection if self._collection is not False else None
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - embeddings are optional by design
            logger.info("Conversation embeddings unavailable; using lexical search")
            self._collection = False
        return self._collection if self._collection is not False else None

    # -- writes ------------------------------------------------------------------

    def index_message(self, *, conversation_id: str, message: MemoryMessage) -> bool:
        """Index (or re-index) one message. Returns True when it was indexed."""

        collection = self._ensure_collection()
        if collection is None:
            return False
        if message.role == "assistant" and message.execution_state == "running":
            # An unfinished streamed turn is indexed once it settles, so the
            # index always holds what was actually said.
            return False
        pieces = [piece for piece in _chunks(message.content) if len(piece) >= _MIN_CHARS]
        if not pieces:
            return False
        try:
            ids = [f"{conversation_id}:{message.id}:{index}" for index in range(len(pieces))]
            collection.upsert(
                ids=ids,
                documents=pieces,
                metadatas=[
                    {
                        "memory_type": "conversation",
                        "conversation_id": conversation_id,
                        "message_id": message.id,
                        "role": message.role,
                        "turn_number": int(message.turn_number),
                        "timestamp": message.timestamp.isoformat(),
                        "execution_state": message.execution_state,
                    }
                    for _ in pieces
                ],
            )
            return True
        except Exception:  # noqa: BLE001
            logger.debug("Conversation indexing failed for %s", message.id, exc_info=True)
            return False

    def index_conversation(self, conversation_id: str, messages: Iterable[MemoryMessage]) -> int:
        """Index a whole conversation; returns the number of messages indexed."""

        count = 0
        for message in messages:
            if self.index_message(conversation_id=conversation_id, message=message):
                count += 1
        return count

    def forget_conversation(self, conversation_id: str) -> None:
        """Remove a deleted conversation from the index (no orphan recall)."""

        collection = self._ensure_collection()
        if collection is None:
            return
        try:
            collection.delete(where={"conversation_id": conversation_id})
        except Exception:  # noqa: BLE001
            logger.debug("Could not forget conversation %s from the index", conversation_id, exc_info=True)

    # -- reads -------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        exclude_conversation: str = "",
    ) -> list[dict[str, Any]]:
        """Return semantically relevant stored messages, best first."""

        collection = self._ensure_collection()
        text = (query or "").strip()
        if collection is None or not text:
            return []
        try:
            where = {"memory_type": "conversation"}
            if exclude_conversation:
                where = {
                    "$and": [
                        {"memory_type": "conversation"},
                        {"conversation_id": {"$ne": exclude_conversation}},
                    ]
                }
            result = collection.query(
                query_texts=[text],
                n_results=max(1, min(limit, 50)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:  # noqa: BLE001
            logger.debug("Conversation search failed", exc_info=True)
            return []

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            # Chroma's default space is squared L2 over normalized embeddings, so
            # a similarity is derived honestly from the distance rather than being
            # presented as a raw probability.
            score = max(0.0, 1.0 - float(distance) / 2.0)
            hits.append({
                "conversation_id": str(metadata.get("conversation_id") or ""),
                "message_id": str(metadata.get("message_id") or ""),
                "role": str(metadata.get("role") or ""),
                "excerpt": str(document or "")[:400],
                "timestamp": metadata.get("timestamp"),
                "score": round(score, 4),
                "match": "semantic",
            })
        return hits
