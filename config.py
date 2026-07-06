"""Application configuration.

All configurable values must be centralized here.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent


# ---- Ollama / LLM ----
OLLAMA_MODEL: str = "qwen2.5:7b"


# ---- Knowledge / Indexing ----
KNOWLEDGE_FOLDER: Path = PROJECT_ROOT / "knowledge"

# Supported extensions: document_loader handles reading.
KNOWLEDGE_GLOB: str = "*.pdf"

# Persistent metadata for incremental indexing.
INDEX_STATE_FILE: Path = PROJECT_ROOT / "database" / "index_state.json"

# Chroma
DATABASE_FOLDER: Path = PROJECT_ROOT / "database"
COLLECTION_NAME: str = "atlas_knowledge"


# ---- Embeddings ----
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"


# ---- Chunking ----
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100


# ---- Retrieval ----
TOP_K: int = 5

# Similarity gating: if best match is below this threshold,
# Atlas must not call the LLM.
# WARNING: Chroma distance metric depends on collection setup.
# Current gating is based on "distance" where smaller is more similar.
# Lower this if you experience excessive gating-off.
MIN_SIMILARITY: float = 0.75

# If True, log retrieved candidates.
LOG_RETRIEVAL: bool = True


# ---- Logging ----
LOG_LEVEL: str = "INFO"
LOG_TO_FILE: bool = False
LOG_FILE: Path = PROJECT_ROOT / "database" / "atlas.log"

