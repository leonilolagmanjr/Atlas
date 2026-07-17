# Configuration

Atlas configuration is centralized in `config.py`.

This repository does not use a `.env` loader in code; the settings are defined as Python constants.

## Knowledge ingestion & indexing

- `KNOWLEDGE_FOLDER: Path`
  - Default: `<repo>/knowledge`
  - Used by `atlas.py` and `indexer.py`.

- `KNOWLEDGE_GLOB: str`
  - Default: `*.pdf`
  - `document_loader.get_documents()` uses this pattern.

- `INDEX_STATE_FILE: Path`
  - Default: `<repo>/database/index_state.json`
  - Used to detect unchanged knowledge files via SHA-256 hashes.

## Vector database (Chroma)

- `DATABASE_FOLDER: Path`
  - Default: `<repo>/database`

- `COLLECTION_NAME: str`
  - Default: `atlas_knowledge`

## Embeddings

- `EMBEDDING_MODEL_NAME: str`
  - Default: `all-MiniLM-L6-v2`
  - Loaded lazily in `vector_store.VectorStore`.

## Chunking

- `CHUNK_SIZE: int`
  - Default: `500`
  - Character-based chunk size.

- `CHUNK_OVERLAP: int`
  - Default: `100`
  - Character overlap between chunks.

## Retrieval

- `TOP_K: int`
  - Default: `5`
  - Number of chunks returned after ranking.

- `MIN_SIMILARITY: float`
  - Default: `0.75`
  - Used as part of the *retrieval acceptance* decision in `knowledge_search._decide()`.
  - The decision is not a single hard threshold; it combines semantic distance with
    keyword/filename match counts.

- `LOG_RETRIEVAL: bool`
  - Default: `True`
  - Enables retrieval diagnostics in logs.

## LLM provider

- `OLLAMA_MODEL: str`
  - Default: `qwen2.5:7b`
  - Used by `providers/ollama_provider.py` when calling `ollama.chat()`.

## Logging

- `LOG_LEVEL: str`
  - Default: `INFO`

- `LOG_TO_FILE: bool`
  - Default: `False`

- `LOG_FILE: Path`
  - Default: `<repo>/database/atlas.log`
  - Only used when `LOG_TO_FILE` is enabled.

