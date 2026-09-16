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

## Command intelligence (natural language → action)

- `ENABLE_LLM_INTERPRETATION: bool`
  - Default: `True`
  - Master switch for Qwen-powered semantic interpretation, LLM-assisted
    planning, and failure recovery. When `False`, Atlas uses only the
    deterministic fast path and never spends local inference on interpretation.

- `INTERPRETER_CONFIDENCE_THRESHOLD: float`
  - Default: `0.75`
  - Requests whose deterministic interpretation meets or exceeds this
    confidence skip the LLM interpreter. This keeps simple requests fast.

- `CLARIFICATION_CONFIDENCE_THRESHOLD: float`
  - Default: `0.45`
  - Below this confidence, an unknown request triggers a clarifying question
    instead of a guess. Intents that need a destination but have none always ask.

- `MAX_RECOVERY_ATTEMPTS: int`
  - Default: `2`
  - Maximum bounded, catalog-constrained recovery attempts per failed step.

- `REASONING_TIMEOUT_SECONDS: float`
  - Default: `60.0`
  - Per-call timeout budget for reasoning stages.

- `DEBUG_PIPELINE: bool`
  - Default: `False`
  - When enabled, Atlas logs one structured trace per request:
    `USER INPUT`, `INTENT`, `PLAN`, `EXECUTION`, `RESULT`.

- `REDACT_KEYS: tuple[str, ...]`
  - Default: `("password", "token", "api_key", "secret")`
  - Keys redacted from diagnostic traces before logging.

## Computer runtime and API

- `EXECUTION_MODE: str`
  - Default: `confirm`
  - Values: `safe`, `confirm`, or `autonomous`.
  - Controls permission decisions for registered tools.

- `COMPUTER_ROOT: Path`
  - Default: repository root.
  - Filesystem tools reject paths outside this root.

- `TASK_STORE_FILE: Path`
  - Default: `<repo>/database/tasks.json`.
  - Stores API task snapshots using atomic replacement.
  - Live execution contexts are not serialized yet; interrupted in-flight
    tasks are marked failed when the API restarts.

The web API binds to `127.0.0.1:8000` when started with the documented
`uvicorn` command. The Vite frontend uses `http://127.0.0.1:8000/api` unless
`VITE_ATLAS_API_URL` is supplied at frontend build time.

## Logging

- `LOG_LEVEL: str`
  - Default: `INFO`

- `LOG_TO_FILE: bool`
  - Default: `False`

- `LOG_FILE: Path`
  - Default: `<repo>/database/atlas.log`
  - Only used when `LOG_TO_FILE` is enabled.

## Reasoning provider

The reasoning stages reuse the existing `OLLAMA_MODEL`. All reasoning calls go
through `providers/ollama_provider.py`. The interpreter, planner, and recovery
stages never import a concrete provider; the callable is injected so the model
can be replaced or faked in tests.

