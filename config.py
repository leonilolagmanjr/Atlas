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


# ---- Memory (Conversation) ----
MEMORY_FOLDER: Path = PROJECT_ROOT / "memory"

# Short-term memory window controls
MAX_RETAINED_MESSAGES: int = 10

# Token budget is optional for now. If set, context builder may use a
# lightweight estimate in the future.
MAX_CONTEXT_TOKENS: int = 2048

# Placeholder summarization threshold (conversation size)
AUTO_SUMMARIZE_THRESHOLD: int = 80

# Cap total number of sessions stored on disk (best-effort)
MAX_SESSIONS: int = 50

# Auto-save session/messaging changes to disk.
AUTO_SAVE: bool = True

# ---- Natural-language reasoning pipeline ----
# Master switch: when False, Atlas uses the deterministic fast path only and
# never spends local inference on interpretation/recovery.
ENABLE_LLM_INTERPRETATION: bool = True
# Requests above this deterministic confidence skip the LLM interpreter.
# Requests below it are routed through Qwen for semantic interpretation.
INTERPRETER_CONFIDENCE_THRESHOLD: float = 0.75
# Below this confidence Atlas asks the user a clarifying question instead of
# guessing at an ambiguous target.
CLARIFICATION_CONFIDENCE_THRESHOLD: float = 0.45
# Bounded recovery: maximum LLM-assisted replan attempts per request.
MAX_RECOVERY_ATTEMPTS: int = 2
# Per-call timeout (seconds) for local Qwen reasoning stages.
REASONING_TIMEOUT_SECONDS: float = 60.0
# Emit a structured per-stage diagnostic trace for every request.
DEBUG_PIPELINE: bool = False
# Keys redacted from diagnostic traces before they are logged.
REDACT_KEYS: tuple[str, ...] = ("password", "token", "api_key", "secret")

# ---- Reasoning engine ----
# Master switch for the reasoning layer. When False, Atlas keeps the legacy
# direct plan/execute path (knowledge-base only) for informational requests.
ENABLE_REASONING_ENGINE: bool = True
# Maximum reasoning iterations per request (one iteration per selected source).
# Bounded so a request can never loop forever.
MAX_REASONING_ITERATIONS: int = 3
# Maximum read-only tool calls the reasoning loop may make per request.
MAX_REASONING_TOOL_CALLS: int = 8
# Maximum ordered sources selected for a single request.
MAX_REASONING_SOURCES: int = 3
# When the local knowledge base has no accepted evidence for an ordinary
# question, Atlas answers from the model and/or the web instead of failing.
ENABLE_GENERAL_QUESTION_FALLBACK: bool = True
# Web research: how many result pages the reasoning layer reads per request.
WEB_RESEARCH_MAX_PAGES: int = 2
# Web research: how many search results to request per query.
WEB_RESEARCH_MAX_RESULTS: int = 6
# Bounded content search over allowed folders (filesystem.search_content).
FILESYSTEM_CONTENT_MAX_FILES: int = 200
FILESYSTEM_CONTENT_MAX_BYTES: int = 200_000

# ---- Computer tools ----
EXECUTION_MODE: str = "confirm"
COMPUTER_ROOT: Path = PROJECT_ROOT

# ---- API task history ----
TASK_STORE_FILE: Path = PROJECT_ROOT / "database" / "tasks.json"

# ---- Logging ----
LOG_LEVEL: str = "INFO"
LOG_TO_FILE: bool = False
LOG_FILE: Path = PROJECT_ROOT / "database" / "atlas.log"


