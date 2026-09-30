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
# Placeholder summarization threshold (conversation size)
AUTO_SUMMARIZE_THRESHOLD: int = 80

# Cap total number of sessions stored on disk (best-effort)
MAX_SESSIONS: int = 50

# Auto-save session/messaging changes to disk.
AUTO_SAVE: bool = True

# ---- Conversation context (Context Manager) ----
#: Maximum number of recent turns included verbatim in a model context.
CONTEXT_MAX_RECENT_TURNS: int = 12
#: Maximum number of *relevant older* turns recalled on top of the recent ones.
CONTEXT_MAX_OLDER_TURNS: int = 6
#: Total character budget for assembled context (recent + summary + recall + memory).
CONTEXT_MAX_CHARS: int = 12000
#: Minimum relevance for a recalled older turn to be included.
CONTEXT_RELEVANCE_FLOOR: float = 0.35
#: Allow the deterministic lexical recall path when embeddings are unavailable.
CONTEXT_LEXICAL_FALLBACK: bool = True

# ---- Conversation retrieval index (semantic search over history) ----
#: Master switch for the conversation embedding index. When False, conversation
#: search falls back to the deterministic lexical search in MemoryManager.
CONVERSATION_INDEX_ENABLED: bool = True
#: Separate Chroma collection so conversation history never mixes with knowledge
#: or experience embeddings. Reuses the existing database folder + embedder.
CONVERSATION_COLLECTION_NAME: str = "atlas_conversation"
#: Conversations summarized after this many messages (real summarization).
CONVERSATION_SUMMARY_MIN_MESSAGES: int = 12

# ---- Long-term user memory ----
#: Master switch for user memory. When False nothing is stored or retrieved.
ENABLE_USER_MEMORY: bool = True
#: Append-only JSONL of durable user memory records (survives restart).
USER_MEMORY_FILE: Path = PROJECT_ROOT / "database" / "user_memory.jsonl"
#: Separate Chroma collection for user-memory embeddings (never mixed).
USER_MEMORY_COLLECTION_NAME: str = "atlas_user_memory"
#: Maximum memories surfaced into one context.
USER_MEMORY_RETRIEVAL_LIMIT: int = 4
#: Retained user-memory record cap so the store stays a bounded history.
USER_MEMORY_MAX_RECORDS: int = 2000

# ---- Conversation streaming ----
#: Allow Server-Sent Events streaming of real execution events. When False the
#: API still serves the same turn, just without a live event stream.
ENABLE_CONVERSATION_STREAMING: bool = True

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
# Observe the UI before/after computer actions so Atlas can verify an effect it
# caused instead of trusting a tool-reported success. Read-only and cheap;
# disable only if window enumeration is undesirable in this environment.
ENABLE_COMPUTER_OBSERVATION: bool = True
# ---- Local computer vision ("Eyes") ----
# Layered perception: Windows UI Automation -> OCR -> image processing -> VLM.
# Every layer is optional and degrades gracefully; the deterministic layers
# (UIA, OCR, image processing) never require a model, and Atlas keeps working
# with no vision model installed at all.
#
# Master switch for the screenshot/OCR/image/VLM perception path. When False,
# the screenshot-based tools report honestly that they are disabled and the
# existing window-only observation path is unaffected.
VISION_ENABLED: bool = True
# Which local vision provider to construct: "ollama" (a local multimodal model
# served by Ollama) or "none" (deterministic perception only, no VLM).
# There is deliberately no cloud computer-use provider.
VISION_PROVIDER: str = "ollama"
# The local multimodal model used for VLM enrichment. It is NOT auto-downloaded;
# configure a model that already exists in the local Ollama instance.
VISION_MODEL: str = "llava:7b"
# Longest edge, in pixels, a screenshot is downscaled to before OCR/VLM use.
# Smaller images mean lower latency and lower VRAM use.
VISION_MAX_IMAGE_SIZE: int = 1280
# Transport timeout (seconds) for one local VLM call.
VISION_TIMEOUT: float = 60.0
# Minimum per-element confidence for a VLM-proposed target to be a candidate.
VISION_CONFIDENCE_THRESHOLD: float = 0.5
# Maximum observation age (seconds) before a visual action referencing it is
# rejected as stale and must re-observe first.
VISION_OBSERVATION_MAX_AGE_SECONDS: float = 30.0
# Maximum number of VLM enrichment calls allowed per request (bounded so a
# request can never loop on inference).
VISION_MAX_VLM_CALLS: int = 2
# Local OCR language(s) for the optional tesseract backend.
VISION_OCR_LANGUAGES: str = "eng"
# Run region-based OCR instead of full-screen OCR whenever a region is known.
VISION_PREFER_REGION_OCR: bool = True

# ---- Experience memory (human feedback -> experience -> retrieval -> planning) ----
# Master switch for the feedback/experience loop. When False, Atlas records no
# experiences and never retrieves them; every other subsystem is unaffected.
ENABLE_EXPERIENCE_MEMORY: bool = True
# Append-only JSONL of durable experience records (survives restart).
EXPERIENCE_STORE_FILE: Path = PROJECT_ROOT / "database" / "experiences.jsonl"
# Where the pending-feedback queue and feedback answers live, so a click of
# Success/Failed is durable even before the experience is evaluated.
FEEDBACK_STORE_FILE: Path = PROJECT_ROOT / "database" / "feedback.jsonl"
# Maximum experiences retrieved for planning context per request (bounded).
EXPERIENCE_RETRIEVAL_LIMIT: int = 4
# Minimum relevance score for a retrieved experience to be offered to planning.
EXPERIENCE_MIN_RELEVANCE: float = 0.35
# Number of "meaningful" evaluated experiences after which periodic pattern
# analysis becomes eligible (never run per-message).
EXPERIENCE_ANALYSIS_THRESHOLD: int = 8
# Cap on the number of stored experiences so the store stays a bounded history.
EXPERIENCE_MAX_RECORDS: int = 5000
# Chroma collection used when experience embeddings are available. The
# experience store works without it (deterministic lexical retrieval fallback).
EXPERIENCE_COLLECTION_NAME: str = "atlas_experience"

# ---- API task history ----
TASK_STORE_FILE: Path = PROJECT_ROOT / "database" / "tasks.json"

# ---- Logging ----
LOG_LEVEL: str = "INFO"
LOG_TO_FILE: bool = False
LOG_FILE: Path = PROJECT_ROOT / "database" / "atlas.log"


