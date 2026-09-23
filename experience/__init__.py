"""Atlas experience memory: human feedback -> experience -> retrieval -> improvement.

This package is the persistent learning loop layered on top of the existing
Atlas runtime. It deliberately does **not** replace or duplicate any working
subsystem:

* interpretation stays in :mod:`reasoning.task_interpreter` / :mod:`reasoning.intent_engine`,
* planning stays in :mod:`reasoning.task_planner`,
* execution/verification stay in :mod:`executor` / :mod:`reasoning.verifier`,
* factual RAG stays in :mod:`vector_store` / :mod:`knowledge_search`,
* conversation storage stays in :mod:`memory`.

What this package adds is a *fifth*, separate memory — experience — plus the
feedback capture, retrieval, ranking, and planning-context rendering that turns
it into behavioural improvement. It is model-agnostic (the model is only ever
given rendered context), local-first (no cloud calls), and it never modifies
source code or model weights.

Modules:

* :mod:`experience.models`  — the record schema and sanitization rules.
* :mod:`experience.store`   — durable append-only JSONL storage.
* :mod:`experience.memory`  — bounded retrieval and ranking.
* :mod:`experience.builder` — execution evidence -> experience record.
* :mod:`experience.service` — the small orchestration façade used by Brain/API.
"""

from experience.builder import build_experience, evaluate_completion
from experience.memory import (
    ExperienceContext,
    ExperienceEmbedder,
    ExperienceMemory,
    RetrievedExperience,
    render_experience_context,
)
from experience.models import (
    COMPLETION_CRITERIA,
    EXPERIENCE_STATES,
    EXPERIENCE_TYPES,
    FAILURE_CATEGORIES,
    FAILURE_CATEGORY_LABELS,
    CompletionChecks,
    Experience,
)
from experience.service import ExperienceService
from experience.store import ExperienceStore, FeedbackEvent

__all__ = [
    "COMPLETION_CRITERIA",
    "CompletionChecks",
    "EXPERIENCE_STATES",
    "EXPERIENCE_TYPES",
    "Experience",
    "ExperienceContext",
    "ExperienceEmbedder",
    "ExperienceMemory",
    "ExperienceService",
    "ExperienceStore",
    "FAILURE_CATEGORIES",
    "FAILURE_CATEGORY_LABELS",
    "FeedbackEvent",
    "RetrievedExperience",
    "build_experience",
    "evaluate_completion",
    "render_experience_context",
]
