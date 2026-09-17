"""Atlas reasoning layer: LLM proposes meaning, deterministic code executes.

Stages:

* task_interpreter - natural language -> structured Task (models_task)
* task_validator - validate a task against the capability registry
* task_planner - turn a validated task into an execution plan
* verifier - observe/verify executed actions
* interpreter - legacy flat-intent interpretation (fallback/back-compat)
* json_llm - strict structured-output helper
* recovery - bounded, catalog-constrained failure recovery
* diagnostics - structured, redacted per-request pipeline trace
* reasoning_engine - source reasoning loop (understand -> gather -> evaluate -> answer)
* query_router / source_selector - semantic source routing and selection
* evidence_manager - collect, rank, and evaluate evidence per source
* answer_generator - answer synthesis with provenance and honest fallbacks
* self_introspection - machine-readable answers about Atlas itself
"""

from importlib import import_module

_EXPORT_MODULES = {
    "answer_generator": ("Answer", "AnswerGenerator"),
    "evidence_manager": ("EvidenceManager",),
    "interpreter": ("SemanticInterpreter", "classify_category"),
    "json_llm": ("extract_json_object", "safe_reasoning_call"),
    "query_router": ("QueryRouter", "RoutingSignals"),
    "reasoning_engine": ("ReasoningEngine",),
    "reasoning_models": (
        "ConfidenceLevel", "ReasoningDecision", "ReasoningStage", "ReasoningTrace",
        "RequestType", "ResponseMode", "SourcePlan", "SourceType",
    ),
    "self_introspection": ("SelfIntrospection",),
    "source_selector": ("SourceSelector",),
    "task_interpreter": ("SemanticTaskInterpreter", "task_to_intent_shim"),
    "task_planner": ("TaskPlanner",),
    "task_validator": ("TaskValidationResult", "TaskValidator"),
    "verifier": ("TaskVerifier", "VerificationOutcome"),
}


def __getattr__(name):
    for module, names in _EXPORT_MODULES.items():
        if name in names:
            value = getattr(import_module(f"reasoning.{module}"), name)
            globals()[name] = value
            return value
    raise AttributeError(name)

__all__ = [
    "Answer",
    "AnswerGenerator",
    "ConfidenceLevel",
    "EvidenceManager",
    "QueryRouter",
    "ReasoningDecision",
    "ReasoningEngine",
    "ReasoningStage",
    "ReasoningTrace",
    "RequestType",
    "ResponseMode",
    "RoutingSignals",
    "SelfIntrospection",
    "SemanticInterpreter",
    "SemanticTaskInterpreter",
    "SourcePlan",
    "SourceSelector",
    "SourceType",
    "TaskPlanner",
    "TaskValidationResult",
    "TaskValidator",
    "TaskVerifier",
    "VerificationOutcome",
    "classify_category",
    "extract_json_object",
    "safe_reasoning_call",
    "task_to_intent_shim",
]
