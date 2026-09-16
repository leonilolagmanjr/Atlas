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
"""

from reasoning.interpreter import SemanticInterpreter, classify_category
from reasoning.json_llm import extract_json_object, safe_reasoning_call
from reasoning.task_interpreter import SemanticTaskInterpreter, task_to_intent_shim
from reasoning.task_planner import TaskPlanner
from reasoning.task_validator import TaskValidationResult, TaskValidator
from reasoning.verifier import TaskVerifier, VerificationOutcome
__all__ = [
    "SemanticInterpreter",
    "SemanticTaskInterpreter",
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
