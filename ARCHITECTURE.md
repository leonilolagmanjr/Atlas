# Atlas Architecture
**Version:** V3.4 (Semantic Task Understanding + Validated Task IR + Dynamic Planning + Verification)

The verified implementation status and migration plan are maintained in
[ARCHITECTURE_ASSESSMENT.md](ARCHITECTURE_ASSESSMENT.md). This document defines
target boundaries as well as the current implemented runtime. Downloads, browser
automation, durable live checkpoints, and GUI/voice/vision capabilities are not
implied to exist.

---

# Vision

Atlas is a modular AI operating system.

Atlas is **not** a chatbot.

Atlas is designed to become an intelligent software platform capable of:

* Reasoning
* Planning
* Remembering
* Searching knowledge
* Using tools
* Learning from documents
* Coordinating multiple AI capabilities

The language model (LLM) is **not Atlas**.

The LLM is only one reasoning component inside Atlas.

Atlas owns the overall intelligence, workflow, and decision-making process.

---

# Core Philosophy

The project follows these principles:

1. Modularity
2. Single Responsibility
3. Extensibility
4. Reliability
5. Maintainability
6. Explainability
7. Performance
8. Local-first

Every module should have one clearly defined responsibility.

Avoid monolithic files.

Avoid tightly coupled code.

Favor composition over inheritance where appropriate.

---

# System Layers

Atlas is divided into independent layers.

```
User -> Brain -> Planner -> Tool Router -> Permission Engine
							   -> Tool Discovery -> Tool Knowledge
							   -> Executor -> Observer -> Verifier
								   |             |
					Computer / Internet / Knowledge runtimes
```

## Semantic task-understanding pipeline (implemented)

Natural-language requests flow through an explicit LLM-reasoning /
deterministic-execution separation:

```text
USER REQUEST
     |
SEMANTIC TASK INTERPRETER (Qwen; deterministic fast path when confident)
     |
TASK IR (task_type, goal, actions[], entities, constraints, confidence)
     |
TASK VALIDATOR (capability existence, required params, types, deps, refs, risk)
     |
DYNAMIC TASK PLANNER (dependency order + $variable output references)
     |
CAPABILITY REGISTRY (registered names + parameter schemas + risk + verification)
     |
PLAN VALIDATION
     |
DETERMINISTIC EXECUTOR
     |
OBSERVATION / VERIFICATION (verified | unverified | failed)
     |
QWEN RECOVERY / bounded replanning (only on recoverable failure)  ->  USER RESPONSE
```

Invariant: **the LLM never executes anything.** It determines *what* should
happen (as a structured Task); the deterministic layer validates it and
determines *how* it is safely executed. The LLM receives a capability catalog
and may only select from registered capabilities. Tasks referencing unknown
capabilities, missing required parameters, wrong parameter types, broken
dependencies, or dangling `$variable` references are rejected before execution.

The Task IR also preserves the distinction the old flat intent could not: a
modifier like `about cars` is a content *topic*, never part of the destination
application name.

Execution categories are coarse and extensible: `CONVERSATION`,
`KNOWLEDGE_QUERY`, `CREATIVE_GENERATION`, `APPLICATION_CONTROL`,
`FILE_OPERATION`, `WEB_SEARCH`, `WEB_NAVIGATION`, `SYSTEM_OPERATION`,
`MULTI_STEP_TASK`.

For the current PowerShell slice, the planning path is:

```text
User request -> IntentClassifier -> Planner -> ToolDiscovery
			 -> PowerShell knowledge record -> Validator -> PermissionEngine
			 -> Executor -> structured result interpreter -> User
```

PowerShell knowledge is data-driven and stored in
`tools/powershell_commands.json`. Only documented read-only commands are
currently executable.

## Memory subsystem (implemented)

Atlas now includes a first-class **Memory** subsystem as a dedicated package: `memory/`.

Responsibilities:
- Conversation Sessions (create/open/rename/list/delete/archive)
- Short-term history window (configurable `MAX_RETAINED_MESSAGES`)
- Persistence to disk under `memory/sessions/<session-id>/`
- Context building for prompt construction via `{conversation_history}`

The Brain/Executor never manipulate storage directly—Executor requests
conversation history from `MemoryManager` and persists user/assistant turns.


Only the required layers should execute for each request.

---

# Current Version (V3.2 + local web console)

Atlas currently contains:

* Local RAG
* Incremental document indexing
* ChromaDB vector storage
* Ollama integration
* Prompt management
* Logging
* Configuration management
* Brain orchestration
* LLM-first semantic task interpretation with a deterministic fast path
* Structured `Task` / `TaskAction` intermediate representation
* Capability registry with parameter schemas, required parameters, and risk
* Task validation before planning
* Dynamic, dependency-ordered planning with `$variable` output references
* Deterministic planning and sequential plan execution
* Post-action observation and honest verification (verified / unverified / failed)
* Bounded, plan-wide replanning with an anti-loop budget
* Rule-based intent classification (no LLM) retained for legacy knowledge plan selection
* Shared execution context models
* Conversation memory and persisted session lifecycle
* Tool registry, permission engine, and permission-aware router
* Data-driven tool knowledge and capability discovery
* Read-only Windows filesystem, process, system, and application inspection
* Explicit executable launch and generic named-application resolution with confirmation pause/resume
* Confirmation-gated application text entry
* Validated, read-only PowerShell execution with structured result interpretation
* Read-only public web search and page retrieval with provenance and SSRF checks
* Durable API task snapshots with task-bound approval and interruption handling
* FastAPI adapter and React/Vite control room
This is the verified current implementation baseline. Future versions will build upon this foundation, but the runtime is already a working local-first AI assistant with explicit tool boundaries and approval gating.

---

# Implemented Slices and Future Versions
## Version 3 — Brain (implemented)

Brain, Planner, Tool Manager, Tool Registry, Tool Router, Permission Engine.

---

## Version 4 — Memory (conversation/session memory implemented)

Conversation Memory and persisted session lifecycle are implemented.
Long-Term Memory and Reflection remain planned.

---

## Version 5 — Tools (partially implemented)

Implemented: Knowledge retrieval, Filesystem inspection, Web Search and page
retrieval, Windows process/system/application inspection, generic named
application launch, application text entry, and read-only PowerShell.
Planned: Python Tool, Calculator, and mutating tools.

---

## Version 6 — Planning (deterministic planning implemented)

Deterministic plans and sequential multi-step execution are implemented.
Task decomposition, execution graphs, dependencies, and recovery remain planned.

---

## Version 7 — Interfaces (planned)

Vision
Voice
GUI automation
Multi-agent collaboration
---

# Project Structure

```
Atlas/

api.py                 FastAPI adapter for health, system, tools, tasks, and approval flows
atlas.py               CLI entry point and REPL loop
brain.py               Request orchestration and execution state tracking
executor.py            Sequential plan runner and tool execution boundary
planner.py             Deterministic plan creation and explicit action intent routing
models.py              Shared execution context, task, and plan models
models_task.py         Task/TaskAction intermediate representation (the Task IR)
config.py              Centralized runtime configuration
logger.py              Central logging configuration
llm.py                 LLM provider wrapper and Ollama call boundary
intent_classifier.py   Rule-based coarse classification for legacy knowledge plan selection
reasoning/             LLM reasoning layer (task interpreter, validator, planner, verifier, prompts, recovery, diagnostics)

knowledge/             User knowledge base (PDFs)
memory/                Conversation sessions, models, storage, and context builder
prompts/               Retrieval and system prompt templates
providers/             Provider interface and concrete provider implementations
tools/                 Tool contracts, registry, router, and permission policy
computer/              Read-only Windows system and app inspection tools
frontend/              React/Vite control room

database/              ChromaDB data, index metadata, and generated runtime state
logs/                  Optional application logs
cache/                 Reserved for future performance/local caches
```

Folders should remain modular.

---

# Module Responsibilities

## atlas.py

Entry point only.

Responsibilities:

* startup
* initialization
* orchestration
* chat loop

Should NOT contain business logic.

---

## llm.py

Responsible ONLY for communicating with the language model.

It should never:

* search documents
* read files
* choose tools
* manage memory

Replacing the LLM should require changing only this module.

---

## config.py

Contains every configurable value.

No hardcoded configuration elsewhere.

---

## logger.py

Central logging configuration.

No module should configure logging independently.

---

## document_loader.py

Reads knowledge documents.

Future support:

* PDF
* TXT
* DOCX
* Markdown
* HTML

Should never generate embeddings.

---

## chunker.py

Splits documents into semantic chunks.

Should not communicate with ChromaDB.

Should not communicate with Ollama.

---

## vector_store.py

Responsible only for vector database operations.

Should not read files.

Should not call the LLM.

Should not perform planning.

---

## knowledge_search.py

Responsible only for retrieval.

Should return:

* chunks
* metadata
* confidence

No indexing.

---

## models.py

Contains shared dataclasses for execution plans, execution steps, planner
decisions, evidence, retrieval results, and request execution context.

Dataclasses should live here when multiple modules need the same model.

---

## reasoning/

Owns the LLM-facing reasoning stages. It contains:

* `task_interpreter.py` — turns free text into a structured `Task` (Qwen-first
  with a deterministic, semantic fast path and fallback). This is the primary
  interpretation path.
* `task_validator.py` — validates a task against the capability registry
  (existence, required parameters, types, dependencies, references, risk).
* `task_planner.py` — turns a validated task into an execution plan with
  dependency ordering and `$variable` output references.
* `verifier.py` — deterministic post-action observation/verification that never
  claims unconfirmed success.
* `interpreter.py` — the legacy flat-intent interpreter, retained for
  back-compatibility and as a shared `classify_category` signal.
* `prompts.py` — small, single-purpose prompts for the task interpreter, planner,
  recovery, and content generation.
* `json_llm.py` — strict structured-output helper with JSON extraction and repair.
* `recovery.py` — bounded, catalog-constrained recovery for failed steps.
* `diagnostics.py` — structured per-request trace (task, validation, plan,
  execution, verification) with redaction.

This package never imports a concrete provider and never executes commands.

---

## planner.py
Responsible only for creating an execution plan.

The planner is deterministic by default: it maps a `StructuredIntent` onto the
smallest valid plan (used for legacy knowledge/compare/summarize workflows and as
an informational fallback). For computer-control requests, the `reasoning/`
`TaskPlanner` maps a validated `Task` onto a dependency-ordered plan.

---

## executor.py
Responsible for executing an `ExecutionPlan` step-by-step and updating the
shared `ExecutionContext`.

It owns retrieval, LLM, and explicitly planned tool execution for the current
plan actions. Tool actions are routed through `ToolRouter` before execution.
After each successful tool call it records an honest observation/verification
(`reasoning.verifier`) and publishes named outputs for later steps. On a
recoverable failure it performs one bounded, catalog-constrained recovery attempt
per step, capped by a plan-wide budget (`MAX_PLAN_RECOVERIES`) to prevent loops.

Future executor versions may add branching and parallelism without moving those
responsibilities back into Brain.

---

## indexer.py

Responsible for:

* hashing
* detecting changes
* incremental indexing

Should not answer user questions.

---

# Brain

The Brain is the central orchestrator.

Responsibilities:

* receive user requests
* create execution context
* request an execution plan
* pass the plan to the executor
* return the final response

Future responsibilities:

* understand user intent
* choose tools
* coordinate memory
* coordinate reasoning

The Brain is the "operating system" of Atlas.

---

# Tool Manager

Atlas tools must be independent.

Examples:

Knowledge

Filesystem

Calculator

Python

Web

Email

Git

Image

The Tool Registry exposes tool metadata and schemas. The Tool Router resolves
structured plan steps and evaluates the Permission Engine before invocation.
The Executor pauses plans that require confirmation. The current registry
contains read-only computer inspection and explicit executable launch tools.

---

# Memory

Conversation memory is implemented. The following layers remain planned:

Conversation Memory

Session Memory

Long-Term Memory

Each layer has different responsibilities.

## Web API and frontend

`api.py` is a thin local FastAPI adapter over the existing Brain, tools, and
computer runtime. It exposes health, system, tool, tool-knowledge,
tool-discovery, application, and durable task endpoints. `frontend/` is a
React/Vite control room that polls task state, renders approval boundaries, and
displays only data returned by the API. It does not implement planning or tool
execution.

Read-only web search, bounded public page retrieval, and validated read-only
PowerShell execution are implemented. Web downloads, browser automation,
durable live execution checkpoints, and GUI/voice/vision automation are not
implemented in the current source.

---

# Design Principles

Atlas should never rely on:

* giant if-else chains
* duplicated code
* hidden side effects
* circular imports

Atlas should prefer:

* reusable modules
* dependency injection where appropriate
* configuration over hardcoding
* explicit interfaces

---

# Error Handling

Atlas should never crash because of:

* missing files
* corrupted PDFs
* database failures
* Ollama being offline
* embedding failures

Recover gracefully whenever possible.

---

# Logging

Every important operation should be logged.

Examples:

Startup

Indexing

Document updates

Search

Warnings

Errors

Performance

---

# Coding Standards

Use:

* pathlib
* type hints
* docstrings
* dataclasses where appropriate
* logging
* descriptive names

Avoid:

* wildcard imports
* duplicated code
* unnecessary globals

---

# Decision Rule

When adding a new feature, ask:

1. Does this belong in an existing module?

If no:

Create a new module.

2. Does this module now have more than one responsibility?

If yes:

Split it.

3. Would replacing one subsystem require changing unrelated files?

If yes:

Reduce coupling.

---

# Long-Term Goal

Atlas should eventually become a fully modular AI operating system capable of coordinating reasoning, memory, retrieval, tools, planning, and specialized agents while remaining maintainable, extensible, and model-agnostic.

The architecture should allow replacing the language model, vector database, embedding model, or tools with minimal changes to the rest of the system.
