# Atlas Architecture
**Version:** V3.2 (Brain + Intent Classification + Deterministic Planning + Staged Hybrid Retrieval + Permissioned Computer Tool Runtime)

The verified implementation status and migration plan are maintained in
[ARCHITECTURE_ASSESSMENT.md](ARCHITECTURE_ASSESSMENT.md). This document defines
target boundaries as well as the current implemented runtime. Observer/verifier,
downloads, browser automation, durable live checkpoints, and GUI/voice/vision
capabilities are not implied to exist.

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

## Command intelligence pipeline (implemented)

Natural-language requests flow through an explicit LLM-reasoning /
deterministic-execution separation:

```text
USER REQUEST
     |
QWEN SEMANTIC INTERPRETER (deterministic fast path when confident)
     |
STRUCTURED INTENT (intent, action, target, topic, tone, style, length, sort, destination, confidence)
     |
TASK PLANNER (deterministic; catalog-constrained LLM fallback)
     |
CAPABILITY CATALOG (registered tool names + parameter schemas)
     |
PLAN VALIDATION
     |
DETERMINISTIC EXECUTOR
     |
STRUCTURED RESULT
     |
QWEN RECOVERY (bounded replanning on recoverable failure)  ->  USER RESPONSE
```

Invariant: **the LLM never executes anything.** It determines *what* should
happen; the deterministic layer determines *how* it is safely executed. The LLM
receives a capability catalog and may only select from registered capabilities.
Plans referencing unknown capabilities, missing tools, or malformed parameter
blocks are rejected before execution.

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
* Rule-based intent classification (no LLM)
* Deterministic planning
* Sequential plan execution
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
config.py              Centralized runtime configuration
logger.py              Central logging configuration
llm.py                 LLM provider wrapper and Ollama call boundary
intent_classifier.py   Rule-based coarse classification for knowledge plan selection
reasoning/             LLM reasoning layer (semantic interpreter, prompts, recovery, diagnostics)

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

* `interpreter.py` — turns free text into a compositional `StructuredIntent`
  (deterministic fast path + Qwen fallback), with clarification policy and
  conversational context merging.
* `prompts.py` — small, single-purpose prompts for interpreter, planner,
  recovery, and content generation.
* `json_llm.py` — strict structured-output helper with JSON extraction and repair.
* `recovery.py` — bounded, catalog-constrained recovery for failed steps.
* `diagnostics.py` — structured per-request trace with redaction.

This package never imports a concrete provider and never executes commands.

---

## planner.py
Responsible only for creating an execution plan.

The planner is deterministic by default: it maps a `StructuredIntent` onto the
smallest valid plan. It consults the LLM only for compositional requests the
rule table cannot express, and then only to select from the capability catalog.

---

## executor.py

Responsible for executing an `ExecutionPlan` step-by-step and updating the
shared `ExecutionContext`.

It owns retrieval, LLM, and explicitly planned tool execution for the current
plan actions. Tool actions are routed through `ToolRouter` before execution.

Future executor versions may add retries, branching, parallelism, and
conditional steps without moving those responsibilities back into Brain.

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
