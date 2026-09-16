# Atlas Architecture

**Version:** V3.2 (Planner + Intent Classification + Staged Hybrid Retrieval)

The verified implementation status and migration plan are maintained in
[ARCHITECTURE_ASSESSMENT.md](ARCHITECTURE_ASSESSMENT.md). This document defines
target boundaries; future computer-agent features are not implied to exist.

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
* Deterministic planning
* Sequential plan execution
* Shared execution context models
* Conversation memory and persisted session lifecycle
* Tool registry, permission engine, and permission-aware router
* Read-only Windows filesystem, process, system, and application inspection
* Explicit executable launch with confirmation pause/resume
* FastAPI adapter and React/Vite control room

This is the verified current implementation baseline. Future versions will build upon this foundation, but the runtime is already a working local-first AI assistant with explicit tool boundaries and approval gating.

---

# Future Versions

## Version 3

Brain

Planner

Tool Manager

---

## Version 4

Conversation Memory

Long-Term Memory

Reflection

---

## Version 5

Filesystem Tool

Python Tool

Web Search

Calculator

---

## Version 6

Planning

Multi-step execution

Task decomposition

---

## Version 7

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
intent_classifier.py   Rule-based request classification for plan selection

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

## planner.py

Responsible only for creating an execution plan.

The current planner is deterministic and does not call the LLM.

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
computer runtime. It exposes health, system, tool, application, and in-memory
task endpoints. `frontend/` is a React/Vite control room that polls task state,
renders approval boundaries, and displays only data returned by the API. It
does not implement planning or tool execution.

Internet search, webpage retrieval, downloads, PowerShell execution, and GUI
automation are not implemented in the current source.

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
