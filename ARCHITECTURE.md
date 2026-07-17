# Atlas Architecture

**Version:** V3.2 (Planner + Intent Classification + Staged Hybrid Retrieval)

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
User

↓

Brain

↓

Planner

↓

Memory (subsystem)

↓

Knowledge

↓

Tool Manager

↓

Tools

↓

LLM

↓

Response
```

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

# Current Version (V3.1)

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

Future versions will build upon this foundation.

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

atlas.py
config.py
logger.py
llm.py

knowledge/
memory/
tools/
prompts/

database/
logs/
cache/
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

It owns retrieval and LLM execution for the current plan actions.

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

# Tool Manager (Future)

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

The Tool Manager chooses which tool executes.

---

# Memory (Future)

Memory is divided into:

Conversation Memory

Session Memory

Long-Term Memory

Each layer has different responsibilities.

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
