# Atlas Roadmap

**Project:** Atlas AI Operating System

---

# Mission

Atlas is a modular, local-first AI operating system foundation.

Atlas is not a chatbot.

Atlas is designed to become an extensible AI platform capable of reasoning, planning, learning, remembering, and using tools.

## Current status (2026-09-16)

Atlas is currently at V3.2: a local retrieval runtime with deterministic planning, persistent conversation memory, permission-aware tool routing, and a FastAPI + React control room. The codebase is not yet a fully autonomous computer agent, but the foundation is implemented and documented.

The currently verified runtime includes:

- local PDF ingestion and incremental indexing
- hybrid retrieval with semantic, keyword, and metadata evidence
- permission-aware tool registry and explicit executable launch gating
- read-only Windows system, filesystem, process, and application inspection
- session memory persistence and CLI commands for conversation lifecycle
- FastAPI adapter and browser-based control-room UI
- durable API task history with explicit interruption handling on restart
- task-ID-bound approval and denial for API actions
- data-driven tool knowledge, capability discovery, and read-only PowerShell execution

## Local-first computer-agent migration

The repository audit is complete. The current implementation is a local RAG
runtime, not yet a computer-control agent. The incremental migration is:

1. Execution context: task lifecycle, tool calls, permissions, observations, and verification state.
2. Tool contract: metadata, schemas, validation, structured results, registry, and routing.
3. Permission engine: risk levels and SAFE, CONFIRM, and AUTONOMOUS modes.
4. Safe computer tools: read-only filesystem, process, application, and system inspection.
5. Controlled PowerShell execution with command validation and structured output.
6. Observer and verifier with evidence-based success reporting.
7. Internet search, webpage retrieval, provenance, controlled downloads, and package management.
8. Multi-tool planning and recovery, followed later by GUI, voice, and vision adapters.

The first execution-runtime hardening slice is now implemented: API task
snapshots persist under `database/tasks.json`, and approval actions are bound
to the matching execution task. In-flight tasks are reported as interrupted
after an API restart rather than appearing resumable without their runtime
context.

The detailed audit, gap analysis, file map, and Phase 1 plan are in
`ARCHITECTURE_ASSESSMENT.md`.


---

# Guiding Principles

Every version of Atlas should improve one or more of the following:

* Intelligence
* Reliability
* Performance
* Extensibility
* Autonomy

Features should never compromise architecture.

---

# Version 2 — Foundation ✅

## Goal

Build a reliable production-quality local RAG assistant.

### Completed

* Local LLM (Ollama)
* ChromaDB
* Chunked document retrieval
* Incremental indexing
* Prompt management
* Logging
* Configuration
* Modular architecture

### Success Criteria

* Stable
* Fast
* Maintainable
* Easy to extend

---

# Version 3 — Brain

## Goal

Atlas becomes the orchestrator.

The LLM is no longer responsible for deciding what to do.

Atlas decides.

### New Components

Brain

Planner

Execution Context

Executor

Tool Manager

Tool Registry


### Capabilities

Current V3.2 foundation:

* Deterministic execution plans
* Sequential plan execution
* Shared execution context
* Planner and executor module boundaries
* Conversation memory and session persistence
* Tool registry and permission-aware routing
* Local FastAPI adapter and control-room frontend

Future V3 capabilities:

* Intent recognition
* Tool selection
* Query rewriting
* Multi-search retrieval
* Confidence analysis
* Retry strategy

### Success Criteria

Atlas can decide:

* Should I search?
* Should I retry?
* Should I answer?
* Which tool should I use?

---

# Version 4 — Memory

## Goal

Atlas remembers.

### Components

Conversation Memory

Session Memory

Long-Term Memory

### Capabilities

Remember:

* preferences
* projects
* goals
* recurring tasks

Forget outdated information.

Store useful long-term facts.

### Success Criteria

Atlas remembers previous conversations and persistent user preferences.

---

# Version 5 — Tools

## Goal

Atlas interacts with the computer.

### Built-in Tools

Knowledge

Filesystem

Calculator

Python Runner

Git

Markdown

CSV

JSON

Terminal

### Future Tools

Email

Calendar

Weather

Database

Image

Audio

Video

### Success Criteria

Atlas chooses tools automatically.

---

# Version 6 — Planning

## Goal

Atlas solves complex tasks.

### Capabilities

Task decomposition

Planning

Execution graphs

Dependencies

Progress tracking

Recovery from failures

### Example

User:

Compare my resume to this job description.

Atlas:

1 Read resume

2 Read job description

3 Extract skills

4 Compare

5 Generate report

---

# Version 7 — Reflection

## Goal

Atlas evaluates itself.

### Capabilities

Confidence scoring

Answer verification

Retry failed reasoning

Detect hallucinations

Improve retrieval

### Success Criteria

Atlas attempts to improve weak answers before responding.

---

# Version 8 — Learning

## Goal

Atlas continuously improves.

### Capabilities

Knowledge ingestion

Automatic tagging

Automatic categorization

Summaries

Relationship graphs

Learning from corrections

Learning from feedback

### Success Criteria

Atlas builds a growing knowledge base.

---

# Version 9 — Vision

## Goal

Atlas understands images.

### Capabilities

OCR

Charts

Diagrams

Screenshots

UI understanding

Image question answering

---

# Version 10 — Voice

## Goal

Natural conversations.

### Components

Speech-to-Text

Text-to-Speech

Wake word

Streaming responses

---

# Version 11 — Desktop Assistant

## Goal

Atlas controls the computer.

### Capabilities

Open applications

Read files

Move files

Edit documents

Automate workflows

Use keyboard

Use mouse

Interact with windows

---

# Version 12 — Internet

## Goal

Real-time knowledge.

### Capabilities

Web Search

News

Wikipedia

APIs

RSS

Documentation

### Rules

Always distinguish between:

Local knowledge

Internet knowledge

Reasoning

---

# Version 13 — Coding Assistant

## Goal

Become a software engineering partner.

### Capabilities

Read repositories

Write code

Run tests

Debug

Refactor

Generate documentation

Review pull requests

---

# Version 14 — Multi-Agent System

## Goal

Atlas coordinates specialist agents.

### Example

Brain

↓

Planner Agent

↓

Research Agent

↓

Coding Agent

↓

Knowledge Agent

↓

Reflection Agent

↓

Synthesizer

↓

Final Response

Each agent has a single responsibility.

---

# Version 15 — Autonomous Workspace

## Goal

Atlas completes long-running projects.

### Capabilities

Task queue

Background jobs

Notifications

Scheduling

Checkpoints

Project memory

Recovery

---

# Long-Term Vision

Atlas should become a modular AI operating system that can coordinate reasoning, retrieval, memory, planning, and specialized tools while remaining model-agnostic.

Replacing the language model, embedding model, vector database, or tools should require minimal changes to the overall architecture.

Atlas should always remain:

* Modular
* Extensible
* Reliable
* Local-first
* Maintainable
