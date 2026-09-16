# Atlas Architecture Assessment

**Audit date:** 2026-09-16 (updated for the V3.4 semantic task pipeline)
**Audited version:** V3.4 codebase

## Current architecture

The implemented runtime is a local-first assistant with LLM interpretation
deterministic orchestration:

```text
CLI / React control room
        |
      Brain
        |-- Reasoning package (LLM proposes meaning, deterministic code executes)
        |     Semantic Task Interpreter -> Task IR -> Task Validator -> Task Planner
        |-- Executor -> ToolRouter -> PermissionEngine -> Computer / PowerShell / Web tools
        |-- Verifier (observation / verification)
        |-- Recovery (bounded, catalog-constrained replanning)
        |-- Knowledge retrieval -> ChromaDB
        |-- Evidence-gated Ollama response
        |-- Conversation Memory
```

- `atlas.py` initializes logging, indexing, vector storage, memory, Brain, and the CLI loop.
- `brain.py` creates one `ExecutionContext` and runs interpret -> validate -> plan -> execute -> verify.
- `reasoning/task_interpreter.py` turns free text into a structured `Task`.
- `reasoning/task_validator.py` validates the task against the capability registry.
- `reasoning/task_planner.py` builds a dependency-ordered plan with `$variable` references.
- `planner.py` retains deterministic legacy plans (knowledge, compare, summarize, informational fallback).
- `executor.py` runs retrieval, LLM, and routed tool steps, records verification, and applies bounded recovery.
- `knowledge_search.py` owns staged semantic, keyword, and metadata retrieval with diagnostics.
- `indexer.py`, `document_loader.py`, `chunker.py`, and `vector_store.py` form the local document pipeline.
- `memory/` provides persisted conversational sessions and prompt history.
- `providers/` provides an Ollama provider interface, but provider selection is still hard-coded by `llm.py`.
- `tools/knowledge.py`, `tools/discovery.py`, and `tools/powershell_commands.json` provide data-driven tool knowledge and discovery.
- `web.py` provides read-only public search, direct YouTube search, bounded page retrieval, provenance, and SSRF checks.
- `computer/launch.py` resolves arbitrary installed application names; `computer/text_entry.py` supports confirmation-gated Windows text entry.
- `launch_atlas.bat` and `launch_atlas.ps1` start the API, frontend, and browser together.

## What works
- CLI conversation and session commands.
- Ollama-backed local generation.
- LLM-first semantic task interpretation with a deterministic fast path.
- Structured `Task` / `TaskAction` intermediate representation.
- Capability registry with parameter schemas, required parameters, risk, and verification contracts.
- Task validation before any planning or execution.
- Dynamic, dependency-ordered planning with `$variable` output references.
- Post-action observation and honest verification (verified / unverified / failed).
- Bounded, plan-wide replanning with an anti-loop budget.
- PDF indexing with incremental file hashing.
- ChromaDB storage and hybrid retrieval.
- Retrieval confidence gating that can skip the LLM when evidence is weak.
- Deterministic plan execution.
- Persisted conversation memory with session management.
- Centralized configuration and logging.
- Read-only PowerShell inspection with command validation and structured results.
- Generic named application resolution and confirmation-gated text entry.
- Public web search with YouTube video links, thumbnails, source attribution, and frontend result cards.

## Partial or incomplete

- `ExecutionContext` tracks task lifecycle, tool calls, permissions, observations, web sources, verification results, and JSON-safe snapshots. The observer/verifier is now implemented for the plannable capability set; capabilities without a verification contract are reported as `unverified` rather than assumed successful. API task snapshots persist durably, while live execution contexts remain process-local.
- The plan/step model supports existing retrieval actions and explicit `invoke_tool` steps routed through `ToolRouter`.
- The provider abstraction exists, while the public LLM facade still constructs `OllamaProvider` directly.
- Memory is conversation/session memory; long-term memory, provenance, confidence, and retrieval are not implemented.
- Error handling records failed plan steps but has no recovery policy or user cancellation state. Restart recovery explicitly marks pending, running, and approval-paused API tasks as interrupted failures.
- Filesystem, process, system, installed-application inspection, PowerShell, generic named launch, and text-entry tools exist behind the tool contracts. Consequential actions use the permission-aware router and executor.
- Automated contract tests cover the runtime, computer tools, PowerShell, web search, YouTube parsing, application resolution, text entry, APIs, and frontend build.

## Planned or missing
- Downloads, browser automation, and richer page extraction.
- Web downloads, browser interaction, richer page extraction, and full source trust scoring.
- Approval-aware multi-tool planning and failure recovery.
- GUI, voice, and vision interfaces.

## Reusable architecture

The existing Brain/Planner/Executor boundary, shared dataclasses, evidence model, provider interface, dependency injection points, memory facade, and retrieval diagnostics should be preserved. Computer and internet capabilities should be added as tools behind contracts rather than embedded in `Brain` or the LLM provider.

## Gap analysis

| Vision requirement | Current state | Migration direction |
| --- | --- | --- |
| Structured execution context | Task lifecycle, JSON-safe snapshots, durable API task history implemented | Add durable execution checkpoints and richer verification evidence |
| Standardized tools | Tool contract, result model, registry, router, and executor integration implemented | Add more runtimes and schema-level validation |
| Permissions | SAFE, CONFIRM, AUTONOMOUS modes and task-bound confirmation pause implemented | Persist per-task approvals and add richer policy configuration |
| Computer control | Inspection, generic named launch, and confirmation-gated text entry routed through Executor | Add richer window targeting and controlled document/file writes |
| Terminal safety | Read-only validated PowerShell with trusted command knowledge and structured results | Add AST-aware validation only if mutation is enabled later |
| Observation and verification | Implemented for the plannable capability set (`reasoning/verifier.py`); unverifiable capabilities reported honestly | Extend verification contracts to more capabilities |
| Internet access | Read-only web search/page retrieval with direct YouTube results, thumbnails, provenance, and SSRF checks | Add browser automation, downloads with approval, and source trust scoring |
| Testing | 55 Python tests plus frontend build and live smoke checks | Add browser automation and broader retrieval/LLM integration tests |

## Proposed target architecture

```text
User -> Brain -> Planner -> Tool Router -> Permission Engine
                                      -> Executor -> Observer -> Verifier
                                           |             |
                         Computer / Internet / Knowledge runtimes
```

`Brain` owns request lifecycle orchestration. `Planner` produces structured steps. The `ToolRegistry` discovers tools and the `ToolRouter` resolves a plan step to a tool. The permission engine evaluates before execution. The executor invokes only validated, permitted tools. The observer records actual results, and the verifier determines whether the requested outcome is supported by evidence.

## Migration strategy

1. Preserve the existing retrieval plan and CLI behavior while introducing additive models.
2. Add contract tests before wiring new tools into the default runtime.
3. Introduce a registry and route explicit tool steps through it; existing knowledge retrieval remains compatible.
4. Add permission evaluation before any mutating computer capability.
5. Add read-only Windows tools, then controlled writes, with verification for each action. Read-only inspection and explicit-path launch are now implemented and routed.
6. Add internet tools with source provenance and download approval before package installation.
7. Keep GUI, voice, and vision as adapters over the same planning and tool contracts.

## Historical Phase 1 implementation plan

The audit and initial contract-hardening phase delivered:

- This assessment and an accurate status in the project documentation.
- A serializable execution context with explicit task, step, tool, permission, observation, and verification state.
- A tool contract and structured tool result model, without enabling OS execution yet.
- A permission vocabulary and policy decision model, defaulting to deny when no policy is supplied.
- A small test suite covering context serialization, tool validation, and permission decisions.

This phase is complete and has been superseded by the current runtime described above.

## Current handoff for the next agent

### Verified entry points

- One-click local runtime: double-click `launch_atlas.bat`.
- API: `\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000`.
- Frontend: `Set-Location frontend; npm run dev`.
- Full tests: `\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- Frontend build: `Set-Location frontend; npm run build`.

### Current request paths

- Knowledge questions: retrieval through ChromaDB, then evidence-gated Ollama response.
- Computer inspection: deterministic `COMPUTER` intent to validated PowerShell or existing computer tools.
- Named applications: generic resolver through PATH, install folders, and Windows uninstall registry metadata.
- Text entry: `WRITE_APPLICATION` intent to `applications.write_text`; medium-risk approval is required before Win32 focus and clipboard paste.
- Web research: `WEB_SEARCH` intent to `web.search`; explicit YouTube searches use direct `ytInitialData` parsing and return only video watch URLs with thumbnails.
- API task state: persisted snapshots in `database/tasks.json`; live execution contexts remain process-local.

### Next recommended work

1. Add browser/API integration tests for web thumbnail cards and approval flows.
2. Add a source-aware page summarization step using `web.fetch`, keeping page text untrusted.
3. Add durable execution checkpoints and cancellation before multi-step autonomy.
4. Add observation/verifier components for application launch and text entry.
5. Add YouTube channel resolution and pagination for reliable “latest videos” lists.
6. Add downloads only with explicit approval, file-type/size limits, and provenance.
7. Avoid enabling mutating PowerShell or unrestricted GUI automation until AST/policy and verification boundaries exist.

### Known operational constraints

- Public search providers can return bot challenges, location-specific results, or no results. The runtime falls back and filters rather than treating unrelated pages as YouTube videos.
- Modern packaged Windows applications may use a launcher PID different from the visible window; text entry includes executable-name window fallback.
- The frontend is still polling task state; SSE/WebSocket events are not implemented.
- The LLM provider facade is still Ollama-specific and provider selection is not fully configurable.
