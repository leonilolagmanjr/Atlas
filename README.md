# Atlas

Atlas is a local-first Python assistant with local Ollama generation, document retrieval, conversation memory, read-only research, and permission-aware Windows tools. A FastAPI adapter and React/Vite control room expose the same runtime. It is not a general autonomous computer agent or a security sandbox.

**Documentation authority:** This README is the current architecture, installation, configuration, testing, and roadmap reference. The other architecture/setup/status documents link here rather than maintaining competing descriptions. [SECURITY.md](SECURITY.md) retains detailed security policy and limitations; [CHANGELOG.md](CHANGELOG.md) retains historical entries, not current capability or test-status guarantees.

## Contents

- [Current capabilities](#current-capabilities)
- [Architecture](#architecture)
- [Sources and evidence](#sources-and-evidence)
- [Tools and execution](#tools-and-execution)
- [Memory and persistence](#memory-and-persistence)
- [Installation](#installation)
- [Running](#running)
- [Configuration](#configuration)
- [API and frontend](#api-and-frontend)
- [Testing](#testing)
- [Limitations and security](#limitations-and-security)
- [Roadmap](#roadmap)

## Current capabilities

- Structured semantic request interpretation using local Qwen, with deterministic fast paths and fallback.
- Source-aware answers from local model knowledge, indexed documents, permitted files, current public web results, conversation history, runtime introspection, and system observations.
- Incremental PDF indexing, ChromaDB storage, and staged semantic/keyword/metadata retrieval.
- Bounded local file listing, search, UTF-8 text reading, PDF text extraction, and content search.
- Registered, permission-aware application launch/text entry and filesystem write/create/move/copy actions.
- Read-only Windows process/system/application inspection and validated PowerShell inspection.
- Dependency-ordered action plans, named output references, sequential execution, approval pause/resume, and bounded recovery.
- Persisted conversation sessions, durable API task snapshots, and sanitized reasoning-stage/provenance displays.

These are implemented paths, not a guarantee that arbitrary natural-language requests work. Model interpretation, search-provider availability, application/window resolution, and evidence quality affect results. Generic software installation, downloads, browser interaction, OCR, and unrestricted GUI automation are not implemented.

## Architecture

Atlas separates request interpretation, source selection, evidence evaluation, answer generation, and consequential execution:

```text
CLI / FastAPI + React
  -> Brain: request context and conversation history
  -> Semantic Task Interpreter: natural language -> Task IR
  -> Task Validator: registry, parameters, dependencies, references, risk
  -> ReasoningEngine
       -> QueryRouter + SourceSelector: ordered information sources
       -> EvidenceManager: collect and evaluate usable evidence
       -> AnswerGenerator: grounded, direct, clarification, or limitation answer
       OR delegate any Task.actions to the existing action pipeline
  -> TaskPlanner + concrete plan validation
  -> Executor -> ToolRouter -> PermissionEngine -> registered tool
       -> observations / verifier / bounded recovery
  -> response, memory, task state, API snapshot
```

`Brain` invokes the reasoning engine for eligible validated tasks when enabled. The engine can finish an informational request without building an execution plan. It serves a task directly when its actions are **all read-only research** (`web.search`/`web.fetch`, and non-mutating `filesystem.list/search/read/metadata/search_content`) and answers from the retrieved evidence; any task that mutates state or controls applications—including a hybrid such as "search the web for X and write it into Notepad"—is preserved and delegated to the existing validator/planner/executor path rather than consumed as an answer-only request. Invalid tasks are not given a source-execution bypass.

### Task interpretation

`models_task.py` defines `Task` and `TaskAction`, the boundary between model interpretation and deterministic execution. A task includes its goal, task type, entities, constraints, actions, confidence, and clarification state, plus structured reasoning fields such as `request_type`, `sources`, `current_information_required`, and `response_mode`.

The semantic interpreter can propose source/request-type fields directly. Routing also uses deterministic signals and conversation context; it is not solely a keyword-to-tool switch. Instructional questions must remain distinct from requests to perform an action. Unsupported or ambiguous targets should produce clarification or a limitation, not an invented tool.

An action identifies a registered capability, parameters, dependencies, and optionally a named output. For example, content generation can publish `generated_text`, which `applications.write_text` consumes as `$generated_text`. Topic, tone, length, and destination are separate parameters: in “write a poem about cars in Notepad,” cars is the topic and Notepad is the application.

The model proposes structure, not shell execution. Validation checks capability existence, required parameters/types, dependency identifiers, and output references. Planning orders dependencies, but cycle validation and order-independent reference validation remain incomplete. The generated-text handoff works; generic `$name` output resolution still has legacy gaps and must not be treated as reliable arbitrary dataflow.

### Module map

| Path | Responsibility |
| --- | --- |
| `atlas.py`, `cli.py` | Startup, indexing initialization, chat loop, session and approval commands |
| `brain.py` | Request lifecycle, interpretation/validation, reasoning integration, action delegation, early-answer persistence |
| `models_task.py`, `models.py` | Task IR, execution contexts, plans, steps, lifecycle and result models |
| `reasoning/task_interpreter.py`, `reasoning/prompts.py`, `reasoning/json_llm.py` | Semantic interpretation and structured model output handling |
| `reasoning/reasoning_engine.py` | Bounded source orchestration and answer/delegation decision |
| `reasoning/query_router.py`, `reasoning/source_selector.py` | Routing signals and ordered source plans |
| `reasoning/evidence_manager.py`, `evidence_models.py` | Evidence collection, provenance and sufficiency checks |
| `reasoning/answer_generator.py`, `reasoning/self_introspection.py` | Answer modes and registry/config-derived self-description |
| `reasoning/task_validator.py`, `reasoning/task_planner.py` | Task validation and dependency-ordered action planning |
| `executor.py`, `reasoning/verifier.py`, `reasoning/recovery.py` | Sequential execution, output publication, observations, verification and bounded recovery |
| `planner.py`, `intent_classifier.py`, `reasoning/interpreter.py` | Retained deterministic/legacy planning and compatibility paths, not the primary Task IR boundary |
| `knowledge_search.py` | Query expansion, staged retrieval, ranking and confidence decision |
| `document_loader.py`, `chunker.py`, `indexer.py`, `vector_store.py` | PDF loading, character chunks, file-hash indexing and ChromaDB access |
| `llm.py`, `providers/` | Injectable model-call boundary; current facade constructs the Ollama provider |
| `tools/` | Tool contracts, actual registry, capability descriptors, router, permissions, knowledge and discovery |
| `computer/`, `web.py` | Windows/filesystem/PowerShell providers and read-only public search/fetch |
| `memory/`, `task_store.py` | Conversation storage and durable API task snapshots |
| `api.py`, `frontend/` | Local HTTP adapter and polling control room |
| `config.py`, `logger.py`, `prompts/` | Runtime constants, logging and prompt templates |

Keep new capabilities behind existing tool contracts; do not put subprocess execution in the interpreter, source selector, or frontend. Prefer injected providers/tools for tests and preserve the distinction between evidence and instructions.

## Sources and evidence

### Source selection

| Request | Current behavior |
| --- | --- |
| Ordinary/general question | Uses the selected local knowledge/model sources; local model answers do not require a successful PDF retrieval. A knowledge miss is not terminal: the model's own knowledge is a source. |
| Current or explicitly online question | Consults registered read-only web search/fetch and answers from the retrieved evidence with sources; unavailable live evidence produces a limitation or clearly qualified, potentially outdated model fallback |
| File lookup/read/content question | Uses permitted local files and, where selected, indexed knowledge; a failed private-file read is not replaced with an invented model answer |
| Personal-document lookup | "find my resume" / "search my documents for 'climate change'" are local file lookups (filename search or content search), not web searches, unless a web platform is named explicitly |
| Local system question | Uses system observations; unavailable machine facts are not inferred from general model knowledge |
| Earlier conversation | Uses available conversation history, not an unimplemented long-term fact database |
| Atlas capabilities/model/tools | Uses actual registered capability metadata and runtime configuration, rather than model claims about installed tools |
| Action request | Preserves `Task.actions` and delegates execution to the permission-gated executor |
| Hybrid request | Combines a source and an action: "search the web for X and write it into Notepad" retrieves evidence, renders it as text, and writes it after confirmation. Composition is limited to the existing action capabilities. |
| Ambiguous request | "open it", "find that file", "make it better" with no resolvable referent return a clarification question instead of guessing a source or action |

Read-only research actions (`web.search`/`web.fetch` and non-mutating `filesystem.list/search/read/metadata/search_content`) are served by the reasoning engine, which synthesizes a cited answer. Tasks that mutate state or control applications—including hybrids—are delegated to the validator/planner/executor path. An ordered source plan is not an autonomous research/action graph; selecting web and computer sources does not implement “research and install any program”, and only existing validated Task actions can run.

### Local knowledge

Startup ingestion reads PDFs from `KNOWLEDGE_FOLDER`, splits text into overlapping character chunks, embeds them with sentence-transformers, and stores them in ChromaDB. File hashes track incremental indexing. This PDF loader uses **pypdf**, not PyMuPDF, and does not perform OCR.

`knowledge_search.py` expands queries, tries semantic retrieval, retries expanded queries when needed, adds keyword/metadata signals, and evaluates merged evidence. `MIN_SIMILARITY` participates in this decision; it is not a global prohibition on all model calls. Rejected retrieval is not promoted to evidence merely because it returned candidate chunks. With the general-question fallback disabled, legacy knowledge-only requests can return “I don't know based on my knowledge base.”

### Files and PDFs

Filesystem tools resolve paths against `COMPUTER_ROOT`, reject outside-root targets, and recheck traversal candidates. The default root is this repository: mentioning Downloads, Documents, or Desktop does **not** grant access to those folders. Change the configured root deliberately if needed.

Read/search paths exclude common runtime, dependency, credential, and private-storage locations. Walks, file counts, input bytes, returned text, and results are bounded; exclusions are not a complete secret detector. `filesystem.search` finds names/paths, while `filesystem.search_content` searches extracted text and returns bounded excerpts. These are distinct contracts.

PDF file tools reuse pypdf with input-size, page-count, output-size, and between-page deadline checks. Image-only PDFs need OCR that is not present. A single parser/page extraction can still block; the PDF limits are not hard preemption. The bounded file-tool extraction path is distinct from startup knowledge ingestion and its indexing workload.

### Web and answer quality

`web.search` uses public search providers with per-provider failure isolation and relevance filtering. YouTube requests can return watch links and native thumbnails. `web.fetch` reads bounded public HTML/plain text, treats it as untrusted evidence, and does not execute page scripts or save downloads. Provider changes, bot challenges, incomplete snippets, and ranking limitations can yield missing or poor results.

Answers distinguish direct model knowledge from retrieved evidence, include source metadata/citations where available, and report missing evidence. Evidence sufficiency and output verification are heuristics, not independent proof that a factual answer is correct or current.

## Tools and execution

`ToolRegistry` is the authority for installed tools. `CapabilityRegistry` derives descriptors from that real registry, with contract metadata for required parameters, risk, outputs, and verifiability. Its plannable catalog is a subset of registered capabilities; descriptors alone do not install tools or grant permission.

| Capability family | Current scope |
| --- | --- |
| `content.generate` | Local-model content generation for composed tasks |
| `filesystem.list/read/metadata/search/search_content` | Bounded local inspection and text/PDF lookup |
| `filesystem.write/create_folder/move/copy` | Root-bounded mutations, permission-gated in default confirm mode |
| `applications.list`, `applications.launch`, `applications.launch_named`, `applications.write_text` | Installed-app inspection, non-shell launch, and targeted Windows text entry |
| `system.info`, `processes.list`, process inspection tools | Registered read-only machine/process information |
| `powershell.execute` | Documented, validated read-only inspection, not arbitrary shell access |
| `web.search`, `web.fetch` | Read-only public search/page evidence |

`tools/knowledge.py`, `tools/discovery.py`, and `tools/powershell_commands.json` provide searchable command knowledge and non-executing discovery. PowerShell planning uses documented inspection commands and structured output. Knowledge records are data, not authorization to execute a command. The lexical PowerShell validator is not an AST sandbox.

### Approval, verification and recovery

In the default `confirm` mode, consequential registered actions pause for approval through `/approve` or the task-specific API endpoint. Denial stops the paused action. Resumption keeps completed steps instead of regenerating their outputs. `safe` permits read-only tools; `autonomous` changes low/medium-risk confirmation behavior and should not be mistaken for a safety guarantee. Critical operations are denied by default, subject to explicit policy overrides.

Application text entry resolves a target and attempts window/control-based delivery, with clipboard/focus fallbacks. It is not unrestricted mouse/keyboard automation and can fail with incompatible applications or ambiguous windows.

The verifier records `verified`, `unverified`, or `failed` according to capability contracts. Many checks inspect returned output shape or tool-reported values (for example, a PID or character count), **not an independent observation of the intended effect**. Bounded recovery can adjust arguments for the same capability, with per-step and plan-wide limits; it is not arbitrary replanning.

New API task submissions use a single FIFO worker, but approval resolution has known concurrency gaps. Do not rely on a global guarantee that all actions can never overlap. Approvals are associated with a task, but existing grants are not fully bound to exact step arguments; changed recovery arguments need stronger reapproval handling. Both are priority hardening work, not completed security properties.

## Memory and persistence

`memory/` persists conversation sessions under `memory/sessions/` and builds prompt history from a recent-message window. Brain now persists user/assistant turns for answers returned early by the reasoning engine; persistence is not limited to the legacy retrieval executor path. This does not imply that every failure/approval path has a complete durable event history.

CLI session commands include `/new [title]`, `/list`, `/open <id>`, `/delete <id>`, `/rename <id> <title>`, `/history`, `/export <id> [path]`, `/import <path>`, `/clear`, and `/help`. `/approve` and `/deny` resolve paused actions.

API task snapshots are stored in `database/tasks.json` using atomic replacement. They are history, **not live execution checkpoints**. Restarted pending/running/approval-paused tasks are marked interrupted rather than resumed with missing context. Long-term semantic memory, learned preferences, reflection, and automatic conversation summarization remain future work.

Runtime/user data includes `database/`, `memory/sessions/`, knowledge PDFs, logs, and exported sessions. Treat these as private local data; do not commit them or assume all are ignored automatically. `.venv/`, frontend dependencies/build outputs, and Python caches are generated artifacts.

## Installation

The documented desktop workflow targets Windows and PowerShell. Windows application/PowerShell tools require Windows. Use Python 3.10+ with dependency-compatible packages, Node.js/npm compatible with the installed Vite version, and a running Ollama service. Dependencies are not comprehensively pinned, so verify the environment rather than assuming every Python/Node release is supported.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:7b
npm --prefix frontend install
```

`requirements.txt` includes `chromadb`, `sentence-transformers`, `pypdf`, `ollama`, `fastapi`, `uvicorn`, and `httpx`. Ollama must be installed separately and serving the configured model. Initial embedding-model loading may require a download. Knowledge PDFs are optional for general model answers; place documents in `knowledge/` for local retrieval.

## Running

### CLI

```powershell
.\.venv\Scripts\python.exe atlas.py
```

Startup initializes the local runtime and indexes changed knowledge PDFs. Ask ordinary questions, inspect supported sources, or submit a supported action. For example, “What is Python?” need not fail merely because the document index has no Python evidence; “What is the latest Python version?” needs current web evidence to be presented as current.

### Web console

In a terminal at the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In another terminal at the repository root:

```powershell
npm --prefix frontend run dev
```

Open `http://localhost:5173`. After dependencies are installed, `launch_atlas.bat` provides the Windows shortcut for starting the backend, frontend, and browser. The launcher checks the virtual environment, Node/npm, and frontend dependencies. Close the service terminals to stop them.

Keep the backend bound to loopback. The documented deployment has no authenticated multi-user boundary and is not intended for public exposure.

## Configuration

Most runtime settings are Python constants in `config.py`; there is no backend `.env` loader. Restart processes after changing them. Some tool-specific hard limits and the plan recovery cap remain module constants rather than centrally configurable settings.

| Setting | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Current local generation/interpretation model |
| `ENABLE_REASONING_ENGINE` | `True` | Enable source-aware reasoning; disabling retains the legacy informational path |
| `ENABLE_LLM_INTERPRETATION` | `True` | Enable model-assisted interpretation/recovery; not a switch disabling all answer generation |
| `INTERPRETER_CONFIDENCE_THRESHOLD` | `0.75` | Confidence at which the deterministic interpretation can skip the model |
| `CLARIFICATION_CONFIDENCE_THRESHOLD` | `0.45` | Low-confidence clarification policy |
| `ENABLE_GENERAL_QUESTION_FALLBACK` | `True` | Allow model fallback instead of requiring accepted local knowledge for ordinary questions |
| `MAX_REASONING_ITERATIONS` | `3` | Maximum selected-source iterations, not unrestricted agent loops |
| `MAX_REASONING_SOURCES` | `3` | Maximum ordered sources consulted |
| `MAX_REASONING_TOOL_CALLS` | `8` | Budget charged per actual reasoning tool call, including each search/fetch/read |
| `REASONING_TIMEOUT_SECONDS` | `60.0` | Reasoning deadline checks and Ollama transport timeout configuration; not hard whole-request preemption |
| `MAX_RECOVERY_ATTEMPTS` | `2` | Bounded recovery attempts; executor also has `MAX_PLAN_RECOVERIES = 3` |
| `WEB_RESEARCH_MAX_RESULTS` / `WEB_RESEARCH_MAX_PAGES` | `6` / `2` | Search results requested / result pages read by reasoning |
| `FILESYSTEM_CONTENT_MAX_FILES` / `FILESYSTEM_CONTENT_MAX_BYTES` | `200` / `200000` | Content-search file and byte bounds |
| `EXECUTION_MODE` | `confirm` | `safe`, `confirm`, or `autonomous` permission policy |
| `COMPUTER_ROOT` | Repository root | Allowed filesystem root |
| `KNOWLEDGE_FOLDER` / `KNOWLEDGE_GLOB` | `knowledge/` / `*.pdf` | Startup knowledge inputs |
| `DATABASE_FOLDER` / `COLLECTION_NAME` | `database/` / `atlas_knowledge` | Chroma persistence and collection |
| `INDEX_STATE_FILE` / `TASK_STORE_FILE` | `database/index_state.json` / `database/tasks.json` | Index hashes and API task snapshots |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | Character-based knowledge chunking |
| `TOP_K` / `MIN_SIMILARITY` | `5` / `0.75` | Retrieval ranking/acceptance inputs |
| `MEMORY_FOLDER` / `MAX_RETAINED_MESSAGES` | `memory/` / `10` | Session storage and prompt history window |
| `MAX_SESSIONS` / `AUTO_SAVE` | `50` / `True` | Session-count warning threshold (creation still proceeds) and autosave |
| `MAX_CONTEXT_TOKENS` / `AUTO_SUMMARIZE_THRESHOLD` | `2048` / `80` | Reserved/placeholder controls, not enforced token budgeting or automatic summarization |
| `LOG_LEVEL` / `LOG_TO_FILE` / `LOG_FILE` | `INFO` / `False` / `database/atlas.log` | Logging |
| `LOG_RETRIEVAL` / `DEBUG_PIPELINE` | `True` / `False` | Retrieval diagnostics and structured pipeline tracing |
| `REDACT_KEYS` | `password`, `token`, `api_key`, `secret` | Diagnostic key redaction, not comprehensive content redaction |

The frontend reads `VITE_ATLAS_API_URL` through Vite, defaulting to `http://127.0.0.1:8000/api`. This is a frontend build/dev setting, not a Python `.env` setting.

## API and frontend

The React frontend does not plan or execute tools. `frontend/src/App.tsx` renders views, `types.ts` describes API-facing data, `services/api.ts` owns HTTP calls, and `styles.css` defines the interface. Tasks are polled; there is no SSE/WebSocket execution stream.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health`, `/api/system` | Runtime health and actual system data |
| GET | `/api/tools`, `/api/tool-knowledge` | Registered tools and command knowledge |
| GET | `/api/tool-discovery?query=...` | Non-executing discovery |
| GET | `/api/applications` | Installed Windows application inventory |
| GET | `/api/tasks`, `/api/tasks/{id}` | Task history and task snapshots |
| GET | `/api/queue` | Current worker/queued submission state |
| POST | `/api/tasks` | Submit `{ "request": "..." }` |
| POST | `/api/tasks/{id}/approve` | Resume the matching approval-paused task |
| POST | `/api/tasks/{id}/deny` | Deny the matching approval-paused task; not general running-task cancellation |

The control room shows task status, plan steps, results, approval controls, applications, tools, and system data. Reasoning snapshots expose bounded stage/detail/iteration fields, response mode, source names, citations, and evidence counts, not raw evidence documents or prompts. Citations are sanitized (including removal of URL credentials/query/fragment and reduction of local paths to names). Persisted reasoning snapshots are sanitized again when restored.

This telemetry filtering is not blanket redaction of all task data: requests, responses, and existing tool outputs can still contain sensitive content. Stage snapshots may appear after a stage/answer completes rather than streaming every in-flight model operation. Files, Knowledge, Memory, and editable Settings views still need dedicated management APIs.

## Testing

Run checks from the repository root after installing dependencies:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
npm --prefix frontend run build
```

The frontend build is **`tsc -b && vite build`**, so it includes TypeScript checking and production bundling. There is no configured Python lint/typecheck command, and no frontend lint or dedicated frontend test script in `package.json`. Do not describe a build as a lint run.

The suite uses `unittest` and covers these categories:

- Semantic Task/source fields, malformed model output, instructional-question versus action behavior, clarification and parameter preservation.
- Registry-derived capability availability/introspection, task validation, dependency ordering, generated-text handoff, action delegation and end-to-end execution with fake models/tools. This coverage does not establish complete cycle/reference correctness.
- General/current/local-file/system/memory source routing, accepted versus rejected evidence, answer provenance, per-call budgets and cooperative deadline/cancellation behavior.
- General-purpose routing regressions: personal-document lookups staying local, create-text-file actions, hybrid search-then-write composition, ambiguous requests clarifying instead of guessing, and the read-only-research-versus-mutation delegation split.
- Bounded filesystem traversal/read/content search, exclusions, root escapes, PDF text handling, and move/copy contracts. Platform or symlink-privilege checks may skip where unavailable.
- Permission decisions, pause/resume, recovery, task snapshots, queue behavior, application resolution and mocked Windows text-entry paths.
- PowerShell validation/result interpretation, public URL and redirect checks, provider fallback, mocked web/YouTube responses, and Ollama timeout configuration.
- API contracts and bounded/sanitized reasoning telemetry, including restored records.

Most contract tests use fakes/mocks; a passing suite does not prove live web freshness, model quality, broad Windows app compatibility, or independently verified computer effects. Some platform smoke paths require Windows/PowerShell. Keep full-suite runs serialized when using shared runtime files. No fixed passing-test total is maintained here; use the current command output and report skips/failures for the exact revision tested.

Manual smoke checks should separately confirm local Ollama availability, one general answer, one current question with provenance, a root-permitted file read, capability introspection, session persistence, and approval/denial for a harmless supported action. Do not run live mutating app/file tests without explicit approval. Main integration validation is separate from this documentation consolidation.

## Limitations and security

- **Not a sandbox:** Python/tool processes run with the user's OS privileges. Root checks, permission policy, and PowerShell validation reduce exposure but do not provide OS isolation. See [SECURITY.md](SECURITY.md).
- **Cooperative limits:** the reasoning deadline is checked between calls; the engine accepts a cancellation callback/token, but Brain/API/UI do not provide full end-to-end running-task cancellation. In-flight model, file/PDF, or network calls are not forcibly preempted by that token.
- **Transport is not wall-clock preemption:** Ollama has a transport timeout, not a hard absolute generation deadline. PDF extraction can block within one page. Reasoning source/tool budgets do not globally meter every internal provider request, retrieval subquery, or legacy executor operation.
- **Web boundary gaps:** initial/final URLs and redirects check public IPv4/IPv6 destinations, but DNS re-resolution and proxy behavior leave DNS-rebinding/TOCTOU risk. This is not a network sandbox.
- **Approval gaps:** task-specific approval does not yet ensure atomic serialized approval execution or immutable argument-bound grants. Recoveries that change effectful arguments need reapproval hardening.
- **Verification limits:** tool-reported output shape often stands in for independent effect checks; citations and retrieval sufficiency do not prove truth.
- **Local-first, not offline-only:** selected web research sends queries to public services; model/embedding setup may download data. Files, memory, task history, and logs can hold private information.
- **Capability limits:** no generic install/package management, controlled downloads, browser automation, OCR, voice/vision, durable resumable execution, or long-term semantic memory. Existing application text entry is narrower than full GUI automation.

## Roadmap

Priority order emphasizes reliability before broader autonomy:

1. Serialize approval/resume through the execution queue and bind approval to exact step/target/arguments, including changed recovery arguments.
2. Complete dependency-cycle validation, order-independent reference checks, generic output resolution, and recovery-state correctness. Add independent effect observations for file/application actions and stronger source/answer verification; test ambiguous targets and failed recovery honestly.
3. Wire cancellation across Brain, API and UI; isolate potentially blocking model/PDF/tool work where hard deadlines are needed; add durable live checkpoints without silently replaying effects.
4. Harden web connection resolution/proxy handling, source trust and provenance, and expand regression coverage for hostile/unavailable sources and platform boundaries.
5. Improve semantic source/action composition within supported capabilities, retrieval evaluation against representative PDFs, and live model/provider compatibility checks.
6. Add bounded Files/Knowledge/Memory APIs, validated Settings, browser regression tests, and an event model before replacing polling with streaming updates.
7. Develop long-term memory with provenance, user-controlled retention/deletion, summarization, confidence, and retrieval; current session history is not this feature.
8. Consider controlled downloads/package installation and browser/GUI tools only behind explicit permissions, stronger validation/isolation, and effect verification. Later directions include voice, vision/OCR, coding integrations, specialist agents, scheduling, and long-running projects.

Roadmap items are future work, not release promises or evidence of implemented capability.
