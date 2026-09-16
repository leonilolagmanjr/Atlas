# Atlas

Atlas is a local-first AI runtime foundation for retrieval, planning, memory,
and eventually controlled computer and internet tools.

It is not designed as a monolithic chatbot. Atlas separates orchestration,
retrieval, indexing, vector storage, prompting, logging, and LLM access so each
part can evolve independently.

## Current Version
Atlas is currently implemented as the V3.4 semantic task-understanding runtime:
a local-first **Semantic Task Interpreter → Task IR → Task Validator → Dynamic
Task Planner → Capability Registry → Deterministic Executor → Observation/
Verification → Bounded Replanning** pipeline on top of the existing Brain,
staged hybrid retrieval, and conversation memory.

The LLM interprets natural language into a structured, validated **Task**; it
never executes anything and never emits shell text. Deterministic code decides
how each validated capability is safely run. See
[ARCHITECTURE_ASSESSMENT.md](ARCHITECTURE_ASSESSMENT.md) for the current handoff
and migration plan.

Implemented foundation:

- Local CLI assistant
- Ollama-backed LLM calls
- PDF knowledge ingestion
- Chunked document processing
- ChromaDB vector storage
- Incremental indexing with file hashing
- Prompt templates
- Central configuration
- Central logging
- Brain orchestration layer
- Deterministic planner and execution plan
- Sequential executor
- Shared execution context models
- Retrieval diagnostics
- Hybrid retrieval with semantic, keyword, and metadata signals
- Adaptive retrieval confidence policy
- Windows computer tools for bounded filesystem, process, system, and installed-application inspection, plus permission-gated executable launch
- Durable API task snapshots with task-bound approval and interruption handling
- Data-driven tool knowledge and capability discovery
- Validated, read-only PowerShell inspection for processes, services, system, and network state
- Confirmation-gated text entry into resolved Windows applications
- Read-only web search and public page retrieval with direct YouTube thumbnails and source links
- Natural-language semantic task interpretation via local Qwen with a deterministic fast path
- A first-class `Task` / `TaskAction` intermediate representation with first-class parameters (topic, tone, style, length, sort, destination, application, folder, filename)
- Machine-readable capability registry (name, description, parameter schema, required parameters, risk, confirmation, verifiability)
- Task validation (capability existence, required parameters, parameter types, dependencies, references, risk) before any planning
- Dynamic multi-step planning with dependency ordering and `$variable` output references between steps
- Post-action observation and honest verification (verified / unverified / failed)
- Plan-wide bounded replanning with an anti-loop budget
- Conversational task state for follow-up requests
- Structured debug trace across every stage (task, validation, plan, execution, verification)
## Semantic Task Understanding (Natural Language → Action)

Atlas understands what the user *means*, not which command they typed. The same
task can be expressed many ways and Atlas normalizes it:

```text
create / write / make / compose a poem about cars in Notepad
put a poem about automobiles into Notepad
make something poetic about cars and put it in Notepad
search / find / look for / show me MrBeast videos on YouTube
```

The pipeline separates **LLM reasoning** from **deterministic execution**:

```text
USER REQUEST
     ↓
SEMANTIC TASK INTERPRETER (Qwen, or deterministic fast path)
     ↓
TASK IR  { task_type, goal, actions[], entities, constraints, confidence }
     ↓
TASK VALIDATOR (capability existence, params, types, deps, refs, risk)
     ↓
DYNAMIC TASK PLANNER (dependency order + $variable output references)
     ↓
CAPABILITY REGISTRY (names + parameter schemas + risk + verification)
     ↓
DETERMINISTIC EXECUTOR  →  OBSERVATION / VERIFICATION
     ↓
QWEN RECOVERY / bounded replanning (only on recoverable failure)  →  USER
```

The LLM never executes anything. It decides *what* should happen; Atlas's
deterministic layer validates it and decides *how* it is safely executed.

### The Task IR
A request is interpreted into a structured task, not a single label. For
`Create a poem in Notepad about cars.`:

```json
{
  "task_type": "content_creation",
  "goal": "create_content",
  "actions": [
    {"action_id": "a1", "capability": "content.generate",
     "parameters": {"content_type": "poem", "topic": "cars"},
     "produces": "generated_text"},
    {"action_id": "a2", "capability": "applications.write_text",
     "parameters": {"application": "Notepad", "text": "$generated_text"},
     "depends_on": ["a1"]}
  ],
  "entities": {"application": "Notepad", "topic": "cars", "content_type": "poem"},
  "execution_required": true,
  "confidence": 0.85
}
```

The critical property: adding `about cars` changes only the **topic**, never the
destination. `Notepad` stays `Notepad`; `cars` becomes the content topic. Without
this separation the modifier leaked into the application name and the command
failed.

### Capabilities, not commands
The interpreter and planner reason over a machine-readable **capability
registry** (name, description, parameter schema, required parameters, risk,
confirmation requirement, verifiability). The model may only select from the
registered capabilities. A plan that references an unknown capability, omits a
required parameter, passes a wrong type, or references a `$variable` nobody
produces is rejected during validation, before anything runs.

### Variables and outputs
A step can publish a named output that later steps consume by reference:

```text
1. content.generate          produces $generated_text
2. applications.write_text   text = $generated_text
3. filesystem.search         produces $largest_match
4. filesystem.move           source = $largest_match
```

Unresolved references never execute silently; they fail validation.

### Verification
After each action Atlas records an honest observation: `verified`, `unverified`,
or `failed`. A launch is verified by a process id, a text write by a character
count, a search by returned results. When Atlas cannot confirm an effect it says
so rather than pretending success.

### Capabilities, not commands

The planner and interpreter receive a machine-readable **capability catalog**
derived from the tool registry (name, description, parameter schema). The model
may only select from registered capabilities. Plans that reference an unknown
capability, a missing tool, or a malformed parameter block are rejected during
validation, before anything runs.

### Confidence and clarification
- High confidence → execute.
- Medium confidence → infer from conversation context where safe.
- Low confidence, or an unresolved reference such as `open it and write something` → ask a clarifying question instead of guessing.

### Focus-independent text entry
Writing into an application does not depend on that application being focused or
in front. Atlas locates the target window by process, executable, or window
class (packaged apps like Windows 11 Notepad launch a stub process, so the pid
cannot be relied on), finds the edit control inside it, and delivers text with
window messages (`WM_SETTEXT`, then `WM_PASTE` as a fallback). Focusing the
window plus a clipboard paste is only used as a last resort.

### Approval and resume
Permission-gated steps (writing into an application, launching an executable)
pause the plan and wait for confirmation. On `/approve` (or the API approval
action) the plan **resumes from the paused step**: already-completed steps such
as content generation are not run again, and the approval is applied to every
gated step in the plan rather than being consumed by the first tool call. A
single approval completes the plan; it does not re-prompt in a loop.

### Task queue and step-by-step execution
Atlas runs one task at a time. Every request is placed on a single-worker FIFO
queue, so two computer-control actions can never run at once. Tasks submitted
while another is running wait in an explicit queue that the control room
displays. Within a task, plan steps execute sequentially and a permission-gated
step pauses the whole queue until it is approved or denied.

### Failure recovery
When a planned step fails, the failure is classified (invalid arguments,
missing target, permission, timeout). Recoverable failures trigger a single
bounded, catalog-constrained recovery attempt that may only correct arguments
for the same capability. Recovery refuses permanent failures, refuses to retry
with identical arguments, and is capped by `MAX_RECOVERY_ATTEMPTS`.

## Retrieval Pipeline

Atlas keeps hallucination protection by requiring retrieved knowledge before
calling the LLM, but retrieval confidence is now evaluated after multiple
search attempts instead of before them.

Current flow:

```text
User question
Brain
Planner
Execution plan
Executor
Intent-aware query expansion
Semantic vector search
Result evaluation
Expanded-query retry when weak
Keyword and metadata search when needed
Merged ranked results
Adaptive confidence decision
LLM call only when retrieval is supported
```

The retrieval layer considers:

- Best semantic distance
- Number of retrieved chunks
- Keyword matches in chunk text
- Filename and document metadata matches
- Combined evidence across multiple queries

If retrieval is rejected (no accepted evidence context), Atlas returns:

```text
I don't know based on my knowledge base.
```

## Query Expansion

The current query expansion is deterministic and modular. It can later be
replaced with an LLM, planner, or agent-generated search strategy without
changing the rest of the retrieval pipeline.

Examples:

- `What certifications do I have?`
  - `certifications`
  - `certificates`
  - `credentials`
  - `training`
  - `licenses`
  - `achievements`

- `Tell me about my resume.`
  - `resume`
  - `experience`
  - `education`
  - `skills`
  - `projects`
  - `profile`

- `Who is Leonilo?`
  - `Leonilo`
  - `Leonilo Lagman`
  - `Lagman`
  - `candidate`
  - `profile`
  - `personal information`

## Project Structure

```text
Atlas/
  atlas.py             Entry point and chat loop
  api.py               FastAPI adapter for health, system, tools, tasks, approval flow
  brain.py             Request orchestration and execution state tracking
  cli.py               CLI session commands and approval callbacks
  chunker.py           Text chunking
  config.py            Central configuration
  document_loader.py   Document reading
  executor.py          Sequential execution plan runner
  indexer.py           Incremental indexing
  intent_classifier.py Rule-based coarse classification (legacy knowledge-plan selection only)
  models_task.py       Task/TaskAction intermediate representation (the Task IR)
  reasoning/           LLM reasoning layer (task interpreter, validator, planner, verifier, recovery, diagnostics)
  knowledge_search.py  Retrieval strategy and confidence policy
  llm.py               Ollama communication
  logger.py            Logging setup
  models.py            Shared execution dataclasses
  planner.py           Deterministic plan creation
  task_store.py        Durable API task snapshot storage
  vector_store.py      ChromaDB operations
  web.py               Read-only web search and bounded page retrieval
  evidence_models.py   Retrieval evidence models
  prompts/
    system.txt
    retrieval.txt
  knowledge/
    *.pdf
  memory/              Conversation sessions, models, storage, context builder
  providers/           Provider interface and Ollama implementation
  tools/               Tool contracts, registry, router, permissions, discovery
  computer/            Read-only Windows system/app tools and PowerShell provider
  frontend/            React/Vite control room
  database/            Local generated ChromaDB data and task history, ignored by git
```

## Responsibilities

`atlas.py` is only the entry point. It initializes logging, loads prompts,
indexes the knowledge folder, creates the vector store and Brain, and runs the
chat loop.

`brain.py` creates the execution context, asks the planner for a plan, passes
that plan to the executor, and returns the final response.

`planner.py` creates deterministic retrieval/response plans and explicit
tool plans for supported intents: knowledge retrieval, web search, computer
inspection via PowerShell, application launch (explicit executable or named),
and application text entry.

`executor.py` owns step-by-step execution. It updates the shared execution
context, runs retrieval or routed tools, pauses for confirmation when required,
decides whether there is enough evidence to call the LLM, and records step
status and failures.

`models.py` contains shared dataclasses for execution context, task lifecycle,
plans, steps, planner decisions, evidence, and retrieval results.

`intent_classifier.py` is retained only as a rule-based (no LLM) signal used by
legacy plan selection for knowledge workflows. It is no longer the primary
natural-language understanding path: the `reasoning/` package interprets free
text into a structured `Task` that drives the planner directly. New natural-
language behavior must not be implemented by adding more patterns here.

`models_task.py` defines the `Task` / `TaskAction` intermediate representation —
the single boundary object between the LLM interpreter and Atlas's
deterministic planning and execution layer.

`reasoning/` owns the LLM-facing reasoning stages. It contains the semantic task
interpreter, the task validator, the dynamic task planner, the action verifier,
modular prompts, strict-JSON recovery helpers, bounded failure recovery, and the
structured request diagnostics trace. It never imports a concrete provider and
never executes commands.

`web.py` provides read-only public web search and bounded page retrieval, with
direct YouTube result parsing, provenance, and SSRF checks.

`knowledge_search.py` owns retrieval. It expands queries, performs staged
retrieval, merges hits, evaluates confidence, formats context, and logs
diagnostics.

`vector_store.py` owns ChromaDB access. It adds chunks, queries embeddings, and
exposes stored chunks for keyword and metadata retrieval.

`llm.py` only talks to Ollama.

`TOOLS.md` describes the general tool knowledge and discovery system. PowerShell
is the first knowledge-backed provider; documented commands are validated
before execution and remain separate from the LLM provider.

The control room exposes the same tool catalog and discovery flow through the
Tools view. PowerShell plans show their selected capability, candidate tools,
generated command, interpreted result, and expandable raw output.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install project dependencies used by the current codebase:

```powershell
pip install -r requirements.txt
```

`requirements.txt` includes `chromadb`, `sentence-transformers`, `pypdf`,
`ollama`, `fastapi`, `uvicorn`, and `httpx`.

Install Ollama and pull the configured model:

```powershell
ollama pull qwen2.5:7b
```

## Running

### One-click web console

Double-click [launch_atlas.bat](launch_atlas.bat). It opens the FastAPI
backend, the Vite frontend, and the control room in your browser.

The launcher expects the repository virtual environment and frontend
dependencies to already exist. For first-time setup, follow [INSTALL.md](INSTALL.md).

Place PDF knowledge files in `knowledge/`, then run:

```powershell
python atlas.py
```

Atlas indexes changed documents at startup and then enters a backend CLI loop.

Explicit executable and named-application launch requests are planned as
permission-gated actions. For example, `Open Discord` resolves a trusted
Discord executable without shell invocation, and the same resolver supports
other installed desktop applications. Use `/approve` or `/deny` after Atlas
pauses for confirmation.

Atlas can also perform controlled text-entry tasks from natural language. These
all resolve to the same task family while preserving their different parameters:

```text
create a poem in notepad
create a poem about cars in notepad
create a funny poem about cars in notepad
write a short poem about racing cars in notepad
open notepad and write a poem about cars
```

The last example is the key one: adding `about cars` changes only the content
topic, never the destination. Atlas generates the content (with your topic, tone,
style, and length) and then writes it into Notepad, pausing for approval before
the write. Because Atlas reasons over the *task* rather than matching a phrase,
an unseen combination such as `write a short story about space exploration in
Notepad` works without any new rule.

Atlas also supports current public-web research from natural language:

```text
search youtube for mrbeast videos
find mrbeast videos on youtube
show me mrbeast's latest videos
find popular mrbeast videos on youtube
search youtube for videos about cars
```

Each resolves to a YouTube search with a cleaned query and a structured sort
(`latest`, `popular`). Results return video thumbnails, watch links, and
snippets. Web content is treated as untrusted evidence and is never executed as
an Atlas instruction.

### Search backends and relevance
General web search gathers results from multiple backends (DuckDuckGo, Bing,
and the Wikipedia API) and then **scores and filters** them: results that do not
match the query's distinctive terms are dropped, so the runtime returns nothing
rather than surfacing stale or unrelated pages. DuckDuckGo challenge pages are
detected and treated as an empty result so a fallback backend can answer.
YouTube result pages embed large JSON payloads, which are read in full (a small
read cap silently produced zero results before).

## Web console

The repository includes a React/Vite control room in `frontend/`. It uses the
local API and does not duplicate Atlas planning or tool execution in the
browser.

Start the API from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The command workspace shows a spinner while a
request is submitted, a live step-by-step progress line while the plan runs, and
a task-queue panel listing what is running and waiting. Authorization buttons
disable and show a spinner while an approval is being applied, so a double click
cannot create a duplicate action; the API also rejects a second approve/deny for
the same task. Durable task polling, task-bound approval
flow, installed applications, tool registry, system inspection, and honest
backend-unavailable states are connected to the current runtime. Files,
Knowledge, Memory, and editable Settings require additional API endpoints.

In addition to normal questions, Atlas supports conversation/session commands:
- `/new [title]`
- `/list`
- `/open <session_id>`
- `/delete <session_id>`
- `/rename <session_id> <new_title>`
- `/history`
- `/export <session_id> [path]`
- `/import <path>`
- `/clear`
- `/approve`
- `/deny`
- `/help`

## Configuration


Important settings live in `config.py`:

- `OLLAMA_MODEL`
- `KNOWLEDGE_FOLDER`
- `DATABASE_FOLDER`
- `COLLECTION_NAME`
- `EMBEDDING_MODEL_NAME`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`
- `MIN_SIMILARITY`
- `LOG_RETRIEVAL`

Task-understanding settings:

- `ENABLE_LLM_INTERPRETATION` — master switch for Qwen-powered interpretation/planning/recovery
- `INTERPRETER_CONFIDENCE_THRESHOLD` — deterministic confidence above which the LLM interpreter is skipped (keeps mechanical requests fast)
- `CLARIFICATION_CONFIDENCE_THRESHOLD` — below which Atlas asks instead of guessing
- `MAX_RECOVERY_ATTEMPTS` — bounded LLM-assisted replans per failed step
- `REASONING_TIMEOUT_SECONDS` — per-call timeout for reasoning stages
- `DEBUG_PIPELINE` — emit the structured per-stage request trace (task, validation, plan, execution, verification)
- `REDACT_KEYS` — keys redacted from diagnostics

The plan-wide recovery budget (`executor.MAX_PLAN_RECOVERIES`, default 3) caps
total retries for a single plan so a multi-step task cannot loop.
`MIN_SIMILARITY` is used as part of the retrieval acceptance decision (`knowledge_search._decide()`),
not as a single hard pre-check before all retries.

## Generated Files

During normal operation Atlas may create or update the following generated artifacts in the repository:

- `database/`
  - ChromaDB storage and persisted index metadata
  - `database/index_state.json` for incremental PDF reindexing
  - optional log output when file logging is enabled
- `memory/sessions/`
  - per-session conversation stores and metadata persisted on disk
- `knowledge/`
  - user-provided source documents, typically PDFs
- `.venv/`
  - local Python environment created during setup
- `frontend/node_modules/`
  - installed NPM dependencies when the UI is started locally

These folders are runtime outputs, not source code, and are expected to change as the project is used.

The following are local runtime artifacts and should not be committed:

- `__pycache__/`
- `.venv/`
- `database/`
- `*.log`
- `.env` (not used by this repository’s code; safe to ignore)

The repository includes `.gitignore` entries for these paths.

## Verification

Useful local checks:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in pathlib.Path('.').glob('*.py')]; print('syntax ok')"
.\.venv\Scripts\python.exe -c "import atlas, brain, chunker, config, document_loader, executor, indexer, knowledge_search, llm, logger, models, planner, reasoning, vector_store; print('imports ok')"
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

The test suite includes:

- `tests/test_task_pipeline.py` — the semantic task-understanding matrix
  (including the `about cars` regression), generalization to unseen wording,
  Task IR robustness against malformed output, validation, planning, and
  verification.
- `tests/test_task_pipeline_e2e.py` — the live Brain → interpreter → validator →
  planner → executor → verifier chain with a fake model and fake tools, including
  model-supplied tasks, rejected invented capabilities, malformed-output
  fallback, and verification recording.
- `tests/test_command_understanding.py` — language variations and parameter
  preservation.
- `tests/test_recovery_and_validation.py` — failure classification and bounded
  recovery.

See [TESTING.md](TESTING.md) for a full description of the test strategy.

## Roadmap

See `ROADMAP.md` for the long-term Atlas plan and `ARCHITECTURE.md` for module
boundaries and design principles.
