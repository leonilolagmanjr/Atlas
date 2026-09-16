# Atlas

Atlas is a local-first AI runtime foundation for retrieval, planning, memory,
and eventually controlled computer and internet tools.

It is not designed as a monolithic chatbot. Atlas separates orchestration,
retrieval, indexing, vector storage, prompting, logging, and LLM access so each
part can evolve independently.

## Current Version

Atlas is currently implemented as the V3.2 Brain foundation (intent-based
deterministic planning, staged hybrid retrieval, and conversation memory).

The computer-agent architecture is being added incrementally. Permission-gated
local computer inspection and explicit executable launch are implemented;
internet access, terminal execution, observation, and result verification are
not yet implemented. See [ARCHITECTURE_ASSESSMENT.md](ARCHITECTURE_ASSESSMENT.md)
for the verified audit and migration plan.

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
  brain.py             Request orchestration
  chunker.py           Text chunking
  config.py            Central configuration
  document_loader.py   Document reading
  executor.py          Sequential execution plan runner
  indexer.py           Incremental indexing
  knowledge_search.py  Retrieval strategy and confidence policy
  llm.py               Ollama communication
  logger.py            Logging setup
  models.py            Shared execution dataclasses
  planner.py           Deterministic plan creation
  vector_store.py      ChromaDB operations
  prompts/
    system.txt
    retrieval.txt
  knowledge/
    *.pdf
  tools/               Tool contracts and runtime integrations (in progress)
  computer/            Read-only local computer tools (in progress)
  database/            Local generated ChromaDB data, ignored by git
```

## Responsibilities

`atlas.py` is only the entry point. It initializes logging, loads prompts,
indexes the knowledge folder, creates the vector store and Brain, and runs the
chat loop.

`brain.py` creates the execution context, asks the planner for a plan, passes
that plan to the executor, and returns the final response.

`planner.py` creates deterministic retrieval/response plans and explicit
application launch plans when the request contains an executable path.

`executor.py` owns step-by-step execution. It updates the shared execution
context, runs retrieval or routed tools, pauses for confirmation when required,
decides whether there is enough evidence to call the LLM, and records step
status and failures.

`models.py` contains shared dataclasses for execution context, plans, steps,
planner decisions, evidence, and retrieval results.

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
pip install chromadb sentence-transformers pypdf ollama
```

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

Open `http://localhost:5173`. The command workspace, durable task polling, task-bound approval
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
.\.venv\Scripts\python.exe -c "import atlas, brain, chunker, config, document_loader, executor, indexer, knowledge_search, llm, logger, models, planner, vector_store; print('imports ok')"
```

## Roadmap

See `ROADMAP.md` for the long-term Atlas plan and `ARCHITECTURE.md` for module
boundaries and design principles.
