# Atlas

Atlas is a modular, local-first AI operating system foundation.

It is not designed as a monolithic chatbot. Atlas separates orchestration,
retrieval, indexing, vector storage, prompting, logging, and LLM access so each
part can evolve independently.

## Current Version

Atlas is currently moving through the V3 Brain milestone.

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

If all retrieval attempts fail, Atlas returns:

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
  database/            Local generated ChromaDB data, ignored by git
```

## Responsibilities

`atlas.py` is only the entry point. It initializes logging, loads prompts,
indexes the knowledge folder, creates the vector store and Brain, and runs the
chat loop.

`brain.py` creates the execution context, asks the planner for a plan, passes
that plan to the executor, and returns the final response.

`planner.py` creates a deterministic two-step plan for every request:
retrieve knowledge, then generate a response.

`executor.py` owns step-by-step execution. It updates the shared execution
context, runs retrieval, decides whether there is enough evidence to call the
LLM, and records step status and failures.

`models.py` contains shared dataclasses for execution context, plans, steps,
planner decisions, evidence, and retrieval results.

`knowledge_search.py` owns retrieval. It expands queries, performs staged
retrieval, merges hits, evaluates confidence, formats context, and logs
diagnostics.

`vector_store.py` owns ChromaDB access. It adds chunks, queries embeddings, and
exposes stored chunks for keyword and metadata retrieval.

`llm.py` only talks to Ollama.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install project dependencies used by the current codebase:

```powershell
pip install chromadb sentence-transformers pymupdf requests
```

Install Ollama and pull the configured model:

```powershell
ollama pull qwen2.5:7b
```

## Running

Place PDF knowledge files in `knowledge/`, then run:

```powershell
python atlas.py
```

Atlas will index changed documents at startup and then enter the CLI chat loop.

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

`MIN_SIMILARITY` is still used as one signal in the adaptive retrieval policy,
not as a single hard gate before retries.

## Generated Files

The following are local runtime artifacts and should not be committed:

- `__pycache__/`
- `.venv/`
- `database/`
- `*.log`
- `.env`

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
