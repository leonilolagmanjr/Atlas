# TODO_MEMORY.md — Modular Memory System (Atlas)

## Step 1 — Repository alignment
- [x] Read core pipeline files: atlas.py, brain.py, planner.py, executor.py, models.py, config.py, logger.py
- [x] Read prompt template: prompts/retrieval.txt
- [x] Confirm architecture constraints: no UI assumptions, memory as separate subsystem

## Step 2 — Memory module implementation (new files)
- [x] Add `memory/` package
- [x] Implement message/session models
- [x] Implement filesystem persistence layer
- [x] Implement session manager (CRUD + active session)
- [x] Implement context builder (recent message window)
- [x] Implement summarizer placeholder (interface only)
- [x] Implement MemoryManager façade API


## Step 3 — Configuration + logging
- [x] Add memory settings to config.py
- [x] Ensure memory modules use existing logger style


## Step 4 — Prompt & executor integration

- [x] Update prompts/retrieval.txt to include `{conversation_history}` placeholder
- [x] Update executor.py to:
  - append user/assistant messages via MemoryManager
  - request conversation history from MemoryManager
  - pass history into prompt


## Step 5 — Brain/atlas plumbing + CLI
- [x] Add MemoryManager initialization in atlas.py
- [x] Update Brain/Executor constructors to accept MemoryManager (dependency injection)
- [x] Replace/extend atlas REPL to support commands:
  - /new /list /open /delete /rename /history /export /import /clear /help


## Step 6 — Smoke tests
- [x] Run `python atlas.py` and verify:
  - restart persistence
  - session switching
  - `/history` output
  - basic memory window behavior


## Step 7 — Documentation
- [x] Update ARCHITECTURE.md with Memory subsystem location


