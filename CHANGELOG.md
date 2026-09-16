# Changelog
## 2026-09-16 (documentation alignment)
### Changed
- Aligned ARCHITECTURE.md with the implemented runtime: corrected the stale
  "Internet search, webpage retrieval, downloads, PowerShell execution, and GUI
  automation are not implemented" claim, expanded the current-version
  capability list, and marked the Version 3-6 slices as implemented/partial
  instead of future work.
- Aligned ROADMAP.md current status, verified-runtime list, and V3 capabilities
  with implemented intent classification, PowerShell, application, and web
  features; annotated the migration checklist with done/planned state.
- Aligned README.md setup to `pip install -r requirements.txt`, refreshed the
  project structure tree, and documented `intent_classifier.py`, `web.py`, and
  the `/approve` and `/deny` CLI commands.
- Aligned TODO.md, SECURITY.md, and TESTING.md with implemented capabilities.

### Verified
- 55 Python tests pass (`python -m unittest discover -s tests`).

## 2026-09-16
### Added

- Atomic JSON-backed API task history in `database/tasks.json`.
- Task-ID-bound approval and denial, with explicit interrupted-task handling on API restart.
- General tool knowledge records, JSON import, capability discovery, and deterministic tool candidates.
- Read-only PowerShell provider with command validation, risk blocking, bounded subprocess execution, and structured results.
- PowerShell planning for process, memory, service, network, and system inspection requests.
- Structured PowerShell JSON summaries, bounded raw task output, and expandable frontend output inspection.
- One-click Windows launcher for the FastAPI backend and React frontend.
- Natural-language named application launch planning and trusted Discord resolution.
- Confirmation-gated application text entry for requests such as writing a short poem in Notepad.
- Read-only public web search with provider fallback, URL safety checks, source provenance, and frontend citations.
- Direct YouTube search with relevant video filtering, watch links, thumbnails, and responsive frontend result cards.
- Local FastAPI adapter for Atlas health, system, tools, applications, and task endpoints.
- React/Vite Atlas control room with command submission, task polling, approval/cancellation UI, applications, tools, system, and backend-dependent states.
- `requirements.txt` for the Python runtime and API dependencies.
- Automated API smoke coverage and documented testing procedures.

### Changed

- Brain exposes the latest execution context to API adapters.
- API tasks remain serialized by the current worker pool, while Brain now tracks approval-paused contexts by task ID.
- Documentation now distinguishes implemented computer capabilities from planned internet, terminal, and knowledge-management APIs.
- Documentation now includes a current architecture handoff and explicit next-agent boundaries.

### Verified

- 55 Python tests pass.
- One real Windows read-only PowerShell pipeline passed through the Atlas provider.
- Frontend TypeScript/Vite production build passes.
- Live API and browser smoke checks pass for health, system, tools, applications, task approval pause, and cancellation.
- Ollama 0.31.2 and `qwen2.5:7b` are installed locally.
