# Changelog

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
- Local FastAPI adapter for Atlas health, system, tools, applications, and task endpoints.
- React/Vite Atlas control room with command submission, task polling, approval/cancellation UI, applications, tools, system, and backend-dependent states.
- `requirements.txt` for the Python runtime and API dependencies.
- Automated API smoke coverage and documented testing procedures.

### Changed

- Brain exposes the latest execution context to API adapters.
- API tasks remain serialized by the current worker pool, while Brain now tracks approval-paused contexts by task ID.
- Documentation now distinguishes implemented computer capabilities from planned internet, terminal, and knowledge-management APIs.

### Verified

- 44 Python tests pass.
- One real Windows read-only PowerShell pipeline passed through the Atlas provider.
- Frontend TypeScript/Vite production build passes.
- Live API and browser smoke checks pass for health, system, tools, applications, task approval pause, and cancellation.
- Ollama 0.31.2 and `qwen2.5:7b` are installed locally.
