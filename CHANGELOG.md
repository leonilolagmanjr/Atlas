# Changelog

## 2026-09-16

### Added

- Local FastAPI adapter for Atlas health, system, tools, applications, and task endpoints.
- React/Vite Atlas control room with command submission, task polling, approval/cancellation UI, applications, tools, system, and backend-dependent states.
- `requirements.txt` for the Python runtime and API dependencies.
- Automated API smoke coverage and documented testing procedures.

### Changed

- Brain exposes the latest execution context to API adapters.
- API tasks are serialized because Brain currently owns one pending approval context.
- Documentation now distinguishes implemented computer capabilities from planned internet, terminal, and knowledge-management APIs.

### Verified

- 20 Python tests pass.
- Frontend TypeScript/Vite production build passes.
- Live API and browser smoke checks pass for health, system, tools, applications, task approval pause, and cancellation.
- Ollama 0.31.2 and `qwen2.5:7b` are installed locally.
