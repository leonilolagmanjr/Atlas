# Atlas V3.2 TODO

## Computer-agent foundation

- [x] Audit current runtime and document implementation gaps
- [x] Add serializable task-oriented ExecutionContext fields
- [x] Add Tool contract, structured ToolResult, and registry
- [x] Add permission levels, policy modes, and confirmation decisions
- [x] Add focused automated tests for the new contracts
- [x] Add read-only filesystem, process, and system tools after the contracts are stable
- [x] Add a computer tool bootstrap/registry for the default runtime
- [x] Add application inspection before any application-launch capability
- [x] Add controlled application launch with explicit permission
- [x] Wire ToolRegistry and ToolRouter into Brain/Executor plan execution
- [x] Add planner intent and approval UX for explicit application actions
- [ ] Resolve application names to trusted executable paths

- [x] Read/confirm current architecture: Brain/Planner/Executor and models
- [x] Implement rule-based intent classifier (no LLM)
- [x] Populate `ExecutionContext.intent` in `Brain.process()` before planning

- [x] Provider abstraction scaffolding (base + ollama provider)
- [x] Planner and executor module boundaries are in place (deterministic planner + sequential executor)

- [x] Audit and align documentation with implementation (README.md, ARCHITECTURE.md, ROADMAP.md)
- [x] Decide whether to add missing doc files (INSTALL.md, CONFIGURATION.md) if they are absent

- [ ] Run syntax/import checks and a few manual CLI questions for multiple intents

