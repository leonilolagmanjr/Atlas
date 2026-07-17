# Atlas V3.2 TODO

- [x] Read/confirm current architecture: Brain/Planner/Executor and models
- [x] Implement rule-based intent classifier (no LLM)
- [x] Populate `ExecutionContext.intent` in `Brain.process()` before planning

- [x] Provider abstraction scaffolding (base + ollama provider)
- [x] Planner and executor module boundaries are in place (deterministic planner + sequential executor)

- [x] Audit and align documentation with implementation (README.md, ARCHITECTURE.md, ROADMAP.md)
- [x] Decide whether to add missing doc files (INSTALL.md, CONFIGURATION.md) if they are absent

- [ ] Run syntax/import checks and a few manual CLI questions for multiple intents

