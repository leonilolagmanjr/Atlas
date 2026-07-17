# Atlas V3.2 TODO

- [ ] Read/confirm current architecture: Brain/Planner/Executor and models
- [x] Implement rule-based intent classifier (no LLM)

- [x] Populate `ExecutionContext.intent` in `Brain.process()` before planning


- [ ] Update `Planner` to generate different plans/step actions based on intent
- [ ] Extend `Evidence` model to support first-class evidence fields (without breaking current behavior)
- [ ] Update `Executor` retrieval step to populate new evidence fields
- [ ] Improve logging (intent, plan, retrieval strategy, evidence summary)
- [ ] Introduce provider abstraction scaffolding (base + ollama provider) and refactor `llm.py` usage to go through provider
- [ ] Keep existing behavior fully functional
- [ ] Update README.md, ARCHITECTURE.md, ROADMAP.md to reflect V3.2 changes
- [ ] Run syntax/import checks and a few manual CLI questions for multiple intents

