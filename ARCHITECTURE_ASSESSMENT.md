# Atlas Architecture Assessment

**Audit date:** 2026-09-16
**Audited version:** V3.2 codebase

## Current architecture

The implemented runtime is a local retrieval assistant with deterministic orchestration:

```text
CLI -> Brain -> IntentClassifier -> Planner -> Executor
                                      |
                                      +-> Knowledge retrieval -> ChromaDB
                                      +-> Evidence-gated Ollama response
                                      +-> Conversation Memory
```

- `atlas.py` initializes logging, indexing, vector storage, memory, Brain, and the CLI loop.
- `brain.py` creates one `ExecutionContext`, classifies intent, creates a plan, and delegates execution.
- `planner.py` creates deterministic retrieval and response plans. It does not select general tools.
- `executor.py` runs a fixed handler map for retrieval, evidence merging, and LLM generation.
- `knowledge_search.py` owns staged semantic, keyword, and metadata retrieval with diagnostics.
- `indexer.py`, `document_loader.py`, `chunker.py`, and `vector_store.py` form the local document pipeline.
- `memory/` provides persisted conversational sessions and prompt history.
- `providers/` provides an Ollama provider interface, but provider selection is still hard-coded by `llm.py`.

## What works

- CLI conversation and session commands.
- Ollama-backed local generation.
- PDF indexing with incremental file hashing.
- ChromaDB storage and hybrid retrieval.
- Retrieval confidence gating that can skip the LLM when evidence is weak.
- Deterministic intent classification and plan execution.
- Persisted conversation memory with session management.
- Centralized configuration and logging.

## Partial or incomplete

- `ExecutionContext` tracks retrieval and response state, but not a full task lifecycle, tool calls, permissions, observations, verification, or serializable results.
- The plan/step model is useful, but actions are string names resolved by a private executor handler map.
- The provider abstraction exists, while the public LLM facade still constructs `OllamaProvider` directly.
- Memory is conversation/session memory; long-term memory, provenance, confidence, and retrieval are not implemented.
- Error handling records failed plan steps but has no recovery policy or user cancellation state.
- Filesystem, process, system, and installed-application inspection tools now exist behind the tool contracts. Executable launch is available only through the permission-aware router and executor, but planner intent and CLI approval UX are not yet implemented.
- Automated contract tests now cover the initial runtime and computer-tool slice.

## Planned or missing

- Tool metadata, schemas, validation, structured results, and registry/router.
- Permission policy and SAFE, CONFIRM, and AUTONOMOUS modes.
- Computer runtime: filesystem, applications, processes, system inspection, and controlled PowerShell.
- Observer and verifier as first-class components.
- Internet runtime with search, webpage retrieval, downloads, provenance, and source trust.
- Approval-aware multi-tool planning and failure recovery.
- GUI, voice, and vision interfaces.

## Reusable architecture

The existing Brain/Planner/Executor boundary, shared dataclasses, evidence model, provider interface, dependency injection points, memory facade, and retrieval diagnostics should be preserved. Computer and internet capabilities should be added as tools behind contracts rather than embedded in `Brain` or the LLM provider.

## Gap analysis

| Vision requirement | Current state | Migration direction |
| --- | --- | --- |
| Structured execution context | Retrieval-centric dataclass | Add task lifecycle fields and serialization helpers |
| Standardized tools | Fixed executor handlers | Add tool contract, result model, registry, and router |
| Permissions | Not present | Add explicit risk levels, policy decisions, and confirmation hooks |
| Computer control | Inspection plus permission-gated executable launch routed through Executor, with explicit-path planning and CLI approval | Resolve application names to trusted executable paths, then add controlled writes |
| Terminal safety | Not present | Add validated PowerShell tool with structured results |
| Observation and verification | Not present | Add post-action observation and evidence-based verification |
| Internet access | Not present | Add source-aware search, retrieval, and download tools |
| Testing | No test suite directory | Add focused contract and integration tests per phase |

## Proposed target architecture

```text
User -> Brain -> Planner -> Tool Router -> Permission Engine
                                      -> Executor -> Observer -> Verifier
                                           |             |
                         Computer / Internet / Knowledge runtimes
```

`Brain` owns request lifecycle orchestration. `Planner` produces structured steps. The `ToolRegistry` discovers tools and the `ToolRouter` resolves a plan step to a tool. The permission engine evaluates before execution. The executor invokes only validated, permitted tools. The observer records actual results, and the verifier determines whether the requested outcome is supported by evidence.

## Migration strategy

1. Preserve the existing retrieval plan and CLI behavior while introducing additive models.
2. Add contract tests before wiring new tools into the default runtime.
3. Introduce a registry and route existing knowledge retrieval through it only after compatibility tests pass.
4. Add permission evaluation in dry-run mode before any mutating computer capability.
5. Add read-only Windows tools, then controlled writes, with verification for each action. The initial filesystem, process, and system inspection tools are now implemented but remain opt-in until routing is added.
6. Add internet tools with source provenance and download approval before package installation.
7. Keep GUI, voice, and vision as adapters over the same planning and tool contracts.

## Phase 1 implementation plan

Phase 1 is an audit and contract-hardening phase. It should deliver:

- This assessment and an accurate status in the project documentation.
- A serializable execution context with explicit task, step, tool, permission, observation, and verification state.
- A tool contract and structured tool result model, without enabling OS execution yet.
- A permission vocabulary and policy decision model, defaulting to deny when no policy is supplied.
- A small test suite covering context serialization, tool validation, and permission decisions.

The next implementation phase can then add safe, read-only filesystem and system inspection tools without changing Brain's public entry point.
