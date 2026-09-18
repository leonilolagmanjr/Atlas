# Atlas Cleanup Report

This report records the full codebase cleanup and optimization pass performed on the
current Atlas implementation. It is a point-in-time record, not a capability
reference — [README.md](README.md) remains the current architecture authority.

Method: the entire repository was audited first (entry points, brain, reasoning,
planner, interpreter, executor, tools, web, retrieval/RAG, memory, models, computer
control, logging, utilities, tests, scripts, configuration, dependencies), then
every suspicious module was traced by import, dynamic reference, string reference,
script/CLI use, and configuration use before anything was removed. The code — not
the documentation — was treated as the source of truth.

## Architecture Before

The active runtime flow was already the Task-IR pipeline:

```
CLI / FastAPI + React
  -> Brain
  -> SemanticTaskInterpreter (reasoning/task_interpreter.py)  -> Task IR
  -> TaskValidator -> TaskPlanner                              -> ExecutionPlan
  -> ReasoningEngine (source routing, evidence, answer)        OR Executor
  -> Executor -> ToolRouter -> PermissionEngine -> registered tool
       -> observations / verifier / bounded recovery
```

Alongside it sat older, partially-overlapping layers that were retained "for
compatibility" but were no longer wired into the request path:

- `intent_classifier.IntentClassifier` — rule-based coarse intents used only by tests.
- `planner.Planner` legacy `_plan_legacy` branches (`COMPUTER`, `WEB_SEARCH`,
  `APPLICATION`, `WRITE_APPLICATION`, `COMPARE`, `COUNT`, `SUMMARIZE`, …) — reachable
  at runtime only through `intent="UNKNOWN"`, so most branches were test-only.
- `planner.Planner.plan_with_llm` — an LLM planning path no runtime caller used.
- `reasoning/interpreter.SemanticInterpreter` — a flat-intent interpreter that
  `Brain` **constructed but never used** (dead state; `Brain` separately builds a
  `SemanticTaskInterpreter`). Three tests were injecting an interpreter into this
  unused slot and silently had no effect.
- `reasoning/document.py` and `reasoning/goal.py` — whole modules with no importers.
- A layer of development scratch files at the repository root and in `scripts/`
  (repair/probe/dump tooling and their output reports) plus three `*.orig.bak`
  backups left in the tree.

## Architecture After

The same active pipeline, with the vestigial compatibility surface removed and the
code that actually runs now unambiguous:

```
CLI / FastAPI + React
  -> Brain                                   (brain.py)
  -> SemanticTaskInterpreter                 (reasoning/task_interpreter.py) -> Task
  -> TaskValidator -> TaskPlanner            (reasoning/task_validator.py, task_planner.py)
  -> ReasoningEngine                         (reasoning/reasoning_engine.py)
        QueryRouter -> SourceSelector -> EvidenceManager -> AnswerGenerator/synthesis
     OR Executor                            (executor.py)
  -> ToolRouter -> PermissionEngine -> tool  (tools/, computer/, web.py)
```

`reasoning/interpreter.py` and `planner.py` remain because their deterministic
pieces are still exercised by the test suite and are documented as legacy/
compatibility paths — see *Remaining Technical Debt*.

## Removed

Files deleted (all confirmed to have no runtime, dynamic, script, or config
references):

| File | Why |
| --- | --- |
| `reasoning/document.py` | No importers anywhere; superseded by `content_formatting.py` + `tools/format.py` for document structuring/rendering. |
| `reasoning/goal.py` | No importers anywhere; only referenced by a deleted scratch probe. |
| `repair_report.txt`, `_report_reasoning.txt`, `pytest_failures.txt` | Stale one-off diagnosis/repair outputs committed to the repo root. |
| `scripts/_dump_ctx.py`, `_dump_fix.py`, `_dump_gt.py`, `_probe_goal.py`, `_probe_s25.py`, `_repair_reasoning.py`, `_repair_reasoning_tests.py`, `_s25_dump.py`, `repair_report_gen.py` | Ad-hoc development probes/repair scripts with no callers. |
| `scripts/_fix_report.txt`, `_gt_report.txt`, `_repair_report.txt`, `_s25_dump.txt` | Output files produced by the deleted scratch scripts. |
| `reasoning/query_router.py.orig.bak`, `reasoning/reasoning_engine.py.orig.bak`, `tests/test_reasoning_behavior.py.orig.bak` | Editor/merge backups accidentally left in the tree. |

Dead code removed from active files:

- `brain.py`: the unused `self._interpreter` attribute and its `SemanticInterpreter`
  import/constructor parameter (never read after construction). Three test helpers
  were updated to stop passing a value into that unused slot — behavior is
  unchanged because `Brain` already builds its own `SemanticTaskInterpreter`.
- `tools/format.py`: a stray duplicate `import re` at end of file and unused
  imports (`Optional`, `ContentType`, `DestinationType`, `format_content`,
  `format_content_with_repair`).
- `executor.py`: unused `CompletionCriteria`, `EvidenceState` imports.
- `reasoning/answer_generator.py`: unused `DocumentStructure`, `synthesize_and_format`
  imports and a redundant local `logger = logging.getLogger(__name__)` shadow.
- `reasoning/synthesis.py`: unused `EvidenceSource` import.
- `reasoning/reasoning_engine.py`: unused `infer_retrieval_task`, `score_source`,
  `RETRIEVAL_GOALS`, `CONTENT_TYPES`.
- `reasoning/task_interpreter.py`: unused `EvidenceSource`, `RETRIEVAL_GOALS`,
  `CONTENT_TYPES`, `SOURCE_TYPE_CLASSES`.
- `reasoning/recovery.py`: unused `Observation` import.
- `reasoning/interpreter.py`: unused `Any` import.
- `memory/memory_manager.py`: unused `MAX_RETAINED_MESSAGES` import.
- `memory/context_builder.py`: unused `List` import.
- `memory/session_manager.py`: unused `Optional` import.
- `memory/storage.py`: unused `Any` import.
- `content_formatting.py`: unused `Mapping` import.
- `intent_classifier.py`: unused `Optional` import.
- `config.py`: unused `MAX_CONTEXT_TOKENS` setting (referenced only in its own file).
- `tests/test_reasoning_behavior.py`: stray UTF-8 BOM removed (it was the only
  file in the repo with a leading BOM and could break tooling).

## Consolidated

- **Interpreter wiring `Brain` previously held two interpreter references — one
  used (`_task_interpreter`) and one dead (`_interpreter`). The dead one was
  removed so there is a single interpretation entry point. No runtime logic changed.
- **Formatting responsibility.** `reasoning/document.py` (deleted) duplicated the
  document-structuring/rendering responsibility already owned by
  `content_formatting.py` + `tools/format.py`; the canonical implementation remains
  the formatting pipeline.

## Optimized

- **Deleted-document handling in incremental indexing** (`indexer.py`). Previously a
  deleted knowledge PDF kept its hash in `index_state.json` and its chunks in
  ChromaDB forever, so removed documents kept being retrieved. Startup ingestion now
  drops state entries (and their vectors) for documents that no longer exist. New
  and modified documents were already handled by hash and are unchanged.
- **Startup/import surface.** Removing two dead modules and the scratch scripts
  reduces what can be imported/scanned at startup; no new initialization was added.
- Unused imports removed above eliminate import-time work in the affected modules.

## Dependencies Removed

None. Every entry in `requirements.txt` is genuinely required:

- `chromadb`, `sentence-transformers`, `pypdf`, `ollama`, `fastapi`, `uvicorn` are
  imported directly.
- `httpx` is **not** imported by Atlas directly, but the `ollama` client declares it
  as a required transitive dependency (`ollama` requires `httpx>=0.27`), so it is
  kept deliberately rather than removed as "unused".

## Latent Bugs Fixed

- **`task.completion_criteria` crash** (`executor.py`). `_verify_completion` set
  `task.completion_criteria.all_met = True` unconditionally when verification
  passed, but `completion_criteria` is `None` for the normal interpreter-produced
  task. This raised `AttributeError` inside `Executor.execute` and could escalate to
  a whole-request `UNKNOWN_RESPONSE` failure. Now guarded by
  `if task.completion_criteria:`.
- **Frontend production build was broken.** `npm --prefix frontend run build`
  (`tsc -b && vite build`) failed on a `thumbnail_url` type mismatch between
  `WebResult` (`string | null`) and the `WebResults` component prop (`string`).
  The prop type was widened to `string | null`; the build now passes. This was
  pre-existing and unrelated to the code cleanup.

## Preserved

All currently working capabilities were left intact and verified: structured task
interpretation with deterministic fast paths, source-aware reasoning and citation,
incremental PDF indexing + hybrid retrieval, bounded filesystem tools, permission-
gated application launch/text entry, read-only Windows/process/PowerShell inspection,
task-aware web research with content-type validation, content formatting/repair,
conversation memory and sessions, API task snapshots, and the React control room.

## Tests Performed

- `python -m unittest discover -s tests` — **346 passed, 1 skipped, exit 0**
  (baseline before changes: same). Run repeatedly after each change group.
- `python -m compileall` over the tree — clean.
- Full import check of all runtime packages/modules — clean.
- `python scripts/smoke_reasoning.py` (offline reasoning smoke: self-description,
  general knowledge, current-question → web path, action delegation, memory recall,
  file lookup limitation, general conversation) — passed.
 Representative scenario tests: Notepad poem about cars (local, not a web search),
  personal-document lookup staying local, hybrid search-then-write composition,
  plain question not reaching for the web, ambiguity → clarification — all passed.
- `npm --prefix frontend run build` — passes after the type fix.

Most suite coverage uses fakes/mocks; this pass did not run live Ollama, live network,
or live mutating application/file tests.

## Remaining Technical Debt (intentionally not changed)

These require an architectural decision larger than a cleanup pass and are left
untouched on purpose:

1. **`intent_classifier.py` and the legacy `Planner` branches**
   (`_plan_legacy`, `plan_with_llm`) are exercised only by tests, not by the runtime
   path. They are retained as documented compatibility/deterministic fallback code.
   Removing them means deleting their tests and confirming no external caller — a
   separate decision.
2. **Two structured-document representations** — `reasoning/synthesis.py`
   `DocumentStructure` and `content_formatting.py` `ContentDocument` — overlap in
   responsibility. Both are independently exercised on live paths, so merging them
   is a design change, not cleanup.
3. **`providers/__init__.py` re-exports** `BaseProvider`/`OllamaProvider` although
   callers import the submodules directly. Left as an idiomatic package facade to
   avoid churn.
4. **Approval/verification hardening** (argument-bound approvals, atomic resume,
   independent effect observation) — unchanged; it is roadmap work, not dead code.

## Recommended Next Step

Complete the legacy planning consolidation that this pass stopped short of: verify
that `intent_classifier.py` and the unused `Planner._plan_legacy` /`plan_with_llm`
paths have no external callers, then remove them and their now-redundant tests, so
the Task-IR interpreter/validator/planner is the single planning path. This is the
highest-value follow-up because it removes the last significant "old architecture"
remnant and matches the documented design. Do **not** implement it as part of this
cleanup.
