# Atlas

Atlas is a local-first Python assistant with local Ollama generation, document retrieval, conversation memory, read-only research, and permission-aware Windows tools. A FastAPI adapter and React/Vite control room expose the same runtime. It is not a general autonomous computer agent or a security sandbox.

**Documentation authority:** This README is the current architecture, installation, configuration, testing, and roadmap reference. The other architecture/setup/status documents link here rather than maintaining competing descriptions. [SECURITY.md](SECURITY.md) retains detailed security policy and limitations. [CLEANUP_REPORT.md](CLEANUP_REPORT.md) records the last codebase cleanup pass (what was removed/consolidated/optimized), not current capability or test-status guarantees.

## Contents

- [Current capabilities](#current-capabilities)
- [Architecture](#architecture)
- [Local computer vision ("Eyes")](#local-computer-vision-eyes)
- [Sources and evidence](#sources-and-evidence)
- [Tools and execution](#tools-and-execution)
- [Memory and persistence](#memory-and-persistence)
- [Installation](#installation)
- [Running](#running)
- [Configuration](#configuration)
- [API and frontend](#api-and-frontend)
- [Testing](#testing)
- [Limitations and security](#limitations-and-security)
- [Roadmap](#roadmap)

## Current capabilities

- Structured semantic request interpretation using local Qwen, with deterministic fast paths and fallback.
- Source-aware answers from local model knowledge, indexed documents, permitted files, current public web results, conversation history, runtime introspection, and system observations.
- Incremental PDF indexing, ChromaDB storage, and staged semantic/keyword/metadata retrieval.
- Bounded local file listing, search, UTF-8 text reading, PDF text extraction, and content search.
- Registered, permission-aware application launch/text entry and filesystem write/create/move/copy actions.
- Read-only Windows process/system/application inspection and validated PowerShell inspection.
- Closed-loop UI observation for computer actions: window inventory, application/window/control matching, and control text reading, so Atlas can observe whether an action it performed actually changed the visible application state before claiming success; failures are classified as execution vs verification vs recovery problems, and bounded recovery can retry the same capability with a different delivery mechanism or wait time.
- **Local computer vision ("Eyes")** — a layered, local-first perception subsystem (Windows UI Automation → local OCR → image processing → optional local VLM) that builds a typed structured visual state, plus permission-gated, observation-validated mouse/keyboard interaction, so Atlas can observe and act on applications and websites it has no dedicated API integration for. Screen capture, window detection, UIA, OCR, image preprocessing, element detection, mouse/keyboard, and observation/verification all work without any external API; the local VLM is optional and everything degrades gracefully.
- Dependency-ordered action plans, named output references, sequential execution, approval pause/resume, and bounded recovery.
- Persisted conversation sessions, durable API task snapshots, and sanitized reasoning-stage/provenance displays.
- **Persistent experience memory** — an explicit Success/Failed feedback loop that records compact task experiences, ranks and retrieves relevant ones for later comparable requests, and supplies them to planning as *supporting context* that never overrides the current instruction. Completion criteria come from deterministic verification, not a model's claim; feedback and experiences survive a restart; failures and user corrections are retained as reusable lessons.

These are implemented paths, not a guarantee that arbitrary natural-language requests work. Model interpretation, search-provider availability, application/window resolution, OCR/VLM quality, and evidence quality affect results. Generic software installation, downloads, unrestricted GUI automation, and voice are not implemented. Local vision generalizes to applications without a dedicated integration only insofar as the layered perception can see them; it is not a claim that every GUI is fully automatable.

## Architecture

Atlas separates request interpretation, source selection, evidence evaluation, answer generation, and consequential execution:

```text
CLI / FastAPI + React
  -> Brain: request context and conversation history
  -> Semantic Task Interpreter: natural language -> Task IR
  -> Intent Engine 2.0: goal / desired outcome / references / capability requirements
  -> Task Validator: registry, parameters, dependencies, references, risk
  -> Experience Memory: retrieve bounded, ranked context from comparable past tasks
  -> ReasoningEngine
       -> QueryRouter + SourceSelector: ordered information sources
       -> EvidenceManager: collect and evaluate usable evidence
       -> AnswerGenerator: grounded, direct, clarification, or limitation answer
       OR delegate any Task.actions to the existing action pipeline
  -> TaskPlanner + concrete plan validation
  -> Executor -> ToolRouter -> PermissionEngine -> registered tool
       -> observations / verifier / bounded recovery
  -> response, memory, task state, API snapshot
  -> user feedback (Success / Failed) -> experience record -> experience memory
```

`Brain` invokes the reasoning engine for eligible validated tasks when enabled. The engine can finish an informational request without building an execution plan. It serves a task directly when its actions are **all read-only research** (`web.search`/`web.fetch`, and non-mutating `filesystem.list/search/read/metadata/search_content`) and answers from the retrieved evidence; any task that mutates state or controls applications—including a hybrid such as "search the web for X and write it into Notepad"—is preserved and delegated to the existing validator/planner/executor path rather than consumed as an answer-only request. Invalid tasks are not given a source-execution bypass.

### Task interpretation

`models_task.py` defines `Task` and `TaskAction`, the boundary between model interpretation and deterministic execution. A task includes its goal, task type, entities, constraints, actions, confidence, and clarification state, plus structured reasoning fields such as `request_type`, `sources`, `current_information_required`, and `response_mode`.

The semantic interpreter can propose source/request-type fields directly. Routing also uses deterministic signals and conversation context; it is not solely a keyword-to-tool switch. Instructional questions must remain distinct from requests to perform an action. Unsupported or ambiguous targets should produce clarification or a limitation, not an invented tool.

An action identifies a registered capability, parameters, dependencies, and optionally a named output. For example, content generation can publish `generated_text`, which `applications.write_text` consumes as `$generated_text`, and a web search publishes `search_results`, which `filesystem.write` consumes to save the result list. Topic, tone, length, and destination are separate parameters: in “write a poem about cars in Notepad,” cars is the topic and Notepad is the application. A file destination is parsed distinctly from an application destination, and a move/copy target folder (“move the largest pdf to Documents”) is the destination, not the search location.

The model proposes structure, not shell execution. Validation checks capability existence, required parameters/types, dependency identifiers, and output references. Planning orders dependencies, but cycle validation and order-independent reference validation remain incomplete. The generated-text handoff works; generic `$name` output resolution still has legacy gaps and must not be treated as reliable arbitrary dataflow.

### Semantic topic decomposition and query normalization

Natural-language instruction words must not leak into a tool call. `reasoning/topic_extraction.py` is the semantic boundary that converts a request span into a structured `TopicReading` — the raw subject, the connector and framing noun that were stripped, the content-type noun that was separated, the clean subject, a confidently normalized subject, and the research query — so that:

```text
USER:            research about sykrim and write it in notepad
SEMANTIC:        action = research   topic = sykrim   destination = Notepad
TOOL INPUT:      web.research("sykrim")            (not "about sykrim")
```

It is semantic, not a phrase table: it reasons about the *role* of a span (action verb, connector, framing head noun, content-type noun, qualifier, content). A connector (`about`, `on`, `regarding`, `concerning`, `related to`, `re:`), a framing head noun (`information`, `info`, `details`, `facts`, `overview`), and a leading kind-of-result noun (`videos`, `articles`) are stripped only when they introduce the subject — an inner connector is preserved, so “the concept of a story about Skyrim” keeps its second “about”, and a bare `on` is stripped only as a lead-in (never from “the effects of X on Y”). Genuine qualifiers are kept (“best Skyrim mods”, “Skyrim survival mode”), and compound subjects stay whole (“how Skyrim's leveling system works”, “the history of Skyrim”).

The interpreter applies this reading on the search, research, create, and informational paths and records it on the Task IR (`raw_topic`, `normalized_topic`, `research_query`, `topic_reading`); the query handed to `web.search`/`web.research` is built from the subject, never from the raw instruction. Typo correction is conservative and happens at the semantic level only when a known entity is one edit away (a capitalized transposition such as “Skryim” → “Skyrim”); an uncertain term is preserved rather than silently invented, and a real English word is never “corrected” (“mode” → “code” is refused).

**Recovery.** If a first research pass returns nothing usable (“No relevant web content could be read for: about sykrim”), the reasoning engine performs one bounded query-normalization retry: it prefers the task's own normalized query/topic (which reflects the full intent), otherwise cleans the failed query itself, and only retries when the query genuinely changed. It is recorded in the execution trace and never loops. A retrieval failure is classified as `retrieval_failed` and attributed to `wrong_interpretation` in the experience layer, because the query — not the network — is the usual cause (see [Experience memory](#experience-memory-human-feedback-loop)).

### Intent Engine 2.0
Between the interpreter and the validator, the **Intent Engine 2.0** (`reasoning/intent_engine.py`) augments the Task IR with an explicit *understanding* of the request, so a small local model does not have to decide every tool call itself:

```text
USER LANGUAGE -> INTENT -> DESIRED OUTCOME -> STRUCTURED TASK -> PLAN
```

It is one component with several deterministic passes — not a swarm of agents — and it never plans, executes, or calls the model:

- **Goal vs operation vs desired outcome.** `goal` is *why* (`research_and_deliver`); `operations` are *how* (`search`, `transform`, `write`); `desired_outcome` is the end state that must be true when Atlas finishes (“a useful Avatar summary exists in Notepad”). An operation is never mistaken for the goal.
- **Context / reference resolution** (`reasoning/reference_resolver.py`). Follow-ups resolve against the previous Task IR and a bounded slice of recent output — never by stuffing the whole conversation into a prompt. “Make it shorter”, “put that in Notepad”, “which one is best?”, and “do the same thing for Interstellar” resolve deterministically (`previous_output`, `previous_task`, ordinal, transformation); an unresolvable reference is surfaced, not hallucinated.
- **Intent hypotheses.** Ambiguous requests produce a small set (1–3) of scored readings; one is selected after validation. Scores are internal — never shown to the user.
- **Two-pass understanding.** Pass 1 reads goal/object/operations/references; pass 2 (`_critique`) repairs the specific mistakes a small model makes: topic-vs-destination (“Notepad” is not the topic), retrieval-vs-action, spurious web for creative tasks, and a transformation mistaken for a new research task. Only concise structured notes are emitted, never chain-of-thought.
- **Capability-aware reasoning.** The reading proposes *semantic* requirements (`web.research`, `content.generate`, `applications.write_text`); Atlas — not the model — confirms each against the live `CapabilityRegistry` (`validate_capabilities`). The model never decides whether a tool exists.
- **Tool necessity.** `needs_web` / `needs_files` / `needs_application` are set from the *outcome*, not from nouns, so Atlas does not search the web because a topic is nameable or open an app because an application was mentioned.
- **Confidence as certainty.** Confidence reflects interpretation certainty, not model confidence theater: an under-specified target (“Get me some videos”) lowers it and raises an ambiguity, while a named entity (“Find MrBeast videos”) resolves it.
- **Memory of corrections** (`reasoning/correction_memory.py`). Reusable, high-confidence structured lessons (JSONL under `memory/`) can override a reading for a class of phrasings. Only curated lessons are stored — never every turn, never an uncontrolled vector dump.

The engine is model-agnostic: it consumes the interpreter's Task IR and works with any local model (or none, on the deterministic fast path). The `PipelineTrace` (`reasoning/diagnostics.py`) records the intent reading and a compact `TOPIC DECOMPOSITION` line (raw topic, stripped connector/framing noun, separated content noun, subject, normalized subject, research query) for observability, so an interpretation problem is distinguishable from a planning or execution problem.

### Module map

| Path | Responsibility |
| --- | --- |
| `atlas.py`, `cli.py` | Startup, indexing initialization, chat loop, session and approval commands |
| `brain.py` | Request lifecycle, interpretation/validation, reasoning integration, action delegation, early-answer persistence |
| `models_task.py`, `models.py` | Task IR, execution contexts, plans, steps, lifecycle and result models |
| `reasoning/task_interpreter.py`, `reasoning/prompts.py`, `reasoning/json_llm.py` | Semantic interpretation and structured model output handling |
| `reasoning/topic_extraction.py` | Semantic topic decomposition and query normalization: connector/framing-noun/content-noun separation, the structured `TopicReading`, and confident (never invented) typo correction |
| `reasoning/intent_engine.py`, `reasoning/reference_resolver.py`, `reasoning/correction_memory.py` | Intent Engine 2.0: goal/outcome/capability understanding, conversation reference resolution, and structured interpretation lessons |
| `reasoning/reasoning_engine.py` | Bounded source orchestration and answer/delegation decision |
| `reasoning/query_router.py`, `reasoning/source_selector.py` | Routing signals and ordered source plans |
| `reasoning/evidence_manager.py`, `evidence_models.py` | Evidence collection, provenance and sufficiency checks |
| `reasoning/answer_generator.py`, `reasoning/synthesis.py`, `reasoning/self_introspection.py` | Answer modes, structured synthesis of retrieved evidence for web-research answers, and registry/config-derived self-description |
| `reasoning/task_validator.py`, `reasoning/task_planner.py` | Task validation and dependency-ordered action planning |
| `content_formatting.py`, `tools/format.py` | Content normalization, type detection, paragraph reconstruction, destination rendering, verification, and repair |
| `executor.py`, `reasoning/verifier.py`, `reasoning/recovery.py` | Sequential execution, output publication, observations, verification and bounded recovery |
| `planner.py`, `intent_classifier.py`, `reasoning/interpreter.py` | Retained deterministic/legacy planning and compatibility paths, not the primary Task IR boundary |
| `knowledge_search.py` | Query expansion, staged retrieval, ranking and confidence decision |
| `document_loader.py`, `chunker.py`, `indexer.py`, `vector_store.py` | PDF loading, character chunks, file-hash indexing and ChromaDB access |
| `llm.py`, `providers/` | Injectable model-call boundary; current facade constructs the Ollama provider |
| `tools/` | Tool contracts, actual registry, capability descriptors, router, permissions, knowledge and discovery |
| `computer/`, `web.py` | Windows/filesystem/PowerShell providers and read-only public search/fetch/research tools |
| `computer/vision/` | Local vision ("Eyes"): `models.py` (typed `VisualState`), `perception.py` (layered orchestrator), `uia.py` (UI Automation), `ocr.py` (local OCR), `imaging.py` (capture + OpenCV processing), `providers.py` (injectable `VisionProvider`), `actions.py` (target validation/coordinate safety), `input.py` (Win32 mouse/keyboard) |
| `computer/vision_tools.py`, `computer/interaction.py` | `computer.observe`/`computer.find` observation tools and the permission-gated visual interaction tools |
| `web_content.py`, `web_research.py`, `web_task.py` | Main-content extraction, task-aware multi-attempt retrieval pipeline, and retrieval task/goal/content-type/scoring/validation models |
| `memory/`, `task_store.py` | Conversation storage and durable API task snapshots |
| `experience/` | Human feedback loop: experience records, durable store, bounded retrieval/ranking, completion evaluation, and planning context |
| `api.py`, `frontend/` | Local HTTP adapter and polling control room |
| `config.py`, `logger.py`, `prompts/` | Runtime constants, logging and prompt templates |

Keep new capabilities behind existing tool contracts; do not put subprocess execution in the interpreter, source selector, or frontend. Prefer injected providers/tools for tests and preserve the distinction between evidence and instructions.

## Local computer vision ("Eyes")

Atlas can visually observe the Windows desktop and act on what it sees without any external/cloud computer-use API. The subsystem extends the existing Task IR, planner, executor, permission system, verifier, and recovery — it does **not** add a second agent framework, planner, or permission system.

The loop is the existing one, extended with a perception stage:

```text
Windows
  -> Screen Capture (region / active-window / full-screen, downscaled)
  -> Deterministic Perception
       Layer 1  Windows UI Automation  (window title, process, control type,
                accessible name, role, bounds, enabled/focused/selected, value, hierarchy)
       Layer 2  Local OCR              (text + bounding box + confidence)
       Layer 3  Image processing       (OpenCV: regions, diff/change detection, color/shape)
  -> VisualState (typed)
  -> Layer 4  Local VLM only when deterministic perception is insufficient
  -> VisualState enrichment (VLM proposes elements/targets, always tagged source=vlm)
  -> Atlas Task/Reasoning system -> Task Planner -> Executor -> Observation -> Verifier / Recovery
```

### Layered perception strategy
Screenshots are **not** sent to a model for every operation. The cheapest, most reliable source is tried first:

1. **UI Automation (Layer 1)** — for normal Windows applications, the accessibility tree already publishes names, roles, bounds, enabled/focused/selected state, values, and hierarchy. This is read first and becomes typed elements with `source: uia`. Missing optional dependency: none (ctypes + COM).
2. **OCR (Layer 2)** — reads visible text UIA cannot expose (canvas apps, browsers, images). Every result carries text, bounds, and a per-run confidence. Region-based OCR is preferred whenever a window/region is known. Backend: the local `tesseract` engine via `pytesseract` or the CLI (both optional).
3. **Image processing (Layer 3)** — OpenCV/numpy for screenshots, regions, change/difference detection, and coarse color/shape localization. A pure-Python PNG encoder keeps capture working even without OpenCV; image processing simply no-ops if it is absent.
4. **Local VLM (Layer 4)** — a local multimodal model (e.g. an Ollama vision model) is consulted **only** when deterministic perception cannot adequately answer a structured question such as "what application is visible?", "where is the search box?", "where is the button labeled Continue?", or "what changed between these screens?". The VLM **proposes observations/targets**; it never executes actions, and Atlas validates and executes them. Results are always tagged `source: vlm` and below-threshold proposals are dropped.

### VisualState
`computer/vision/models.py` defines a typed representation (`VisualState`) with `screen` (dimensions, monitors, scale), `active_window`, `windows`, `elements` (`id`, `type`, `name`, `bounds`, `confidence`, `source`, enabled/visible/focused/selected/value, `observation_id`, `application`, relationships), `text` (OCR/UIA runs with bounds and confidence), `sources` (which layers contributed), `vlm_interpretation`, `detected_application`, `screenshot_metadata`, `region`, and `timestamp`/`observation_id` — all JSON-serializable via `to_dict()`.

### Tool contracts
| Capability | Risk | Purpose |
| --- | --- | --- |
| `computer.observe` | read-only | Observe the active/named application; with `full_screen`/`region`/`include_ocr`/`allow_vlm` it returns the full structured visual state, otherwise the window/control observation |
| `computer.vision_observe` | read-only | Capture + analyze the screen into a `VisualState` (UIA + OCR + image + optional VLM) |
| `computer.find` | read-only | Find a visual/UI element by label, semantic description, or type; returns candidates with bounds and confidence |
| `computer.click` / `computer.double_click` / `computer.right_click` / `computer.move` / `computer.drag` / `computer.type` / `computer.keypress` / `computer.scroll` / `computer.focus` | permission-gated | Interact with observed elements |

The interaction tools accept a target as an `element_id` from the latest observation, a visible `description`, and (only deliberately) explicit `x`/`y`. They are registered through the same `ToolRegistry`/`CapabilityRegistry` and are permission-gated like other consequential actions (`computer.type` is high risk; clicks/keypress are medium; move/scroll/focus are low).

### Coordinate safety
A model is never allowed to issue raw screen coordinates that Atlas executes blindly. Every visual action is validated (`computer/vision/actions.py`) against the observation that produced the target:

1. The referenced observation must still be valid (not stale — `VISION_OBSERVATION_MAX_AGE_SECONDS`).
2. The target element must still be present, visible, and enabled.
3. Explicit coordinates must fall inside the target element's bounds.
4. The active window/application must match the observation's, when the action declares one.
5. Significant UI change means re-observe first; a rejected target is never acted on.

### Closed-loop control
Visual actions are added to the executor's `UI_STATEFUL_CAPABILITIES`, so the existing `OBSERVE → ACT → OBSERVE → VERIFY` loop re-observes after each one. The verifier reports a visual action as `unverified` unless the tool itself confirmed an effect (a scheduled input event is not proof), and the executor's completion check and subsequent observation remain the authority on whether the intended effect occurred. Recovery for a visual failure is deterministic and bounded: re-observe with a longer wait (target missing), then allow the local VLM to help locate a stubborn element, then widen the acceptable observation age once — always for the *same* capability, never a new one.

### VLM provider architecture

The vision model is not hard-coded. `computer/vision/providers.py` defines an injectable `VisionProvider` (`OllamaVisionProvider` → local Ollama multimodal model; `NullVisionProvider` when unavailable), built from configuration and degradable. The provider answers structured JSON questions and returns untrusted data; it never executes anything. Reuse of the existing LLM provider injection pattern is deliberate, but the vision boundary is separate because it takes an image and returns structured data rather than `ask(system_prompt, user_prompt) -> str`.

### Graceful degradation
Each layer is optional and the system never hard-depends on a model:

| Present | Still works |
| --- | --- |
| UIA + OCR + image + VLM | Full layered perception with model enrichment |
| UIA + OCR + image (no VLM) | Deterministic perception; VLM calls are skipped honestly |
| UIA + image (no OCR) | Window/control/text structure from UIA |
| OCR + image (no UIA) | Text and region perception |
| No vision extras at all | Existing Atlas functionality is unchanged; `computer.observe` returns the window-only observation |

### Screenshot efficiency
Capture is region/active-window scoped by default (full screen only when asked), downscaled to `VISION_MAX_IMAGE_SIZE`, cached as the last observation, and change-detected so an expensive VLM call is spent only on a real difference. VLM calls are budgeted per request (`VISION_MAX_VLM_CALLS`).

## Sources and evidence

### Source selection

| Request | Current behavior |
| --- | --- |
| Ordinary/general question | Uses the selected local knowledge/model sources; local model answers do not require a successful PDF retrieval. A knowledge miss is not terminal: the model's own knowledge is a source. |
| Current or explicitly online question | Consults registered read-only web search/fetch and answers from the retrieved evidence with sources; unavailable live evidence produces a limitation or clearly qualified, potentially outdated model fallback |
| File lookup/read/content question | Uses permitted local files and, where selected, indexed knowledge; a failed private-file read is not replaced with an invented model answer |
| Personal-document lookup | "find my resume" / "search my documents for 'climate change'" are local file lookups (filename search or content search), not web searches, unless a web platform is named explicitly |
| Local system question | Uses system observations; unavailable machine facts are not inferred from general model knowledge |
| Earlier conversation | Uses available conversation history, not an unimplemented long-term fact database |
| Atlas capabilities/model/tools | Uses actual registered capability metadata and runtime configuration, rather than model claims about installed tools |
| Action request | Preserves `Task.actions` and delegates execution to the permission-gated executor |
| Hybrid request | Combines a source and an action: "search the web for X and write it into Notepad" runs task-aware retrieval (goal + content-type inference → source ranking → extraction → validation → bounded retry) and writes the validated content after confirmation, not the search links and not a page about the subject. Composition is limited to the existing action capabilities. |
| Search results saved to a file | "search for X and save the results to a file" plans `web.search` → `filesystem.write` and persists the rendered result list (links and snippets), deriving a safe default filename from the topic when none is given. A transformation saves its *output*: "search X, summarize, save to a file" runs `web.research` → `content.generate` → `content.format` → `filesystem.write`. |
| Transformation vs. artifact | A transformation noun ("summary", "overview") names how the result is reshaped, not the source to retrieve. "summarize the car videos" retrieves *information about* videos (`must_be_artifact=False`, source content type) and then generates the summary; it does not hunt for a "summary document". A direct artifact request ("get the bee movie script") still sets `must_be_artifact=True`. |
| Ambiguous request | "open it", "find that file", "make it better" with no resolvable referent return a clarification question instead of guessing a source or action |

Read-only research actions (`web.search`/`web.fetch` and non-mutating `filesystem.list/search/read/metadata/search_content`) are served by the reasoning engine, which synthesizes a cited answer. Tasks that mutate state or control applications—including hybrids—are delegated to the validator/planner/executor path. An ordered source plan is not an autonomous research/action graph; selecting web and computer sources does not implement “research and install any program”, and only existing validated Task actions can run.

### Local knowledge

Startup ingestion reads PDFs from `KNOWLEDGE_FOLDER`, splits text into overlapping character chunks, embeds them with sentence-transformers, and stores them in ChromaDB. File hashes track incremental indexing. This PDF loader uses **pypdf**, not PyMuPDF, and does not perform OCR.

`knowledge_search.py` expands queries, tries semantic retrieval, retries expanded queries when needed, adds keyword/metadata signals, and evaluates merged evidence. `MIN_SIMILARITY` participates in this decision; it is not a global prohibition on all model calls. Rejected retrieval is not promoted to evidence merely because it returned candidate chunks. With the general-question fallback disabled, legacy knowledge-only requests can return “I don't know based on my knowledge base.”

### Files and PDFs

Filesystem tools resolve paths against `COMPUTER_ROOT`, reject outside-root targets, and recheck traversal candidates. The default root is this repository: mentioning Downloads, Documents, or Desktop does **not** grant access to those folders. Change the configured root deliberately if needed.

Read/search paths exclude common runtime, dependency, credential, and private-storage locations. Walks, file counts, input bytes, returned text, and results are bounded; exclusions are not a complete secret detector. `filesystem.search` finds names/paths, while `filesystem.search_content` searches extracted text and returns bounded excerpts. These are distinct contracts.

PDF file tools reuse pypdf with input-size, page-count, output-size, and between-page deadline checks. Image-only PDFs need OCR that is not present. A single parser/page extraction can still block; the PDF limits are not hard preemption. The bounded file-tool extraction path is distinct from startup knowledge ingestion and its indexing workload.

### Web and answer quality

`web.search` uses public search providers with per-provider failure isolation and relevance filtering. YouTube requests can return watch links and native thumbnails. `web.fetch` reads bounded public HTML/plain text, treats it as untrusted evidence, and does not execute page scripts or save downloads.

### Task-aware retrieval (`web.research`)

`web.research` is Atlas's **task-aware** retrieval path, used when content must be *retrieved and isolated* rather than merely listed. Chunk relevance alone is not enough: a Wikipedia page *about* a film can out-score the film's actual script. So retrieval runs the pipeline **REASON → PLAN → SEARCH → RANK SOURCES → EXTRACT → VALIDATE → ACT**, with a bounded **reformulate + retry** on validation failure.

1. **`web_task.infer_retrieval_task`** turns the request into a `RetrievalTask`: a `goal` (`retrieve_document`, `find_information`, `find_page`, `find_review`, `find_media`, `find_reference`, …), the requested `content_type` (`movie_script`, `transcript`, `lyrics`, `article`, `review`, `video`, `reference`, `code`, …), whether the artifact *itself* is wanted versus information about it (`must_be_artifact`), and the destination. This is the interpreter's Task IR extended, not a parallel model.
2. **Source ranking before chunk ranking.** `web_task.score_source` scores each candidate page on title/URL/snippet/body term match, detected-vs-requested content-type match, presence of the requested artifact (dialogue/script structure), source quality, and completeness, and penalises pages that merely describe the subject when the artifact was requested. Chunks are ranked only *within* the winning source.
3. **Content-type detection** (`web_task.detect_source_type`) classifies a page as script/transcript/lyrics/article/review/reference/video/forum/product/code/document_host/… from host, title, structure, dialogue density, and card-chrome density. A document-store/listing landing page (Scribd/studylib/etc.) that *advertises* an uploaded document is typed `document_host` and rejected for artifact requests even when it ranks first and its title names the artifact, so it is never copied instead of the real content.
4. **Task-aware validation** (`web_task.validate_content`) decides whether the extracted content satisfies the task and explains a rejection ("expected a movie_script but the source is reference"). A failed validation never reaches an action: `web.research` retries with a query **reformulated from the missing content type** (`reformulate_query`), bounded by `max_attempts`, and fails honestly when nothing qualifies.
5. **LLM validation is narrow.** The local model may judge a *borderline* decision (a well-ranked source the deterministic rules rejected) via `web_task.llm_validate_content`, returning structured `{match, content_type_match, contains_target, completeness, action, reason}`. Routing, scoring, extraction, thresholds, retry limits, and execution all stay deterministic.

This is what a hybrid such as "search the web for the bee movie script and copy it in Notepad" consumes: the destination receives the script content, not the search links and not a page about the film. Retrieval diagnostics (goal, candidates, source scores, detected types, validation, retry query, final selection) are logged for every attempt.

Fetched pages are reduced to their readable main content by `web_content.PageContentParser`: when a page marks a main region (`<main>`, `<article>`, `role=main`) only that region is kept, and otherwise navigation, headers, footers, sidebars, cookie/consent banners, adverts, related/comment/share widgets, scripts/styles, and all media (images, video, audio, iframes) including their alt text are dropped. This deterministic boilerplate removal is what stops a hybrid "get X and write it in Notepad" from pasting an entire page of chrome. It is a heuristic, not a full readability engine: unusual layouts can still keep some noise or drop content. Provider changes, bot challenges, incomplete snippets, and ranking limitations can yield missing or poor results.

### Content formatting pipeline (`content.format`)

Before retrieved or generated content reaches an application (Notepad, Word, code editor, Markdown file, terminal), it passes through the **content formatting pipeline** in `content_formatting.py` and the `content.format` tool. This stage ensures human-readable structure without summarizing or rewriting source content.

Pipeline stages:

1. **Normalization** (`ContentNormalizer`): Removes navigation/boilerplate artifacts, HTML entities, duplicate lines, and normalizes whitespace/line endings. Code indentation is auto-detected and preserved.
2. **Content type detection** (`ContentTypeDetector`): Deterministic signals classify content as `script`, `transcript`, `lyrics`, `article`, `code`, `list`, `general_prose`, etc., from structural cues (speaker lines, timestamps, indentation, list markers, dialogue density).
3. **Paragraph reconstruction** (`ParagraphReconstructor`): For prose, broken lines are rejoined into semantic paragraphs; for scripts/transcripts/lyrics/code, exact line structure is preserved.
4. **Destination-aware rendering** (`DestinationRenderer`): Formats for Notepad (plain text with blank-line paragraph separation), Markdown (headings, fenced code), Word/WordPad (richer headings), code editors (exact content), terminal (compact).
5. **Verification** (`FormattingVerifier`): Checks for empty output, collapsed walls of text, excessive blank lines, content truncation, code indentation damage, and duplicate content.
6. **Automatic repair**: On verification failure, bounded repair attempts (default 2) re-normalize or reconstruct paragraphs, then re-verify.

The pipeline is invoked automatically by the semantic task interpreter for:
- Hybrid web→application tasks (e.g., "search for the Bee Movie script and copy it to Notepad")
- Content generation tasks with a destination (e.g., "write a poem about cars in Notepad")

This ensures the Bee Movie script in Notepad has readable dialogue blocks, not a single collapsed paragraph; lists stay numbered; code keeps indentation; transcripts preserve speaker/timestamp structure.

Answers distinguish direct model knowledge from retrieved evidence, include source metadata/citations where available, and report missing evidence. Evidence sufficiency and output verification are heuristics, not independent proof that a factual answer is correct or current.

## Tools and execution

`ToolRegistry` is the authority for installed tools. `CapabilityRegistry` derives descriptors from that real registry, with contract metadata for required parameters, risk, outputs, and verifiability. Its plannable catalog is a subset of registered capabilities; descriptors alone do not install tools or grant permission.

| Capability family | Current scope |
| --- | --- |
| `content.generate` | Local-model content generation for composed tasks |
| `content.format` | Normalization, content-type detection, paragraph reconstruction, destination-aware rendering, verification, and bounded repair for human-readable output |
| `filesystem.list/read/metadata/search/search_content` | Bounded local inspection and text/PDF lookup |
| `filesystem.write/create_folder/move/copy` | Root-bounded mutations, permission-gated in default confirm mode |
| `applications.list`, `applications.launch`, `applications.launch_named`, `applications.write_text` | Installed-app inspection, non-shell launch, and targeted Windows text entry |
| `computer.windows`, `computer.observe`, `computer.vision_observe`, `computer.find` | Read-only UI perception: window inventory, window/control observation, layered visual-state observation, and element search |
| `computer.click`, `computer.double_click`, `computer.right_click`, `computer.move`, `computer.drag`, `computer.type`, `computer.keypress`, `computer.scroll`, `computer.focus` | Permission-gated visual interaction, validated against the current observation |
| `system.info`, `processes.list`, process inspection tools | Registered read-only machine/process information |
| `powershell.execute` | Documented, validated read-only inspection, not arbitrary shell access |
| `web.search`, `web.fetch`, `web.research` | Read-only public search/page evidence and task-aware multi-page retrieval with source ranking, content-type detection, validation, and query reformulation |

`tools/knowledge.py`, `tools/discovery.py`, and `tools/powershell_commands.json` provide searchable command knowledge and non-executing discovery. PowerShell planning uses documented inspection commands and structured output. Knowledge records are data, not authorization to execute a command. The lexical PowerShell validator is not an AST sandbox.

### Approval, verification and recovery

In the default `confirm` mode, consequential registered actions pause for approval through `/approve` or the task-specific API endpoint. Denial stops the paused action. Resumption keeps completed steps instead of regenerating their outputs. `safe` permits read-only tools; `autonomous` changes low/medium-risk confirmation behavior and should not be mistaken for a safety guarantee. Critical operations are denied by default, subject to explicit policy overrides.

Application text entry (`applications.write_text`) resolves a target and attempts window/control-based delivery, with clipboard/focus fallbacks. Mouse/keyboard interaction is available separately through the observation-validated `computer.*` interaction tools. Both can fail with incompatible applications or ambiguous windows; visual interaction is validated against the current observation but is not a general guarantee that every window can be observed or driven.

The verifier records `verified`, `unverified`, or `failed` according to capability contracts. Many checks inspect returned output shape or tool-reported values (for example, a PID or character count), **not an independent observation of the intended effect**. Bounded recovery can adjust arguments for the same capability, with per-step and plan-wide limits; it is not arbitrary replanning.

New API task submissions use a single FIFO worker, but approval resolution has known concurrency gaps. Do not rely on a global guarantee that all actions can never overlap. Approval grants are bound to the exact planned step arguments (a deterministic signature over the capability and its planned parameters), so a bounded recovery that rewrites a consequential step's arguments is not covered by the earlier grant and pauses for fresh approval. Atomic serialized approval execution remains future hardening work.

## Memory and persistence

`memory/` persists conversation sessions under `memory/sessions/` and builds prompt history from a recent-message window. Brain now persists user/assistant turns for answers returned early by the reasoning engine; persistence is not limited to the legacy retrieval executor path. This does not imply that every failure/approval path has a complete durable event history.

CLI session commands include `/new [title]`, `/list`, `/open <id>`, `/delete <id>`, `/rename <id> <title>`, `/history`, `/export <id> [path]`, `/import <path>`, `/clear`, and `/help`. `/approve` and `/deny` resolve paused actions.

API task snapshots are stored in `database/tasks.json` using atomic replacement. They are history, **not live execution checkpoints**. Restarted pending/running/approval-paused tasks are marked interrupted rather than resumed with missing context. Automatic conversation summarization remains future work.

### Experience memory (human feedback loop)

Atlas accumulates **verified experience** from explicit user feedback and uses it as supporting context for later planning. This is behavioural improvement, not model training: the local model stays the reasoning engine, no weights are modified, and no source code is ever rewritten by Atlas.

```text
USER REQUEST -> INTENT -> PLAN -> EXECUTION -> VERIFICATION -> RESULT
   -> USER FEEDBACK (Success / Failed) -> EXPERIENCE RECORD
   -> EXPERIENCE MEMORY -> FUTURE RETRIEVAL -> BETTER PLANNING
```

`experience/` is a **separate memory** from factual RAG. Atlas keeps four categories apart and does not merge them: KNOWLEDGE (what is true), EXPERIENCE (what worked or failed on a task), USER MEMORY (preferences), and REASONING KNOWLEDGE (principles). Experience embeddings, when available, live in their own Chroma collection (`atlas_experience`) tagged `memory_type: "experience"`, so they can never contaminate a factual answer; when the embedder is unavailable, retrieval falls back to deterministic lexical similarity and the store remains fully usable.

| Path | Responsibility |
| --- | --- |
| `experience/models.py` | Record schema, completion criteria, failure vocabulary, credential scrubbing |
| `experience/store.py` | Append-only JSONL durability (`database/experiences.jsonl`, `database/feedback.jsonl`) |
| `experience/memory.py` | Bounded retrieval + ranking over similarity, trust, and metadata |
| `experience/builder.py` | Execution evidence -> compact experience record |
| `experience/service.py` | The façade Brain and the API use: record, evaluate, retrieve, analyse |

**Feedback controls.** After a task that actually *did* something — tool execution, research, an application action, a multi-step task, a generated deliverable — the control room shows `✓ Success` / `✗ Failed` attached to that specific result. A pure conversational answer shows no controls. Failure is optional to justify: the user can pick a category (wrong interpretation, wrong action, incomplete result, did not follow instruction, wrong information, failed to deliver, verification failed, other) and add a free-text correction, or click without a reason. Feedback is changeable and cannot be duplicated. When the user does not pick a category, Atlas derives one deterministically: a retrieval that returned nothing usable is classified `retrieval_failed` and attributed to `wrong_interpretation`, not to the research tool, because the query (not the network) is the usual cause — so a misinterpreted request is not mislabeled as a tool failure.

**Verification stays authoritative.** Completion criteria (`intent_match`, `required_information_present`, `transformation_completed`, `requested_application_used`, `requested_destination_reached`, `execution_completed`, `delivery_verified`, `final_result_valid`) are derived from tool results and the existing verifier, not from a model's claim. A model saying "Done." is never evidence. An effect the tool could not confirm stays *unknown* rather than being upgraded to a verified success, and a destination written from `applications.write_text` is only `delivery_verified` when the text was read back. When the user's answer disagrees with Atlas's own verification, that disagreement is surfaced rather than averaged away.

**Experience is not authority.** Retrieved experience is rendered into the planner prompt as *supporting context* that explicitly states the current request always wins. A previous task that answered in chat does not justify skipping a destination the user names today. Retrieval is bounded (a small number of records, metadata-filtered, no whole-store scan), runs only for sufficiently complex requests, and never calls a model on the hot path.

**Safe self-improvement.** Repeated success can promote an experience to `reliable`, and periodic analysis (gated on the number of newly evaluated experiences, never per message) proposes candidate improvements such as "destination requirements are frequently lost between interpretation and execution". Proposals are **never applied automatically** — Atlas does not modify its own source code or reasoning policy, and the report is returned as data (`applied: false`).

Feedback and experiences survive an application restart. Only a compact task representation is stored — never a whole conversation — and obvious credentials are scrubbed before persisting.

Runtime/user data includes `database/`, `memory/sessions/`, knowledge PDFs, logs, and exported sessions. Treat these as private local data; do not commit them or assume all are ignored automatically. `.venv/`, frontend dependencies/build outputs, and Python caches are generated artifacts.

## Installation

The documented desktop workflow targets Windows and PowerShell. Windows application/PowerShell tools require Windows. Use Python 3.10+ with dependency-compatible packages, Node.js/npm compatible with the installed Vite version, and a running Ollama service. Dependencies are not comprehensively pinned, so verify the environment rather than assuming every Python/Node release is supported.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:7b
npm --prefix frontend install
```

`requirements.txt` includes `chromadb`, `sentence-transformers`, `pypdf`, `ollama`, `fastapi`, `uvicorn`, and `httpx`. Ollama must be installed separately and serving the configured model. Initial embedding-model loading may require a download. Knowledge PDFs are optional for general model answers; place documents in `knowledge/` for local retrieval.

### Local vision ("Eyes") setup
Vision is **optional and layered**; the base install already captures the screen and reads the Win32 window/control state. To enable each extra layer:

- **UI Automation** works out of the box on Windows (ctypes + COM, no extra package).
- **Local OCR** needs the `tesseract` engine installed and on `PATH`, plus the optional Python extras: `pip install pytesseract pillow` (or just the `tesseract` binary — the CLI is used as a fallback).
- **Image processing** uses `opencv-python` and `numpy` when present: `pip install opencv-python numpy`. Without them, capture still works (pure-Python PNG) and image analysis no-ops.
- **Local VLM** needs a local multimodal model already present in Ollama, e.g. `ollama pull llava:7b`, and `VISION_PROVIDER=ollama` with `VISION_MODEL=llava:7b`. Models are **not** auto-downloaded; Atlas never selects a multi-gigabyte model on its own. There is no cloud vision provider.

Nothing above requires an API key. The vision subsystem is local-first: capture, window detection, UIA, OCR, image processing, element detection, mouse/keyboard, and observation/verification all run on the machine. If a layer is missing, Atlas degrades honestly (see [Local computer vision](#local-computer-vision-eyes)) rather than failing.

## Running

### CLI

```powershell
.\.venv\Scripts\python.exe atlas.py
```

Startup initializes the local runtime and indexes changed knowledge PDFs. Ask ordinary questions, inspect supported sources, or submit a supported action. For example, “What is Python?” need not fail merely because the document index has no Python evidence; “What is the latest Python version?” needs current web evidence to be presented as current.

### Web console

In a terminal at the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In another terminal at the repository root:

```powershell
npm --prefix frontend run dev
```

Open `http://localhost:5173`. After dependencies are installed, `launch_atlas.bat` provides the Windows shortcut for starting the backend, frontend, and browser. The launcher checks the virtual environment, Node/npm, and frontend dependencies. Close the service terminals to stop them.

Keep the backend bound to loopback. The documented deployment has no authenticated multi-user boundary and is not intended for public exposure.

## Configuration

Most runtime settings are Python constants in `config.py`; there is no backend `.env` loader. Restart processes after changing them. Some tool-specific hard limits and the plan recovery cap remain module constants rather than centrally configurable settings.

| Setting | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Current local generation/interpretation model |
| `ENABLE_REASONING_ENGINE` | `True` | Enable source-aware reasoning; disabling retains the legacy informational path |
| `ENABLE_LLM_INTERPRETATION` | `True` | Enable model-assisted interpretation/recovery; not a switch disabling all answer generation |
| `INTERPRETER_CONFIDENCE_THRESHOLD` | `0.75` | Confidence at which the deterministic interpretation can skip the model |
| `CLARIFICATION_CONFIDENCE_THRESHOLD` | `0.45` | Low-confidence clarification policy |
| `ENABLE_GENERAL_QUESTION_FALLBACK` | `True` | Allow model fallback instead of requiring accepted local knowledge for ordinary questions |
| `MAX_REASONING_ITERATIONS` | `3` | Maximum selected-source iterations, not unrestricted agent loops |
| `MAX_REASONING_SOURCES` | `3` | Maximum ordered sources consulted |
| `MAX_REASONING_TOOL_CALLS` | `8` | Budget charged per actual reasoning tool call, including each search/fetch/read |
| `REASONING_TIMEOUT_SECONDS` | `60.0` | Reasoning deadline checks and Ollama transport timeout configuration; not hard whole-request preemption |
| `MAX_RECOVERY_ATTEMPTS` | `2` | Bounded recovery attempts; executor also has `MAX_PLAN_RECOVERIES = 3` |
| `WEB_RESEARCH_MAX_RESULTS` / `WEB_RESEARCH_MAX_PAGES` | `6` / `2` | Search results requested / result pages read by reasoning |
| `FILESYSTEM_CONTENT_MAX_FILES` / `FILESYSTEM_CONTENT_MAX_BYTES` | `200` / `200000` | Content-search file and byte bounds |
| `EXECUTION_MODE` | `confirm` | `safe`, `confirm`, or `autonomous` permission policy |
| `COMPUTER_ROOT` | Repository root | Allowed filesystem root |
| `KNOWLEDGE_FOLDER` / `KNOWLEDGE_GLOB` | `knowledge/` / `*.pdf` | Startup knowledge inputs |
| `DATABASE_FOLDER` / `COLLECTION_NAME` | `database/` / `atlas_knowledge` | Chroma persistence and collection |
| `INDEX_STATE_FILE` / `TASK_STORE_FILE` | `database/index_state.json` / `database/tasks.json` | Index hashes and API task snapshots |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | Character-based knowledge chunking |
| `TOP_K` / `MIN_SIMILARITY` | `5` / `0.75` | Retrieval ranking/acceptance inputs |
| `MEMORY_FOLDER` / `MAX_RETAINED_MESSAGES` | `memory/` / `10` | Session storage and prompt history window |
| `MAX_SESSIONS` / `AUTO_SAVE` | `50` / `True` | Session-count warning threshold (creation still proceeds) and autosave |
| `AUTO_SUMMARIZE_THRESHOLD` | `80` | Reserved/placeholder control, not an enforced automatic-summarization trigger |
| `LOG_LEVEL` / `LOG_TO_FILE` / `LOG_FILE` | `INFO` / `False` / `database/atlas.log` | Logging |
| `LOG_RETRIEVAL` / `DEBUG_PIPELINE` | `True` / `False` | Retrieval diagnostics and structured pipeline tracing |
| `ENABLE_EXPERIENCE_MEMORY` | `True` | Master switch for the feedback/experience loop; when false no experience is recorded or retrieved |
| `EXPERIENCE_STORE_FILE` / `FEEDBACK_STORE_FILE` | `database/experiences.jsonl` / `database/feedback.jsonl` | Durable experience records and feedback answers |
| `EXPERIENCE_COLLECTION_NAME` | `atlas_experience` | Separate Chroma collection for experience embeddings (never mixed with knowledge) |
| `EXPERIENCE_RETRIEVAL_LIMIT` / `EXPERIENCE_MIN_RELEVANCE` | `4` / `0.35` | Bounded retrieval count and relevance floor for planning context |
| `EXPERIENCE_ANALYSIS_THRESHOLD` / `EXPERIENCE_MAX_RECORDS` | `8` / `5000` | Evaluations required before periodic analysis; retained record cap |
| `REDACT_KEYS` | `password`, `token`, `api_key`, `secret` | Diagnostic key redaction, not comprehensive content redaction |
| `ENABLE_COMPUTER_OBSERVATION` | `True` | Read-only Windows UI observation (window inventory, application matching, control text reading) before and after computer actions, so Atlas can verify effects it caused instead of trusting tool-reported success alone |
| `VISION_ENABLED` | `True` | Master switch for the screenshot/OCR/image/VLM perception path; when false, the vision tools report that they are disabled and existing functionality is unaffected |
| `VISION_PROVIDER` | `ollama` | Local vision provider: `ollama` (local multimodal model) or `none` (deterministic perception only). No cloud provider exists |
| `VISION_MODEL` | `llava:7b` | Local multimodal model used for VLM enrichment; must already exist in the local Ollama instance (never auto-downloaded) |
| `VISION_MAX_IMAGE_SIZE` | `1280` | Longest-edge pixel cap for downscaling captures before OCR/VLM |
| `VISION_TIMEOUT` | `60.0` | Transport timeout (seconds) for one local VLM call |
| `VISION_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for a VLM-proposed target to be a candidate |
| `VISION_OBSERVATION_MAX_AGE_SECONDS` | `30.0` | Observation age (seconds) after which a visual action referencing it is rejected as stale |
| `VISION_MAX_VLM_CALLS` | `2` | Maximum VLM enrichment calls per request |
| `VISION_OCR_LANGUAGES` | `eng` | OCR language(s) for the optional tesseract backend |
| `VISION_PREFER_REGION_OCR` | `True` | Run OCR on a known region instead of the whole screen |

The frontend reads `VITE_ATLAS_API_URL` through Vite, defaulting to `http://127.0.0.1:8000/api`. This is a frontend build/dev setting, not a Python `.env` setting.

## API and frontend

The React frontend does not plan or execute tools. `frontend/src/App.tsx` renders views, `types.ts` describes API-facing data, `services/api.ts` owns HTTP calls, and `styles.css` defines the interface. Tasks are polled; there is no SSE/WebSocket execution stream.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health`, `/api/system` | Runtime health and actual system data |
| GET | `/api/tools`, `/api/tool-knowledge` | Registered tools and command knowledge |
| GET | `/api/tool-discovery?query=...` | Non-executing discovery |
| GET | `/api/applications` | Installed Windows application inventory |
| GET | `/api/tasks`, `/api/tasks/{id}` | Task history and task snapshots |
| GET | `/api/experience` | Experience-loop counts and candidate (unapplied) findings |
| GET | `/api/queue` | Current worker/queued submission state |
| POST | `/api/tasks` | Submit `{ "request": "..." }` |
| POST | `/api/tasks/{id}/approve` | Resume the matching approval-paused task |
| POST | `/api/tasks/{id}/feedback` | Record explicit Success/Failed feedback (optional category + correction) |
| POST | `/api/tasks/{id}/deny` | Deny the matching approval-paused task; not general running-task cancellation |

The control room shows task status, plan steps, results, approval controls, applications, tools, and system data. Reasoning snapshots expose bounded stage/detail/iteration fields, response mode, source names, citations, and evidence counts, not raw evidence documents or prompts. Citations are sanitized (including removal of URL credentials/query/fragment and reduction of local paths to names). Persisted reasoning snapshots are sanitized again when restored.

This telemetry filtering is not blanket redaction of all task data: requests, responses, and existing tool outputs can still contain sensitive content. Stage snapshots may appear after a stage/answer completes rather than streaming every in-flight model operation. Files, Knowledge, Memory, and editable Settings views still need dedicated management APIs.

## Testing

Run checks from the repository root after installing dependencies:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
npm --prefix frontend run build
```

The frontend build is **`tsc -b && vite build`**, so it includes TypeScript checking and production bundling. There is no configured Python lint/typecheck command, and no frontend lint or dedicated frontend test script in `package.json`. Do not describe a build as a lint run.

The suite uses `unittest` and covers these categories:

- Semantic Task/source fields, malformed model output, instructional-question versus action behavior, clarification and parameter preservation.
- **Semantic topic decomposition** (`tests/test_topic_decomposition.py`): connector/framing-noun/content-noun separation and the structured reading; the full research matrix (“research about/on/regarding/concerning Skyrim”, “find information about Skyrim”, “look up information on Skyrim”, “tell me about Skyrim”, “research the history of Skyrim”, “research about the best Skyrim mods”); the exact failure (“research about sykrim and write it in notepad” must produce query “sykrim”, never “about sykrim”); research+transformation+destination separation; the non-research cases (poem about cars, videos about Skyrim on YouTube, explanation about photosynthesis); preservation of inner “about” and compound subjects; conservative typo correction that never rewrites a real word; and the bounded query-normalization recovery. The intermediate representation is asserted, not only the final action.
- **Intent Engine 2.0** (`tests/test_intent_engine.py`): answer-vs-research, creation-without-web, research+transformation+delivery, follow-up reference resolution (“make it shorter”, “put that in Notepad”, “do the same for Interstellar”), application-vs-information, local-file-vs-web, ambiguity and its resolution, multi-step composition, and the negative cases — no spurious web, no spurious application, no lost context, no invented capability, and malformed model output never reaching the executor.
- Registry-derived capability availability/introspection, task validation, dependency ordering, generated-text handoff, action delegation and end-to-end execution with fake models/tools. This coverage does not establish complete cycle/reference correctness.
- General/current/local-file/system/memory source routing, accepted versus rejected evidence, answer provenance, per-call budgets and cooperative deadline/cancellation behavior.
- General-purpose routing regressions: personal-document lookups staying local, create-text-file actions, hybrid search-then-write composition, transformation-versus-artifact separation, search-results-saved-to-file and transform-then-save composition, generation-without-destination, named-file pronoun resolution, ambiguous requests clarifying instead of guessing, and the read-only-research-versus-mutation delegation split.
- Content formatting destination resolution (plain-text file vs. Markdown vs. generic), answer routing between the deterministic artifact synthesiser and the evidence-grounded prompt, and argument-bound approval grants (a changed planned step argument requires reapproval).
- Bounded filesystem traversal/read/content search, exclusions, root escapes, PDF text handling, and move/copy contracts. Platform or symlink-privilege checks may skip where unavailable.
- Permission decisions, pause/resume, recovery, task snapshots, queue behavior, application resolution and mocked Windows text-entry paths.
- PowerShell validation/result interpretation, public URL and redirect checks, provider fallback, mocked web/YouTube responses, and Ollama timeout configuration.
- Readable-main-content extraction: main-region preference and dropping of navigation/header/footer/cookie/ad/share chrome, scripts/styles, and media alt text.
- Task-aware web retrieval: retrieval-goal and content-type inference, source ranking before chunk ranking (a page describing the subject scores below the subject's own document), content-type detection (including document-store/listing pages rejected as `document_host`), task-aware validation and rejection reasons, query reformulation, and a bounded retry loop — including the artifact-vs-information, reference-page, review, media, YouTube and Scribd-listing cases.
- API contracts and bounded/sanitized reasoning telemetry, including restored records.
- Local vision perception: `VisualState` model geometry, UIA tree parsing and control-type mapping, OCR parsing (`pytesseract` DICT and `tesseract` TSV) with confidence normalization, region filtering, VLM result coercion/proposal extraction, target validation/coordinate safety (element id, description, explicit coordinates, stale observation, wrong active window, disabled element, out-of-bounds point), vision tool contracts, registration, permissions (observation read-only vs. interaction confirmation-gated), visual verification, visual recovery, and the closed-loop re-observe after a visual action. All use injected fake UIA/OCR/VLM/capture/input providers, so no GUI session is required.
- **Experience memory** (`tests/test_experience_memory.py`, `tests/test_experience_api.py`): feedback eligibility (a conversational answer offers no controls; a meaningful task does), success and failure experience persistence, user-selected and Atlas-derived failure categories, corrections and their restart survival, similar-task retrieval and its irrelevance bound, bounded retrieval that never scans the store, successful and failed experiences rendered into planning context, the current instruction's authority over experience, deterministic verification staying authoritative (an unconfirmed delivery is not a verified success and a model claiming "Done." is not evidence), feedback lifecycle (changeable, non-duplicating, promotion only after repeated success), sensitive-data scrubbing and compact records, periodic analysis that is gated and never auto-applied, retry trajectories, and the API feedback endpoint (success, failure category, correction, invalid vocabulary, unknown task, running task) plus the Brain end-to-end loop with real wiring. All use a temporary store and fake tools/models.

Most contract tests use fakes/mocks; a passing suite does not prove live web freshness, model quality, broad Windows app compatibility, OCR accuracy, local VLM quality, or independently verified computer effects. Some platform smoke paths require Windows/PowerShell. Keep full-suite runs serialized when using shared runtime files. No fixed passing-test total is maintained here; use the current command output and report skips/failures for the exact revision tested.

Manual smoke checks should separately confirm local Ollama availability, one general answer, one current question with provenance, a root-permitted file read, capability introspection, session persistence, and approval/denial for a harmless supported action. Do not run live mutating app/file tests without explicit approval. Main integration validation is separate from this documentation consolidation.

## Limitations and security

- **Not a sandbox:** Python/tool processes run with the user's OS privileges. Root checks, permission policy, and PowerShell validation reduce exposure but do not provide OS isolation. See [SECURITY.md](SECURITY.md).
- **Cooperative limits:** the reasoning deadline is checked between calls; the engine accepts a cancellation callback/token, but Brain/API/UI do not provide full end-to-end running-task cancellation. In-flight model, file/PDF, or network calls are not forcibly preempted by that token.
- **Transport is not wall-clock preemption:** Ollama has a transport timeout, not a hard absolute generation deadline. PDF extraction can block within one page. Reasoning source/tool budgets do not globally meter every internal provider request, retrieval subquery, or legacy executor operation.
- **Web boundary gaps:** initial/final URLs and redirects check public IPv4/IPv6 destinations, but DNS re-resolution and proxy behavior leave DNS-rebinding/TOCTOU risk. This is not a network sandbox.
- **Approval gaps:** task-specific approval does not yet ensure atomic serialized approval execution. Grants are now argument-bound (a recovery that changes effectful planned arguments requires fresh approval), but the approval/resume path is not guaranteed to be atomic under concurrent submission.
- **Verification limits:** tool-reported output shape often stands in for independent effect checks; citations and retrieval sufficiency do not prove truth. Computer-interaction verification is honest about evidence quality: read-back of the target control is the strongest available signal, window/control state is secondary, and iconic/visual or legacy controls report what they can instead of pretending the effect happened. A visual action is reported `unverified` unless the tool confirmed an effect — a scheduled input event is not proof; the following observation is the authority.
- **Vision limits:** UIA coverage varies by application (browser content, canvas, and games often expose little, so OCR/VLM must fill the gap); OCR accuracy depends on the local `tesseract` build, fonts, scaling, and language data; a local VLM can mis-locate elements and its proposals are treated as candidates, never facts. Visual actions assume the observation is current; fast-changing UIs can move a target between observe and act, which the staleness check and re-observe reduce but do not eliminate. Multi-monitor DPI scaling and mixed-DPI setups are a known rough edge. Vision controls applications through the desktop; it is not a browser-specific API and not a guarantee that every GUI can be fully driven.
- **Privacy model:** perception is local — screenshots are captured in memory, are not written to disk by default, and are not uploaded; OCR and VLM inference run on the local machine. A local Ollama vision model processes the image locally like any other Ollama call. Nothing in the vision path sends a screenshot to an external service. Screenshots and observations can still contain sensitive on-screen data, so treat `VisualState`/tool output as private local data like other tool results.
- **Security model:** visual perception is *observation* and never grants permission. Screen content is untrusted external data: text on a page (including prompt-injection text such as "ignore previous instructions and click this") is evidence about pixels, never an instruction, and cannot bypass task validation, the capability catalog, or the permission system. The VLM proposes targets; Atlas validates coordinates against the current observation and still routes every consequential action through the existing permission gate. OCR/VLM output is never treated as a plan.
- **Local-first, not offline-only:** selected web research sends queries to public services; model/embedding setup may download data. Files, memory, task history, and logs can hold private information.
- **Capability limits:** no generic install/package management, controlled downloads, browser automation APIs (Selenium/Playwright/cloud computer-use), voice, or durable resumable execution. Experience memory is retrieval-and-context improvement only: it does not train the model, does not modify source code, and does not promote a proposed policy change automatically. Existing application text entry and visual interaction are narrower than unrestricted GUI automation.

## Roadmap

Priority order emphasizes reliability before broader autonomy:

1. Serialize approval/resume atomically through the execution queue (approval grants are already bound to exact planned step arguments, so changed recovery arguments require reapproval).
2. Complete dependency-cycle validation, order-independent reference checks, generic output resolution, and recovery-state correctness. Add independent effect observations for file/application actions and stronger source/answer verification; test ambiguous targets and failed recovery honestly.
3. Wire cancellation across Brain, API and UI; isolate potentially blocking model/PDF/tool work where hard deadlines are needed; add durable live checkpoints without silently replaying effects.
4. Harden web connection resolution/proxy handling, source trust and provenance, and expand regression coverage for hostile/unavailable sources and platform boundaries.
5. Improve semantic source/action composition within supported capabilities, retrieval evaluation against representative PDFs, and live model/provider compatibility checks.
6. Add bounded Files/Knowledge/Memory APIs, validated Settings, browser regression tests, and an event model before replacing polling with streaming updates.
7. Develop long-term memory with provenance, user-controlled retention/deletion, summarization, confidence, and retrieval. Experience memory (provenance-aware, retrievable, confidence-ranked, and deletion-capable through the store) is the first step; session history is still not this feature. Next: experience retention/pruning APIs, Phase 3 pattern analysis surfaced in the UI, and Phase 4 evaluation-driven reasoning-policy promotion with an explicit accept/reject gate.
8. Extend the local vision layer: broader element perception (UIA depth/patterns for browsers), multi-monitor and DPI handling, region-selective OCR tuning, and stronger visual effect verification (e.g. before/after element-diff assertions per action) so a click can be confirmed against a change rather than the next observation alone.
9. Consider controlled downloads/package installation and dedicated browser automation only behind explicit permissions, stronger validation/isolation, and effect verification — never as the foundation of computer control. Later directions include voice, coding integrations, specialist agents, scheduling, and long-running projects.

Roadmap items are future work, not release promises or evidence of implemented capability.
