# Atlas Testing

## Automated checks

Run from the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in pathlib.Path('.').rglob('*.py') if '.venv' not in p.parts and '__pycache__' not in p.parts]; print('syntax ok')"
.\.venv\Scripts\python.exe -c "import atlas, api, brain, cli, executor, intent_classifier, planner, models, tools, computer; print('imports ok')"
Push-Location frontend; npm run build; Pop-Location
```

The Python suite uses the standard library `unittest` runner and covers:

- Semantic task understanding (`tests/test_task_pipeline.py`): the
  `Create a poem in Notepad` vs `Create a poem in Notepad about cars`
  regression, many phrasings resolving to the same task family, generalization
  to unseen combinations, the full specification matrix, Task IR robustness
  against malformed model output, validation, dependency-ordered planning with
  `$variable` references, and verification.
- The live task pipeline (`tests/test_task_pipeline_e2e.py`): Brain →
  interpreter → validator → planner → executor → verifier with a fake model and
  fake tools, including model-supplied tasks, rejected invented capabilities,
  malformed-output fallback, and verification recording.
- Execution context serialization and lifecycle states.
- Tool metadata, validation, registration, routing, and permission decisions.
- Bounded filesystem inspection and traversal rejection.
- Process/system/application inspection contracts.
- Non-shell application launch and confirmation gating.
- Application intent, deterministic planning, executor pause/resume boundaries, and CLI approval callbacks.
- FastAPI health and system smoke endpoints.
- Durable task snapshot restoration and interrupted-task handling.
- Tool knowledge JSON import and capability discovery.
- PowerShell command safety validation, structured execution, and timeout boundaries.
- Natural-language computer intent and knowledge-backed PowerShell planning.
- Real Windows PowerShell smoke execution through `PowerShellTool` with JSON result interpretation.
- Frontend API contracts for tool knowledge and non-executing tool discovery.
- Natural-language arbitrary named application resolution and confirmation-gated launch planning.
- Controlled application text-entry planning and mocked clipboard-paste execution.
- Web search intent, Bing fallback parsing, public URL validation, source attribution, and untrusted-page handling.
- Direct YouTube renderer parsing, MrBeast video relevance filtering, watch-link generation, and thumbnail metadata.

## Semantic task verification
Set `DEBUG_PIPELINE = True` in `config.py` to emit the structured per-request
trace for every request:

```text
Atlas pipeline trace
USER INPUT: Create a poem in Notepad about cars.
TASK: {"task_type": "content_creation", "goal": "create_content", "actions": [...]}
VALIDATION: {"valid": true, "requires_confirmation": true, "risk_level": "medium_risk"}
INTENT: {...}
PLAN:
  1. {"id": "a1", "tool": "content.generate", "parameters": {"topic": "cars", ...}}
  2. {"id": "a2", "tool": "applications.write_text", "parameters": {"text": "$generated_text"}}
EXECUTION:
  1. {...}
  2. {...}
VERIFICATION:
  1. {"capability": "content.generate", ...}
  2. {"capability": "applications.write_text", "verified": true, ...}
RESULT: ...
```

Sensitive keys are redacted via `REDACT_KEYS`.

## Manual runtime checks completed

On 2026-09-16:

- Frontend production build passed with Vite and strict TypeScript.
- API health returned `online`, local mode, model `qwen2.5:7b`.
- API system returned real Windows OS, CPU, Python, and disk data.
- API tools returned the registered computer capabilities and permission metadata.
- API applications returned installed Windows registry entries.
- A real explicit executable task reached `WAITING_FOR_CONFIRMATION` and was cancelled through the API without launching the application.
- The browser console rendered the command workspace, approval state, cancellation state, and live Applications view.
- Ollama 0.31.2 was present and `qwen2.5:7b` was installed.

## Not verified in this audit

- A successful Ollama generation through `Brain`, because it depends on the running Ollama service and a retrieval context accepted by the local Chroma index.
- Retrieval quality against a representative PDF corpus.
- Controlled web search and public page retrieval are implemented; downloads, webpage prompt-injection automation, and browser interaction remain pending.
- Broad live PowerShell coverage: one safe process pipeline was verified; the unit suite continues to mock subprocesses for deterministic failure and timeout cases.
- Notepad text entry is covered through planning and mocked execution; no unsolicited live window is opened by the test suite.
- A real Notepad text-entry smoke test successfully wrote a harmless test sentence through the Win32-focus and clipboard-paste path.
- A real MrBeast YouTube search returned a video watch URL and `i.ytimg.com` thumbnail.
- Resuming a live execution context after an API restart: task snapshots are
	durable, but execution checkpoints are not implemented yet.
- Browser automated regression coverage: the live browser smoke check was manual through the VS Code browser tool.

## Security checks

The implemented computer boundary was checked for:

- Filesystem traversal outside the configured repository root: rejected.
- Application launch shell injection: launch uses an executable path plus argument list with `shell=False`.
- Unapproved application launch: router returns `confirmation_required` and Executor pauses the plan.
- Critical permission defaults: denied.

- Read-only web search/fetch: HTTP(S)-only URLs, rejection of loopback/private/link-local targets (SSRF), response-size and timeout limits, and untrusted labeling of page content.

Web downloads, webpage-instruction automation, browser automation, and secret-handling security checks remain pending because those capabilities are not implemented.
PowerShell mutation, AST-aware validation, and live command cancellation remain pending.

> Note: the frontend production build is verified with `Set-Location frontend; npm run build`. It was not re-run during the documentation alignment pass because `npm` was not available on the PATH in that environment.
