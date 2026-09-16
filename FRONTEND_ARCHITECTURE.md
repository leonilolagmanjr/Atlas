# Atlas Frontend Architecture

**Status:** Implemented initial control room
**Date:** 2026-09-16

## Existing backend assessment

Atlas remains a Python CLI-first runtime. Brain classifies requests, Planner creates deterministic plans, Executor runs retrieval or explicit tool steps, and Memory persists conversations. The computer runtime currently provides bounded filesystem inspection, process/system inspection, installed-application inventory, and permission-gated executable launch. Ollama, ChromaDB, and the local knowledge index remain backend-only.

There was no existing GUI or HTTP API. The frontend therefore uses a thin FastAPI adapter and does not duplicate planning, retrieval, permission, or tool execution logic in TypeScript.

## Frontend structure

```text
frontend/
  src/
    App.tsx                 Application shell and section views
    styles.css              Dark operational design system
    types.ts                API-facing TypeScript types
    services/api.ts         Centralized HTTP client
    main.tsx                React entry point
  package.json              Vite scripts and dependencies
  vite.config.ts
```

The shell is desktop-first and responsive. The Command workspace is the primary surface. Tasks, Applications, Tools, and System are backed by live endpoints. Files, Knowledge, Memory, and Settings deliberately show backend-dependent states until their APIs exist.

## API contract

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Local API/model availability |
| GET | `/api/system` | Real OS, CPU, Python, and disk data |
| GET | `/api/tools` | Registered tool metadata and permission levels |
| GET | `/api/applications` | Installed Windows application metadata |
| GET | `/api/tasks` | In-memory task records |
| POST | `/api/tasks` | Submit `{ "request": "..." }`, returns a queued task |
| GET | `/api/tasks/{id}` | Poll task status, plan, tool calls, warnings, and errors |
| POST | `/api/tasks/{id}/approve` | Resume the current approval-paused task |
| POST | `/api/tasks/{id}/deny` | Cancel the current approval-paused task |

Task execution is polled because the current backend has no event stream. The API runs Brain in a bounded worker pool and reports the actual `ExecutionContext` state. Approval is intentionally explicit and local to the current API process.

## Implemented frontend features

- Atlas control-room shell with navigation and responsive mobile drawer.
- Natural-language task submission against the real Brain runtime.
- Polling task status and rendering plan steps, approval state, results, warnings, and failures.
- Application inventory from real Windows uninstall metadata.
- Tool registry with category and permission/risk display.
- System dashboard using real backend values and `Unavailable` for unsupported metrics.
- Honest unavailable states for backend surfaces that do not yet have APIs.

## Backend-dependent follow-up

- Knowledge documents, chunks, indexing status, and retrieval diagnostics need read-only API endpoints.
- Memory sessions and long-term memory need API endpoints.
- Files need safe browse/read/search endpoints with the same root bounds as computer tools.
- Terminal execution needs a separate validated PowerShell tool and streaming output model.
- Settings need a validated configuration read/write contract.
- Task history needs durable storage rather than the API process's in-memory records.
- SSE/WebSocket events can replace polling after an event model exists.
