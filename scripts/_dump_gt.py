"""Dump all ground truth needed to fix the section-25 reasoning tests."""
from __future__ import annotations

import io
import subprocess
import sys

OUT = "scripts/_gt_report.txt"
buf = io.StringIO()
w = buf.write


def dump(path, patterns, ctx=12):
    w(f"\n===== {path} (patterns={patterns}) =====\n")
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError as exc:
        w(f"MISSING: {exc}\n")
        return
    w(f"(total {len(lines)} lines)\n")
    hits = [i for i, line in enumerate(lines) if any(p in line for p in patterns)]
    if not hits:
        w("(no pattern hits)\n")
        return
    ranges = []
    for h in hits:
        if ranges and h - ranges[-1][1] <= 2:
            ranges[-1] = (ranges[-1][0], h)
        else:
            ranges.append((h, h))
    for lo, hi in ranges:
        a = max(0, lo - ctx)
        b = min(len(lines) - 1, hi + ctx)
        for i in range(a, b + 1):
            w(f"{i+1}: {lines[i]}\n")
        w("-----\n")


w("===== tests/test_reasoning_behavior.py (FULL) =====\n")
try:
    src = open("tests/test_reasoning_behavior.py", encoding="utf-8").read()
    for i, line in enumerate(src.splitlines()):
        w(f"{i+1}: {line}\n")
except OSError as exc:
    w(f"MISSING: {exc}\n")

dump(
    "reasoning/reasoning_engine.py",
    [
        "def handle_request",
        "def _should_delegate",
        "def _delegate",
        "requires_computer",
        "requires_action",
        "def _gather",
        "retrieved_chunks",
        "def _retrieve",
        "def _fetch",
        "def _knowledge",
        "def _files_evidence",
        "def _web_evidence",
        "def _run_",
        "invoke(",
    ],
    ctx=14,
)

dump(
    "reasoning/query_router.py",
    [
        "class QuerySignals",
        "explicit_web_request",
        "web_hint",
        "time_sensitive",
        "def route(",
        "QuerySignals(",
    ],
    ctx=12,
)

dump("knowledge_search.py", ["class RetrievalResult", "class Retrieval"], ctx=25)

w("\n===== PYTEST (one-line causes) =====\n")
try:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_reasoning_behavior.py",
            "-q",
            "--tb=line",
            "-rf",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=".",
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    w("\n".join(text.splitlines()[-45:]) + "\n")
except Exception as exc:
    w(f"PYTEST RUN FAILED: {exc!r}\n")

open(OUT, "w", encoding="utf-8").write(buf.getvalue())
print("WROTE", OUT, len(buf.getvalue()), "chars")
