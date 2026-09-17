"""Diagnose and repair the section-25 reasoning behavior tests.

Dumps engine/router/test regions into _report_reasoning.txt, applies guarded
fixes (explicit-web router signal, retrieval-shape wrapper around the knowledge
gather), re-runs the behavior tests after each attempt, keeps the best config,
and records everything in the report file.
"""
from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENGINE = ROOT / "reasoning" / "reasoning_engine.py"
ROUTER = ROOT / "reasoning" / "query_router.py"
TESTS = ROOT / "tests" / "test_reasoning_behavior.py"
SS = ROOT / "reasoning" / "source_selector.py"
REPORT = ROOT / "_report_reasoning.txt"

out_lines: list[str] = []

def say(line: str = "") -> None:
    print(line)
    out_lines.append(line)

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")

def dump_def(src: str, name: str, limit: int = 4000) -> str:
    m = re.search(rf"^(?:def {name}\\b|class {name}\\b|    def {name}\\b)", src, re.M)
    if not m:
        return f"<<{name} not found>>"
    nxt = re.search(r"^(?:def \\w+|class \\w+|    def \\w+)", src[m.end():], re.M)
    end = m.end() + nxt.start() if nxt else len(src)
    return src[m.start():end][:limit]

def run_pytest(args: list[str], timeout: int = 600):
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        outp = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, [], "TIMEOUT"
    m = re.search(r"(\\d+) passed", outp)
    passed = int(m.group(1)) if m else -1
    failed = re.findall(r"FAILED (\\S+)", outp)
    return passed, failed, outp

def tail(outp: str, n: int = 25) -> str:
    return "\\n".join(outp.strip().splitlines()[-n:])

engine_src, router_src, tests_src = read(ENGINE), read(ROUTER), read(TESTS)
for p in (ENGINE, ROUTER, TESTS):
    bak = p.with_name(p.name + ".orig.bak")
    if not bak.exists():
        bak.write_text(read(p), encoding="utf-8")

say("=" * 72); say("PHASE 1 DIAGNOSIS"); say("=" * 72)
try:
    from knowledge_search import RetrievalResult
    rfields = [f.name for f in dataclasses.fields(RetrievalResult)]
except Exception as exc:
    rfields = []
    say(f"RetrievalResult introspection failed: {exc!r}")
say("RetrievalResult fields: " + (", ".join(rfields) if rfields else "<unknown>"))

say("--- engine structure ---")
for m in re.finditer(r"^(?:class \\w+|def \\w+|    def \\w+)", engine_src, re.M):
    say(m.group(0))
say("--- router structure ---")
for m in re.finditer(r"^(?:class \\w+|def \\w+|    def \\w+)", router_src, re.M):
    say(m.group(0))
say("--- router signal sites ---")
for i, line in enumerate(router_src.splitlines(), 1):
    if re.search(r"explicit_web_request|time_sensitive|web_hint|_TIME_SENSITIVE_TERMS", line):
        say(f"{i}: {line.rstrip()}")
say("--- engine delegation/action sites ---")
for i, line in enumerate(engine_src.splitlines(), 1):
    if re.search(r"delegat|RequestType\\.|requires_computer|requires_action|task\\.actions|handle_request", line):
        say(f"{i}: {line.rstrip()}")
for m in re.finditer(r"(?:    def|def) (\\w*delegat\\w*)", engine_src):
    nm = m.group(1)
    say(f"--- engine {nm} ---")
    say(dump_def(engine_src, nm))
for name in ("handle_request", "_gather_knowledge", "_gather_files", "_gather_web", "_gather_system", "_compose"):
    body = dump_def(engine_src, name)
    if not body.startswith("<<"):
        say(f"--- engine {name} ---")
        say(body)
ss_src = read(SS)
say("--- source_selector requires_action sites ---")
for i, line in enumerate(ss_src.splitlines(), 1):
    if "requires_action" in line or "RequestType" in line:
        say(f"{i}: {line.rstrip()}")

bp, bf, bout = run_pytest(["tests/test_reasoning_behavior.py"])
say(f"BASELINE behavior tests: passed={bp} failed={len(bf)}")
for f in bf:
    say("FAILED " + f)
say(tail(bout))
for f in bf:
    tname = f.rsplit("::", 1)[-1].split("[")[0]
    say(f"--- failing test {tname} ---")
    say(dump_def(tests_src, tname, 2600))

say("=" * 72); say("PHASE 2 ATTEMPTS"); say("=" * 72)
cur_e, cur_r, cur_t = engine_src, router_src, tests_src
best_p = bp if bp >= 0 else 0
best_f = list(bf)

def attempt(label: str, e: str, r: str, t: str) -> None:
    global cur_e, cur_r, cur_t, best_p, best_f
    write(ENGINE, e); write(ROUTER, r); write(TESTS, t)
    p_, f_, o_ = run_pytest(["tests/test_reasoning_behavior.py"])
    say(f"ATTEMPT {label}: passed={p_} failed={len(f_)}")
    for x in f_:
        say("  FAILED " + x)
    if p_ > best_p:
        best_p, best_f = p_, f_
        cur_e, cur_r, cur_t = e, r, t
        say("  -> kept")
    else:
        say("  -> reverted")

r2 = cur_r
if "_EXPLICIT_WEB_RE" not in r2:
    anchor = "_QUESTION_LEADS: tuple[str, ...] = ("
    block = (
        "_EXPLICIT_WEB_RE = re.compile(\\n"
        "    r\"\\\\b(?:search|look|find|check|go)\\\\b[^.?!]{0,40}\\\\b(?:web|internet|online|google|bing|duckduckgo)\\\\b\"\\n"
        "    r\"|\\\\b(?:google|bing|duckduckgo)\\\\b[^.?!]{0,40}\\\\b(?:search|it|for)\\\\b\"\\n"
        "    r\"|\\\\bsearch the (?:web|internet|net)\\\\b\",\\n"
        ")\\n\\n"
    )
    if anchor in r2:
        r2 = r2.replace(anchor, block + anchor, 1)
m_field = re.search(r"( *)time_sensitive: bool[^\\n]*\\n", r2)
if m_field and "explicit_web_request" not in r2:
    r2 = r2.replace(m_field.group(0), m_field.group(0) + m_field.group(1) + "explicit_web_request: bool = False\\n", 1)
m_call = re.search(r"\\n( +)time_sensitive=([^\\n]+),", r2)
if m_call:
    rhs = m_call.group(2)
    vm = re.search(r"_matches_terms\\(\\s*(\\w+)", rhs) or re.search(r"search\\(\\s*(\\w+)", rhs)
    var = vm.group(1) if vm else "lowered"
    r2 = r2.replace(
        m_call.group(0),
        m_call.group(0) + f'\\n{m_call.group(1)}explicit_web_request=bool(_EXPLICIT_WEB_RE.search({var} or "")),',
        1,
    )
else:
    m_attr = re.search(r"\\n( +)(\\w+)\\.time_sensitive\\s*=\\s*([^\\n]+)", r2)
    if m_attr and "explicit_web_request" not in r2:
        r2 = r2.replace(
            m_attr.group(0),
            m_attr.group(0) + f'\\n{m_attr.group(1)}{m_attr.group(2)}.explicit_web_request = bool(_EXPLICIT_WEB_RE.search(lowered or ""))',
            1,
        )
say("router explicit-web patch changed file: " + str(r2 != cur_r))
attempt("C-explicit-web", cur_e, r2, cur_t)

def engine_with_retrieve_wrap(src: str, chunk_style: str) -> str:
    if "_wrap_retrieval" in src:
        return src
    if chunk_style == "dict":
        chunk_line = (
            "                chunks = [{\"text\": context, \"source\": str(sources[0]) if sources else \"local knowledge base\"}] if context else []\\n"
        )
    else:
        chunk_line = "                chunks = [context] if context else []\\n"
    helper = (
        "def _wrap_retrieval(retrieve):\\n"
        "    \"\"\"Normalize knowledge-retrieval results to a stable dict contract.\\n\\n"
        "    Callers may return a RetrievalResult, a complete dict, or a partial\n"
        "    dict; missing pieces are filled deterministically so the gather step\n"
        "    never crashes on a partial result.\"\"\"\\n\\n"
        "    if retrieve is None:\n"
        "        return None\\n\\n"
        "    def _wrapped(*args, **kwargs):\\n"
        "        result = retrieve(*args, **kwargs)\\n"
        "        if result is None:\n"
        "            return None\\n"
        "        if isinstance(result, dict):\\n"
        "            data = dict(result)\\n"
        "            context = str(data.get(\"context\") or \"\")\\n"
        "            sources = list(data.get(\"sources\") or [])\\n"
        "            chunks = data.get(\"retrieved_chunks\")\\n"
        "            if chunks is None:\n"
        "                if not context and sources:\n"
        "                    context = \", \".join(str(s) for s in sources)\\n"
        + chunk_line
        + "                data[\"retrieved_chunks\"] = chunks\\n"
        "            data.setdefault(\"context\", context)\\n"
        "            data.setdefault(\"sources\", sources)\\n"
        "            return data\\n"
        "        return result\\n\\n"
        "    return _wrapped\\n\\n\\n"
    )
    m = re.search(r"( *)self\\._retrieve\\s*=\\s*([^\\n]+)\\n", src)
    if not m:
        return src
    src = src.replace(m.group(0), f"{m.group(1)}self._retrieve = _wrap_retrieval({m.group(2).strip()})\\n", 1)
    cm = re.search(r"^class \\w+", src, re.M)
    if cm:
        return src[:cm.start()] + helper + src[cm.start():]
    return src + "\\n\\n" + helper

attempt("A1-retrieve-wrap-dict-chunks", engine_with_retrieve_wrap(engine_src, "dict"), cur_r, cur_t)
attempt("A2-retrieve-wrap-str-chunks", engine_with_retrieve_wrap(engine_src, "str"), cur_r, cur_t)

write(ENGINE, cur_e); write(ROUTER, cur_r); write(TESTS, cur_t)
cc = subprocess.run([sys.executable, "-m", "py_compile", str(ENGINE), str(ROUTER), str(TESTS)], capture_output=True, text=True)
say("py_compile rc=" + str(cc.returncode) + ((" " + cc.stderr[:400]) if cc.returncode else ""))
fp, ff, fout = run_pytest(["tests/test_reasoning_behavior.py"])
say(f"FINAL behavior tests: passed={fp} failed={len(ff)}")
for x in ff:
    say("FAILED " + x)
say(tail(fout))
REPORT.write_text("\\n".join(out_lines), encoding="utf-8")
print("report written:", REPORT)

rp, rf, rout = run_pytest(["tests/", "--ignore=tests/test_api.py"], timeout=900)
with REPORT.open("a", encoding="utf-8") as fh:
    fh.write(f"\\nREGRESSION SWEEP: passed={rp} failed={len(rf)}\\n")
    for x in rf:
        fh.write("FAILED " + x + "\\n")
print(f"REGRESSION SWEEP: passed={rp} failed={len(rf)}")
