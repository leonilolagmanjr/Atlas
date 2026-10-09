"""External-dependency probing for the evaluation system.

The evaluation must be able to distinguish *"Atlas behaved incorrectly"* from
*"an external dependency is unavailable"*. That distinction is made here, once,
by probing each dependency and reporting one of three honest states:

``available``
    The dependency is present and usable.
``degraded``
    The dependency is present but partially limited (e.g. a local model that
    answers but cannot stream, or a Windows host whose file tools work while an
    application launcher does not). The case still runs; the degradation is
    reported with the result.
``unavailable``
    The dependency is missing. A case that ``requires`` it is ``BLOCKED``: not a
    PASS, not a FAIL, and excluded from the pass/fail statistics.

Probes are cheap, side-effect-free where possible, and never mutate state. A
probe failure is reported as ``unavailable`` with its reason rather than raised,
because a broken probe is information, not a crash.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass, field
from typing import Callable

from evaluation.cases import DependencyStatus


@dataclass
class DependencyProbe:
    """One probed external dependency and the state it was found in."""

    name: str
    status: DependencyStatus
    reason: str = ""
    #: Optional evidence gathered by the probe (e.g. model names, versions).
    detail: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "detail": dict(self.detail),
        }


def _probe_local_model() -> DependencyProbe:
    """Probe the configured local Ollama model.

    Presence of the ``ollama`` package and a reachable server are not the same
    thing, so this probes the server with a cheap ``list`` call. A server that is
    running but does not hold the configured model is *degraded* (Ollama itself
    works; the model does not), never unavailable.
    """

    name = "local_model"
    try:
        import ollama  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - a missing package is a real state
        return DependencyProbe(name, DependencyStatus.UNAVAILABLE, f"ollama package not importable: {type(exc).__name__}")

    try:
        from config import OLLAMA_MODEL
    except Exception:  # noqa: BLE001
        OLLAMA_MODEL = "qwen2.5:7b"

    try:
        client = ollama.Client() if hasattr(ollama, "Client") else ollama
        response = client.list()
        models = _model_names(response)
    except Exception as exc:  # noqa: BLE001 - an unreachable server is not availability
        return DependencyProbe(name, DependencyStatus.UNAVAILABLE, f"ollama server unreachable: {type(exc).__name__}")

    if not models:
        return DependencyProbe(name, DependencyStatus.UNAVAILABLE, "ollama server lists no models", {"models": models})
    if OLLAMA_MODEL not in models:
        return DependencyProbe(
            name,
            DependencyStatus.DEGRADED,
            f"server is up but the configured model {OLLAMA_MODEL!r} is not pulled",
            {"configured": OLLAMA_MODEL, "models": models},
        )
    return DependencyProbe(name, DependencyStatus.AVAILABLE, f"model {OLLAMA_MODEL!r} present", {"models": models})


def _model_names(response: object) -> list[str]:
    """Extract model names from an Ollama ``list`` response of either shape."""

    names: list[str] = []
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models")
    for item in models or []:
        if isinstance(item, dict):
            value = item.get("model") or item.get("name")
        else:
            value = getattr(item, "model", None) or getattr(item, "name", None)
        if value:
            names.append(str(value))
    return names


def _probe_web() -> DependencyProbe:
    """Probe public web access without spending a real query.

    A DNS/HTTP reachability check against a well-known host is enough: the
    evaluation's web cases are driven by a recording fake so the *routing* is
    deterministic, but a live web case needs a real network path.
    """

    name = "web"
    try:
        import socket

        socket.setdefaulttimeout(3.0)
        socket.gethostbyname("duckduckgo.com")
    except Exception as exc:  # noqa: BLE001 - offline is a real state
        return DependencyProbe(name, DependencyStatus.UNAVAILABLE, f"no DNS/network path: {type(exc).__name__}")
    # httpx is the documented transport; its absence is a degraded state.
    try:
        import httpx  # noqa: F401
    except Exception:  # noqa: BLE001
        return DependencyProbe(name, DependencyStatus.DEGRADED, "network resolves but httpx is not importable")
    return DependencyProbe(name, DependencyStatus.AVAILABLE, "network path resolves")


def _probe_windows() -> DependencyProbe:
    """Probe the Windows application-control surface.

    The file/read-only tools work cross-platform; application launch and screen
    interaction require Windows. This reports ``available`` on Windows and
    ``unavailable`` elsewhere, with the reason stated.
    """

    name = "windows"
    system = platform.system()
    if system != "Windows":
        return DependencyProbe(name, DependencyStatus.UNAVAILABLE, f"host platform is {system}, not Windows", {"platform": system})
    return DependencyProbe(name, DependencyStatus.AVAILABLE, "Windows host", {"platform": system})


def _probe_vision_ocr() -> DependencyProbe:
    """Probe the OCR layer (tesseract binary or pytesseract binding)."""

    name = "vision_ocr"
    try:
        import pytesseract  # noqa: F401

        return DependencyProbe(name, DependencyStatus.AVAILABLE, "pytesseract importable")
    except Exception:  # noqa: BLE001 - fall through to the CLI probe
        pass
    binary = shutil.which("tesseract")
    if binary:
        return DependencyProbe(name, DependencyStatus.AVAILABLE, "tesseract CLI on PATH", {"binary": binary})
    return DependencyProbe(name, DependencyStatus.UNAVAILABLE, "neither pytesseract nor the tesseract CLI is present")


def _probe_vision_vlm() -> DependencyProbe:
    """Probe the optional local VLM layer through the existing vision provider."""

    name = "vision_vlm"
    try:
        from computer.vision.providers import NullVisionProvider, build_vision_provider
        from config import VISION_MODEL, VISION_PROVIDER, VISION_TIMEOUT
    except Exception as exc:  # noqa: BLE001
        return DependencyProbe(name, DependencyStatus.UNAVAILABLE, f"vision provider module unavailable: {type(exc).__name__}")
    try:
        provider = build_vision_provider(
            provider=VISION_PROVIDER,
            model=VISION_MODEL,
            timeout=VISION_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001
        return DependencyProbe(name, DependencyStatus.UNAVAILABLE, f"vision provider build failed: {type(exc).__name__}")
    if isinstance(provider, NullVisionProvider) or getattr(provider, "name", "") in {"null", "none"}:
        return DependencyProbe(name, DependencyStatus.UNAVAILABLE, "vision provider degraded to the null provider")
    return DependencyProbe(name, DependencyStatus.AVAILABLE, f"vision provider {getattr(provider, 'name', '?')!r}")


_PROBES: dict[str, Callable[[], DependencyProbe]] = {
    "local_model": _probe_local_model,
    "web": _probe_web,
    "windows": _probe_windows,
    "vision_ocr": _probe_vision_ocr,
    "vision_vlm": _probe_vision_vlm,
}


def known_dependencies() -> tuple[str, ...]:
    return tuple(_PROBES)


def probe_dependencies(names: list[str] | None = None) -> dict[str, DependencyProbe]:
    """Probe every requested dependency once and return the results.

    Unknown dependency names are reported ``unavailable`` rather than raising:
    a case declaring a typo must be visible as a blocked case, not a crash.
    """

    requested = list(names) if names else list(_PROBES)
    results: dict[str, DependencyProbe] = {}
    for dep in requested:
        probe = _PROBES.get(dep)
        if probe is None:
            results[dep] = DependencyProbe(dep, DependencyStatus.UNAVAILABLE, "unknown dependency name")
            continue
        try:
            results[dep] = probe()
        except Exception as exc:  # noqa: BLE001 - a broken probe is information
            results[dep] = DependencyProbe(dep, DependencyStatus.UNAVAILABLE, f"probe error: {type(exc).__name__}: {exc}")
    return results
