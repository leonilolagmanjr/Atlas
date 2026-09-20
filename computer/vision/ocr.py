"""Local OCR (Layer 2) for Atlas.

OCR reads visible text that Windows UI Automation cannot expose (canvas apps,
browsers, images, games). It runs entirely locally: the backend is the
``tesseract`` binary through either the ``pytesseract`` Python binding or a
direct subprocess call. If neither is present, OCR degrades to "not available"
and perception continues with UIA and image processing.

OCR results always carry a bounding box and a confidence, and OCR output is
treated as untrusted external data: it is evidence about pixels, never an
instruction.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod

from computer.vision.models import Bounds, ObservationSource, TextObservation

logger = logging.getLogger(__name__)


class OcrProvider(ABC):
    """A local OCR backend. Implementations never raise for "no text"."""

    name: str = "ocr"

    @abstractmethod
    def available(self) -> bool:
        """Return True when this backend can actually run."""

    @abstractmethod
    def recognize(self, image: bytes) -> list[TextObservation]:
        """Return text observations found in ``image`` (PNG bytes)."""


class NullOcrProvider(OcrProvider):
    """An OCR backend that always reports no text (used when none is installed)."""

    name = "none"

    def available(self) -> bool:
        return False

    def recognize(self, image: bytes) -> list[TextObservation]:
        return []


#: tesseract's image_to_data returns per-word confidence in [0, 100]; below this
#: a word is treated as noise rather than text.
_MIN_WORD_CONFIDENCE = 30.0


def _confidence_to_unit(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0:
        return 0.0
    return max(0.0, min(1.0, number / 100.0))


class TesseractOcrProvider(OcrProvider):
    """OCR through the local tesseract engine (pytesseract or the CLI)."""

    name = "tesseract"

    def __init__(self, *, languages: str = "eng", timeout: float = 20.0) -> None:
        self._languages = languages or "eng"
        self._timeout = float(timeout)

    def available(self) -> bool:
        try:
            import pytesseract  # noqa: F401
            # A binding existing is not enough; the binary must resolve too.
            return bool(shutil.which("tesseract") or _pytesseract_binary())
        except Exception:  # noqa: BLE001 - optional dependency
            return bool(shutil.which("tesseract"))

    def recognize(self, image: bytes) -> list[TextObservation]:
        try:
            import pytesseract
            from PIL import Image  # noqa: F401

            return self._recognize_with_pytesseract(image, pytesseract)
        except Exception:  # noqa: BLE001 - fall back to the CLI
            return self._recognize_with_cli(image)

    def _recognize_with_pytesseract(self, image: bytes, pytesseract) -> list[TextObservation]:
        import io
        from PIL import Image

        pil_image = Image.open(io.BytesIO(image))
        data = pytesseract.image_to_data(
            pil_image,
            lang=self._languages,
            output_type=pytesseract.Output.DICT,
            timeout=self._timeout,
        )
        return parse_tesseract_data(data)

    def _recognize_with_cli(self, image: bytes) -> list[TextObservation]:
        binary = shutil.which("tesseract")
        if not binary:
            return []
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as handle:
                handle.write(image)
                handle.flush()
                # ``tsv`` gives word bounding boxes and confidences: the shape
                # parse_tesseract_tsv consumes and the same shape tests mock.
                completed = subprocess.run(
                    [binary, handle.name, "stdout", "-l", self._languages, "tsv"],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    check=False,
                    shell=False,
                )
        except (OSError, subprocess.SubprocessError):
            return []
        if completed.returncode != 0:
            return []
        return parse_tesseract_tsv(completed.stdout)


def _pytesseract_binary() -> str:
    try:
        import pytesseract

        return str(getattr(pytesseract.pytesseract, "tesseract_cmd", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def parse_tesseract_data(data: dict) -> list[TextObservation]:
    """Parse pytesseract's ``image_to_data`` DICT output into observations.

    Only words (not layout blocks) with an acceptable confidence are kept, and
    words on the same text line are grouped into a single observation so a
    consumer sees "Continue" as one run of text even when it is one word.
    """

    observations: list[TextObservation] = []
    texts = data.get("text") if isinstance(data, dict) else None
    if not isinstance(texts, list):
        return []
    # Group consecutive words that share (block, par, line) into one run.
    current: dict | None = None
    for index, raw_text in enumerate(texts):
        word = str(raw_text or "").strip()
        if not word:
            continue
        confidence = _confidence_to_unit(_value_at(data, "conf", index))
        if confidence * 100.0 < _MIN_WORD_CONFIDENCE:
            continue
        left = _value_at(data, "left", index)
        top = _value_at(data, "top", index)
        width = _value_at(data, "width", index)
        height = _value_at(data, "height", index)
        key = (
            _value_at(data, "block_num", index),
            _value_at(data, "par_num", index),
            _value_at(data, "line_num", index),
        )
        if current is not None and current["key"] == key:
            current["words"].append(word)
            current["right"] = max(current["right"], int(left) + int(width))
            current["bottom"] = max(current["bottom"], int(top) + int(height))
            current["confidence"] = min(current["confidence"], confidence)
        else:
            if current is not None:
                observations.append(_observation_from_run(current))
            current = {
                "key": key,
                "words": [word],
                "left": int(left),
                "top": int(top),
                "right": int(left) + int(width),
                "bottom": int(top) + int(height),
                "confidence": confidence,
            }
    if current is not None:
        observations.append(_observation_from_run(current))
    return observations


def parse_tesseract_tsv(text: str) -> list[TextObservation]:
    """Parse ``tesseract ... tsv`` output into text observations."""

    lines = str(text or "").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    try:
        index = {name: position for position, name in enumerate(header)}
    except Exception:  # noqa: BLE001
        return []
    needed = ("text", "conf", "left", "top", "width", "height")
    if any(name not in index for name in needed):
        return []
    observations: list[TextObservation] = []
    current: dict | None = None
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) < len(header):
            continue
        word = cells[index["text"]].strip()
        if not word:
            continue
        confidence = _confidence_to_unit(cells[index["conf"]])
        if confidence * 100.0 < _MIN_WORD_CONFIDENCE:
            continue
        key = (
            cells[index["block_num"]] if "block_num" in index else "",
            cells[index["par_num"]] if "par_num" in index else "",
            cells[index["line_num"]] if "line_num" in index else "",
        )
        left = _safe_int(cells[index["left"]])
        top = _safe_int(cells[index["top"]])
        width = _safe_int(cells[index["width"]])
        height = _safe_int(cells[index["height"]])
        if current is not None and current["key"] == key:
            current["words"].append(word)
            current["right"] = max(current["right"], left + width)
            current["bottom"] = max(current["bottom"], top + height)
            current["confidence"] = min(current["confidence"], confidence)
        else:
            if current is not None:
                observations.append(_observation_from_run(current))
            current = {
                "key": key,
                "words": [word],
                "left": left,
                "top": top,
                "right": left + width,
                "bottom": top + height,
                "confidence": confidence,
            }
    if current is not None:
        observations.append(_observation_from_run(current))
    return observations


def _observation_from_run(run: dict) -> TextObservation:
    return TextObservation(
        text=" ".join(run["words"]),
        bounds=Bounds(run["left"], run["top"], run["right"], run["bottom"]),
        confidence=run["confidence"],
        source=ObservationSource.OCR,
    )


def _value_at(data: dict, key: str, index: int) -> object:
    values = data.get(key)
    if isinstance(values, list) and index < len(values):
        return values[index]
    return 0


def _safe_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def build_ocr_provider(*, name: str = "tesseract", languages: str = "eng", timeout: float = 20.0) -> OcrProvider:
    """Construct the configured OCR provider, or a null provider when missing."""

    if str(name).casefold() in {"tesseract", ""}:
        provider = TesseractOcrProvider(languages=languages, timeout=timeout)
        return provider if provider.available() else NullOcrProvider()
    return NullOcrProvider()
