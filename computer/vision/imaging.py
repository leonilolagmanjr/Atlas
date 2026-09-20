"""Screen capture and local image processing for Atlas.

Layer 3 of perception. Screen capture uses the Win32 GDI/ctypes path (no extra
Python dependency required for capture on Windows), producing a PNG-encoded
bytes buffer plus metadata. Image processing uses OpenCV/numpy *when available*
and degrades to a pure-Python no-op when it is not, so deterministic perception
never hard-fails on a missing optional dependency.

Nothing here executes an action; it only reads pixels. Screenshots are held in
memory and returned as bytes; they are never written to disk by default.
"""

from __future__ import annotations

import ctypes
import io
import sys
from dataclasses import dataclass
from typing import Any, Optional

from computer.vision.models import Bounds


def screen_capture_available() -> bool:
    """Return True when this platform can capture the screen."""

    return sys.platform == "win32"


def imaging_available() -> bool:
    """Return True when OpenCV/numpy image processing is importable."""

    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001 - optional dependency
        return False
    return True


@dataclass(frozen=True)
class ScreenImage:
    """A captured screen image plus the region it represents."""

    data: bytes
    width: int
    height: int
    region: Bounds
    mode: str = "RGB"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "region": self.region.to_dict(),
            "bytes": len(self.data),
            "mode": self.mode,
        }


# --- virtual screen geometry -------------------------------------------------


def virtual_screen_bounds() -> Bounds:
    """Return the bounding rectangle of all monitors (virtual desktop)."""

    if not screen_capture_available():
        return Bounds(0, 0, 0, 0)
    try:
        user32 = ctypes.windll.user32
        return Bounds(
            0,
            0,
            int(user32.GetSystemMetrics(0)),  # SM_CXSCREEN
            int(user32.GetSystemMetrics(1)),  # SM_CYSCREEN
        )
    except Exception:  # noqa: BLE001 - geometry is best-effort
        return Bounds(0, 0, 0, 0)


def monitors() -> tuple[dict[str, Any], ...]:
    """Return per-monitor geometry using the Win32 monitor enumeration API."""

    if not screen_capture_available():
        return ()

    class _MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", ctypes.c_long * 4),
            ("rcWork", ctypes.c_long * 4),
            ("dwFlags", ctypes.c_ulong),
        ]

    found: list[dict[str, Any]] = []
    try:
        user32 = ctypes.windll.user32
    except Exception:  # noqa: BLE001 - non-Windows
        return ()

    MONITORENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(ctypes.c_long * 4), ctypes.c_double
    )

    def collect(_handle, _hdc, _rect, _param) -> int:
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        try:
            if user32.GetMonitorInfoW(_handle, ctypes.byref(info)):
                rect = info.rcMonitor
                primary = bool(info.dwFlags & 1)  # MONITORINFOF_PRIMARY
                found.append(
                    {
                        "left": int(rect[0]),
                        "top": int(rect[1]),
                        "right": int(rect[2]),
                        "bottom": int(rect[3]),
                        "primary": primary,
                    }
                )
        except Exception:  # noqa: BLE001 - skip unreadable monitors
            pass
        return 1

    try:
        user32.EnumDisplayMonitors(0, 0, MONITORENUMPROC(collect), 0)
    except Exception:  # noqa: BLE001
        return tuple(found)
    return tuple(found)


# --- capture -----------------------------------------------------------------


def capture_region(region: Optional[Bounds] = None, *, max_size: int = 0) -> ScreenImage:
    """Capture a screen region and return it as PNG bytes.

    ``max_size`` downscales the longest edge when it is exceeded, which lowers
    OCR and VLM cost. It only applies when OpenCV/numpy is available; otherwise
    the raw capture is returned unscaled.
    """

    if not screen_capture_available():
        raise OSError("Screen capture requires Windows")

    full = virtual_screen_bounds()
    target = region or full
    width = max(1, target.width)
    height = max(1, target.height)

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hdc_screen = user32.GetDC(0)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    bitmap = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
    gdi32.SelectObject(hdc_mem, bitmap)
    try:
        SRCCOPY = 0x00CC0020
        gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_screen, target.left, target.top, SRCCOPY)
        image = _bitmap_to_png(gdi32, bitmap, width, height)
    finally:
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)

    if max_size and max(width, height) > max_size and imaging_available():
        image = downscale(image, max_size=max_size)
    return image


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


def _bitmap_to_png(gdi32: Any, bitmap: Any, width: int, height: int) -> ScreenImage:
    """Read a GDI bitmap into a top-down BGRA buffer and PNG-encode it."""

    header = _BITMAPINFOHEADER()
    header.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    header.biWidth = width
    # Negative height requests a top-down DIB so rows are not flipped.
    header.biHeight = -height
    header.biPlanes = 1
    header.biBitCount = 32
    header.biCompression = 0  # BI_RGB

    buffer = ctypes.create_string_buffer(width * height * 4)
    # GetDIBits needs a device context; use the screen DC.
    screen_dc = ctypes.windll.user32.GetDC(0)
    try:
        rows = gdi32.GetDIBits(screen_dc, bitmap, 0, height, buffer, ctypes.byref(header), 0)
    finally:
        ctypes.windll.user32.ReleaseDC(0, screen_dc)
    if not rows:
        raise OSError("Could not read the screen bitmap")

    png_bytes = _encode_png_bgra(bytes(buffer), width, height)
    return ScreenImage(data=png_bytes, width=width, height=height, region=virtual_screen_bounds())


def _encode_png_bgra(bgra: bytes, width: int, height: int) -> bytes:
    """PNG-encode a BGRA buffer.

    Uses OpenCV/PIL when available; falls back to a minimal pure-Python PNG
    writer (zlib + struct) so capture works without any imaging dependency.
    """

    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore

        array = np.frombuffer(bgra, dtype=np.uint8).reshape((height, width, 4))
        rgb = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
        ok, encoded = cv2.imencode(".png", rgb)
        if ok:
            return encoded.tobytes()
    except Exception:  # noqa: BLE001 - fall through to pure Python
        pass

    try:
        from PIL import Image  # type: ignore

        image = Image.frombytes("RGBA", (width, height), _bgra_to_rgba(bgra))
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
    except Exception:  # noqa: BLE001 - fall through to minimal encoder
        pass

    return _minimal_png(_bgra_to_rgba(bgra), width, height)


def _bgra_to_rgba(bgra: bytes) -> bytes:
    output = bytearray(len(bgra))
    output[0::4] = bgra[2::4]
    output[1::4] = bgra[1::4]
    output[2::4] = bgra[0::4]
    output[3::4] = bgra[3::4]
    return bytes(output)


def _minimal_png(rgba: bytes, width: int, height: int) -> bytes:
    """Encode raw RGBA to PNG with only the standard library."""

    import struct
    import zlib

    raw = bytearray()
    stride = width * 4
    for row in range(height):
        raw.append(0)  # filter type 0 (None)
        start = row * stride
        raw.extend(rgba[start:start + stride])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


# --- image processing (Layer 3) ----------------------------------------------


def downscale(image: ScreenImage, *, max_size: int) -> ScreenImage:
    """Downscale an image so its longest edge is at most ``max_size``."""

    if not imaging_available() or max_size <= 0:
        return image
    if max(image.width, image.height) <= max_size:
        return image
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore

        array = np.frombuffer(image.data, dtype=np.uint8)
        decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if decoded is None:
            return image
        scale = max_size / max(image.width, image.height)
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resized = cv2.resize(decoded, new_size, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", resized)
        if not ok:
            return image
        return ScreenImage(
            data=encoded.tobytes(),
            width=new_size[0],
            height=new_size[1],
            region=image.region,
            mode=image.mode,
        )
    except Exception:  # noqa: BLE001 - optional dependency
        return image


def decode_rgb(image: ScreenImage) -> Optional[Any]:
    """Decode image bytes into an RGB numpy array, or None when unavailable."""

    if not imaging_available():
        return None
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore

        array = np.frombuffer(image.data, dtype=np.uint8)
        decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if decoded is None:
            return None
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    except Exception:  # noqa: BLE001
        return None


def image_difference(previous: ScreenImage, current: ScreenImage) -> float:
    """Return a coarse [0, 1] change ratio between two same-size screenshots.

    Deterministic change detection: used to decide whether a UI actually
    changed, so an expensive VLM call is only spent on a real difference.
    Returns 0.0 when the images cannot be compared (never claims "changed").
    """

    if previous.width != current.width or previous.height != current.height:
        return 1.0
    if not imaging_available():
        # Without numpy, compare the encoded bytes coarsely.
        if previous.data == current.data:
            return 0.0
        return 1.0
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore

        left = cv2.imdecode(np.frombuffer(previous.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        right = cv2.imdecode(np.frombuffer(current.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None or left.shape != right.shape:
            return 0.0
        diff = cv2.absdiff(left, right)
        changed = float((diff > 12).sum()) / float(diff.size)
        return max(0.0, min(1.0, changed))
    except Exception:  # noqa: BLE001
        return 0.0


def find_color_regions(
    image: ScreenImage,
    *,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
    min_area: int = 200,
    max_regions: int = 20,
) -> list[Bounds]:
    """Return bounding boxes of connected regions matching an RGB color range.

    A deterministic primitive for localizing colored UI affordances (buttons,
    highlights) without a model. Returns an empty list when imaging is absent.
    """

    array = decode_rgb(image)
    if array is None:
        return []
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(array, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[Bounds] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width * height < min_area:
                continue
            regions.append(Bounds(x, y, x + width, y + height))
        regions.sort(key=lambda bounds: bounds.area, reverse=True)
        return regions[: max(1, max_regions)]
    except Exception:  # noqa: BLE001
        return []
