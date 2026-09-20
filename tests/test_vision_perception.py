"""Deterministic tests for the local vision ("Eyes") perception layer.

No GUI session is required: every layer is driven through an injected fake
(UIA window source, OCR provider, VLM provider, screenshot capture, input
backend). These tests cover the visual-state models, layer parsing, coordinate
conversion, region handling, staleness/validation, confidence, tool
registration, permissions, verification, and recovery.
"""

from __future__ import annotations

import unittest

from computer.vision.actions import validate_visual_target
from computer.vision.models import (
    Bounds,
    ElementType,
    ObservationSource,
    ScreenInfo,
    TextObservation,
    VisualElement,
    VisualState,
)
from computer.vision.ocr import (
    NullOcrProvider,
    parse_tesseract_data,
    parse_tesseract_tsv,
)
from computer.vision.perception import ObserveRequest, PerceptionEngine
from computer.vision.providers import (
    NullVisionProvider,
    coerce_bounds,
    coerce_vlm_result,
    extract_json,
    proposed_elements,
)
from computer.vision.uia import UiaElement, UiaWindow, parse_uia_tree, uia_elements_to_visual


# --- fakes -------------------------------------------------------------------


def _window_payload() -> list[dict]:
    return [
        {
            "hwnd": 101,
            "title": "Untitled - Notepad",
            "class_name": "Notepad",
            "pid": 4242,
            "application": "notepad.exe",
            "bounds": (100, 100, 900, 700),
            "focused": True,
            "elements": [
                {"name": "File", "control_type": 50011, "bounds": (110, 110, 150, 130)},
                {"name": "", "control_type": 50004, "bounds": (120, 200, 880, 680), "value": "hello"},
                {"name": "Continue", "control_type": 50000, "bounds": (700, 640, 780, 670)},
                {"name": "disabled", "control_type": 50000, "bounds": (10, 10, 40, 30), "enabled": False},
            ],
        }
    ]


class FakeWindowsProvider:
    def __init__(self, payload):
        self.payload = payload

    def __call__(self):
        return self.payload


class FakeOcrProvider(NullOcrProvider):
    name = "fake"

    def __init__(self, runs):
        self.runs = runs
        self.calls = 0

    def available(self) -> bool:
        return True

    def recognize(self, image: bytes):
        self.calls += 1
        return list(self.runs)


class FakeCapture:
    def __init__(self):
        from computer.vision.imaging import ScreenImage

        self.screen_image = ScreenImage(
            data=b"\x89PNG\r\n\x1a\nfake", width=1920, height=1080, region=Bounds(0, 0, 1920, 1080)
        )
        self.calls = 0

    def __call__(self, region, max_size=0):
        self.calls += 1
        return self.screen_image


class FakeVisionProvider(NullVisionProvider):
    name = "fake"

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def available(self) -> bool:
        return True

    def query(self, *, image, task, question="", context=None, timeout=None):
        self.calls += 1
        return dict(self.result)


class FakeInputBackend:
    def __init__(self):
        self.events: list[tuple] = []

    def move(self, x, y):
        self.events.append(("move", x, y))

    def click(self, x, y, *, button="left", count=1):
        self.events.append(("click", x, y, button, count))

    def drag(self, sx, sy, ex, ey, *, button="left"):
        self.events.append(("drag", sx, sy, ex, ey))

    def scroll(self, amount):
        self.events.append(("scroll", amount))

    def keypress(self, key):
        self.events.append(("key", key))

    def type_text(self, text):
        self.events.append(("type", text))


def _engine(
    *,
    ocr=None,
    vision=None,
    windows=None,
    capture=None,
    allow_vision=True,
):
    from computer.vision.imaging import ScreenImage

    capture = capture or FakeCapture()
    return PerceptionEngine(
        ocr=ocr if ocr is not None else NullOcrProvider(),
        vision=vision if vision is not None else NullVisionProvider(),
        windows_provider=windows or FakeWindowsProvider(_window_payload()),
        capture=capture,
        max_image_size=0,
    )


# --- model tests -------------------------------------------------------------


class VisualStateModelTests(unittest.TestCase):
    def test_bounds_derived_geometry(self):
        bounds = Bounds(10, 20, 40, 60)
        self.assertEqual(bounds.width, 30)
        self.assertEqual(bounds.height, 40)
        self.assertEqual(bounds.center, (25, 40))
        self.assertEqual(bounds.area, 1200)

    def test_bounds_contains_point_and_intersection(self):
        bounds = Bounds(0, 0, 100, 100)
        self.assertTrue(bounds.contains_point(50, 50))
        self.assertFalse(bounds.contains_point(150, 50))
        self.assertTrue(bounds.contains_point(101, 101, tolerance=2))
        self.assertIsNone(bounds.intersection(Bounds(200, 200, 300, 300)))
        self.assertEqual(bounds.intersection(Bounds(50, 50, 150, 150)), Bounds(50, 50, 100, 100))

    def test_bounds_from_rect_variants(self):
        self.assertEqual(Bounds.from_rect((1, 2, 3, 4)), Bounds(1, 2, 3, 4))
        self.assertEqual(Bounds.from_rect({"left": 1, "top": 2, "width": 3, "height": 4}), Bounds(1, 2, 4, 6))
        self.assertEqual(Bounds.from_rect("garbage"), Bounds(0, 0, 0, 0))

    def test_visual_state_to_dict_is_json_shaped(self):
        state = VisualState(screen=ScreenInfo(1920, 1080))
        state.elements.append(
            VisualElement(
                id="e1",
                type=ElementType.BUTTON,
                name="Continue",
                bounds=Bounds(1, 2, 3, 4),
                observation_id=state.observation_id,
            )
        )
        payload = state.to_dict()
        self.assertEqual(payload["observation_id"], state.observation_id)
        self.assertEqual(payload["elements"][0]["type"], "button")
        self.assertEqual(payload["screen"]["width"], 1920)

    def test_find_by_name_ranks_exact_then_partial(self):
        state = VisualState()
        state.elements = [
            VisualElement(id="a", type=ElementType.TEXT, name="Continue later"),
            VisualElement(id="b", type=ElementType.BUTTON, name="Continue"),
            VisualElement(id="c", type=ElementType.BUTTON, name="Cancel"),
        ]
        matches = state.find_by_name("continue")
        self.assertEqual(matches[0].id, "b")

    def test_full_text_reads_in_reading_order(self):
        state = VisualState()
        state.text = [
            TextObservation("bottom", Bounds(0, 100, 10, 120)),
            TextObservation("top", Bounds(0, 0, 10, 20)),
        ]
        self.assertEqual(state.full_text(), "top\nbottom")

    def test_staleness(self):
        state = VisualState(timestamp=0.0)
        self.assertTrue(state.is_stale(max_age_seconds=1.0))
        self.assertFalse(VisualState().is_stale(max_age_seconds=30.0))


# --- UIA tests ---------------------------------------------------------------


class UiaParsingTests(unittest.TestCase):
    def test_parse_uia_tree_skips_malformed_entries(self):
        windows = parse_uia_tree([{"hwnd": "bad"}, 42, _window_payload()[0]])
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].hwnd, 101)

    def test_uia_elements_to_visual_types_and_confidence(self):
        window = parse_uia_tree(_window_payload())[0]
        elements = uia_elements_to_visual(window, observation_id="obs_1")
        by_name = {element.name: element for element in elements if element.name}
        self.assertEqual(by_name["Continue"].type, ElementType.BUTTON)
        self.assertEqual(by_name["Continue"].source, ObservationSource.UIA)
        self.assertGreater(by_name["Continue"].confidence, 0.9)
        self.assertEqual(by_name["File"].type, ElementType.MENU_ITEM)

    def test_uia_element_with_value_no_name_is_kept(self):
        window = parse_uia_tree(_window_payload())[0]
        elements = uia_elements_to_visual(window)
        editable = [element for element in elements if element.type == ElementType.TEXT_FIELD]
        self.assertTrue(editable)
        self.assertEqual(editable[0].value, "hello")

    def test_control_type_mapping(self):
        self.assertEqual(UiaElement(name="x", control_type=50000).element_type, ElementType.BUTTON)
        self.assertEqual(UiaElement(name="x", control_type=50004).element_type, ElementType.TEXT_FIELD)
        self.assertEqual(UiaElement(name="x", control_type=99999).element_type, ElementType.UNKNOWN)

    def test_observation_id_propagates_to_elements(self):
        window = parse_uia_tree(_window_payload())[0]
        elements = uia_elements_to_visual(window, observation_id="obs_xyz")
        self.assertTrue(all(element.observation_id == "obs_xyz" for element in elements))


# --- OCR tests ---------------------------------------------------------------


class OcrParsingTests(unittest.TestCase):
    def test_parse_tesseract_data_groups_words_on_a_line(self):
        data = {
            "text": ["Continue", "later", "", "Cancel"],
            "conf": ["95", "80", "-1", "70"],
            "left": [10, 70, 0, 10],
            "top": [100, 100, 0, 200],
            "width": [50, 60, 0, 40],
            "height": [20, 20, 0, 20],
            "block_num": [1, 1, 1, 1],
            "par_num": [1, 1, 1, 1],
            "line_num": [1, 1, 1, 2],
        }
        runs = parse_tesseract_data(data)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0].text, "Continue later")
        self.assertEqual(runs[0].bounds, Bounds(10, 100, 130, 120))
        self.assertEqual(runs[1].text, "Cancel")

    def test_parse_tesseract_data_drops_low_confidence(self):
        data = {
            "text": ["noise"],
            "conf": ["10"],
            "left": [0],
            "top": [0],
            "width": [5],
            "height": [5],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
        }
        self.assertEqual(parse_tesseract_data(data), [])

    def test_confidence_is_normalized_to_unit_interval(self):
        data = {
            "text": ["a"],
            "conf": ["100"],
            "left": [0],
            "top": [0],
            "width": [5],
            "height": [5],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
        }
        self.assertEqual(parse_tesseract_data(data)[0].confidence, 1.0)

    def test_parse_tesseract_tsv(self):
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t10\t20\t50\t18\t92\tSearch\n"
            "5\t1\t1\t1\t1\t2\t65\t20\t60\t18\t88\tYouTube\n"
        )
        runs = parse_tesseract_tsv(tsv)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].text, "Search YouTube")

    def test_parse_tesseract_tsv_handles_garbage(self):
        self.assertEqual(parse_tesseract_tsv("not tsv at all"), [])


# --- perception orchestrator -------------------------------------------------


class PerceptionEngineTests(unittest.TestCase):
    def test_observe_builds_visual_state_from_uia(self):
        engine = _engine()
        state = engine.observe(ObserveRequest(include_ocr=False))
        self.assertEqual(state.detected_application, "notepad.exe")
        self.assertIsNotNone(state.active_window)
        self.assertEqual(state.active_window.hwnd, 101)
        self.assertIn("uia", state.sources)
        self.assertTrue(any(element.name == "Continue" for element in state.elements))

    def test_observe_includes_ocr_text_and_elements(self):
        ocr = FakeOcrProvider(
            [TextObservation("Search", Bounds(200, 200, 260, 220), confidence=0.9)]
        )
        engine = _engine(ocr=ocr)
        state = engine.observe(ObserveRequest(include_ocr=True))
        self.assertIn("ocr", state.sources)
        self.assertEqual(ocr.calls, 1)
        self.assertTrue(any(run.text == "Search" for run in state.text))

    def test_ocr_region_preference_filters_outside_text(self):
        # OCR text outside the active window's bounds is dropped.
        ocr = FakeOcrProvider(
            [TextObservation("outside", Bounds(2000, 2000, 2100, 2050), confidence=0.9)]
        )
        engine = _engine(ocr=ocr)
        state = engine.observe(ObserveRequest(include_ocr=True))
        self.assertFalse(any(run.text == "outside" for run in state.text))

    def test_vlm_only_consulted_when_allowed(self):
        vision = FakeVisionProvider(
            {"found": True, "label": "Sign in", "type": "button", "bounds": {"left": 1, "top": 2, "right": 3, "bottom": 4}, "confidence": 0.9}
        )
        engine = _engine(vision=vision)
        engine.observe(ObserveRequest(allow_vlm=False))
        self.assertEqual(vision.calls, 0)
        state = engine.observe(ObserveRequest(allow_vlm=True, vlm_question="find login"))
        self.assertEqual(vision.calls, 1)
        self.assertIn("vlm", state.sources)

    def test_vlm_proposals_below_threshold_are_ignored(self):
        from config import VISION_CONFIDENCE_THRESHOLD

        vision = FakeVisionProvider(
            {"found": True, "label": "maybe", "type": "button", "bounds": {"left": 1, "top": 2, "right": 3, "bottom": 4}, "confidence": VISION_CONFIDENCE_THRESHOLD - 0.3}
        )
        engine = _engine(vision=vision)
        state = engine.observe(ObserveRequest(allow_vlm=True, vlm_question="x"))
        self.assertFalse(any(element.source == ObservationSource.VLM for element in state.elements))

    def test_disabled_vision_reports_notes(self):
        engine = _engine()
        state = engine.observe(ObserveRequest(allow_vlm=True, vlm_question="x"))
        self.assertTrue(any("No local vision model" in note for note in state.notes))

    def test_find_matches_by_label_and_by_type(self):
        engine = _engine()
        by_label = engine.find("Continue")
        self.assertTrue(by_label)
        self.assertEqual(by_label[0].name, "Continue")
        by_type = engine.find("the search box")
        self.assertTrue(all(element.type == ElementType.TEXT_FIELD for element in by_type))

    def test_application_selection_by_name(self):
        engine = _engine()
        state = engine.observe(ObserveRequest(application="Notepad", include_ocr=False))
        self.assertEqual(state.active_window.hwnd, 101)

    def test_missing_application_yields_no_active_window(self):
        engine = _engine()
        state = engine.observe(ObserveRequest(application="Photoshop", include_ocr=False))
        self.assertIsNone(state.active_window)

    def test_full_screen_region_none_and_explicit_region(self):
        capture = FakeCapture()
        engine = _engine(capture=capture)
        engine.observe(ObserveRequest(full_screen=True, include_ocr=False))
        engine.observe(ObserveRequest(region=Bounds(0, 0, 100, 100), include_ocr=False))
        self.assertEqual(capture.calls, 2)

    def test_change_ratio_is_zero_on_first_capture(self):
        engine = _engine()
        state = engine.observe(ObserveRequest())
        self.assertEqual(state.screenshot_metadata.get("change_ratio"), 0.0)


# --- validation / coordinate safety ------------------------------------------


class VisualTargetValidationTests(unittest.TestCase):
    def _state(self) -> VisualState:
        state = VisualState(screen=ScreenInfo(1920, 1080))
        state.elements = [
            VisualElement(
                id="b1",
                type=ElementType.BUTTON,
                name="Continue",
                bounds=Bounds(700, 640, 780, 670),
                confidence=0.99,
                source=ObservationSource.UIA,
                observation_id=state.observation_id,
            ),
            VisualElement(
                id="d1",
                type=ElementType.BUTTON,
                name="Disabled",
                bounds=Bounds(10, 10, 40, 30),
                enabled=False,
                observation_id=state.observation_id,
            ),
        ]
        return state

    def test_by_element_id_succeeds(self):
        state = self._state()
        result = validate_visual_target(state=state, element_id="b1")
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.target.observation_id, state.observation_id)
        self.assertEqual(result.target.name, "Continue")

    def test_by_description_succeeds(self):
        state = self._state()
        result = validate_visual_target(state=state, description="continue")
        self.assertTrue(result.ok, result.reason)

    def test_missing_element_is_rejected(self):
        result = validate_visual_target(state=self._state(), element_id="nope")
        self.assertFalse(result.ok)
        self.assertIn("not present", result.reason)

    def test_coordinates_outside_element_are_rejected(self):
        result = validate_visual_target(state=self._state(), element_id="b1", x=0, y=0)
        self.assertFalse(result.ok)
        self.assertIn("outside", result.reason)

    def test_coordinates_inside_element_are_accepted(self):
        result = validate_visual_target(state=self._state(), element_id="b1", x=740, y=655)
        self.assertTrue(result.ok, result.reason)
        self.assertEqual((result.target.x, result.target.y), (740, 655))

    def test_disabled_element_is_rejected(self):
        result = validate_visual_target(state=self._state(), element_id="d1")
        self.assertFalse(result.ok)
        self.assertIn("disabled", result.reason)

    def test_stale_observation_is_rejected(self):
        state = self._state()
        state.timestamp = 0.0
        result = validate_visual_target(state=state, element_id="b1", max_age_seconds=1.0)
        self.assertFalse(result.ok)
        self.assertIn("stale", result.reason)

    def test_wrong_active_application_is_rejected(self):
        state = self._state()
        state.detected_application = "notepad.exe"
        result = validate_visual_target(state=state, element_id="b1", expected_application="chrome")
        self.assertFalse(result.ok)
        self.assertIn("active application", result.reason)

    def test_no_observation_with_coordinates_is_flagged(self):
        result = validate_visual_target(state=None, x=5, y=5)
        self.assertTrue(result.ok)
        self.assertEqual(result.target.source, "explicit")
        self.assertIn("without visual verification", result.reason)

    def test_no_observation_and_no_coordinates_is_rejected(self):
        result = validate_visual_target(state=None)
        self.assertFalse(result.ok)

    def test_explicit_coordinates_within_screen_accepted(self):
        result = validate_visual_target(state=self._state(), x=10, y=10)
        self.assertTrue(result.ok)

    def test_explicit_coordinates_outside_screen_rejected(self):
        result = validate_visual_target(state=self._state(), x=99999, y=10)
        self.assertFalse(result.ok)


# --- VLM provider coercion ---------------------------------------------------


class VisionProviderCoercionTests(unittest.TestCase):
    def test_extract_json_from_fenced_block(self):
        text = 'Here you go:\n```json\n{"answer": "notepad"}\n```'
        self.assertEqual(extract_json(text), {"answer": "notepad"})

    def test_extract_json_from_prose(self):
        self.assertEqual(extract_json('sure: {"a": 1} done'), {"a": 1})

    def test_extract_json_returns_none_for_garbage(self):
        self.assertIsNone(extract_json("no json here"))

    def test_coerce_vlm_result_adds_source_and_clamps(self):
        result = coerce_vlm_result('{"confidence": 5}', task="describe_screen")
        self.assertEqual(result["confidence"], 1.0)
        self.assertEqual(result["source"], "vlm")
        self.assertTrue(result["available"])

    def test_unstructured_output_is_reported_not_trusted(self):
        result = coerce_vlm_result("I think it is Notepad", task="identify_application")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["error"], "unstructured_output")

    def test_proposed_elements_extracts_list_and_single(self):
        single = coerce_vlm_result(
            '{"found": true, "label": "Continue", "bounds": {"left": 1, "top": 2, "right": 3, "bottom": 4}, "confidence": 0.8}',
            task="locate_element",
        )
        proposals = proposed_elements(single)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0][0], "Continue")
        self.assertEqual(proposals[0][2], Bounds(1, 2, 3, 4))

    def test_coerce_bounds_rejects_negative(self):
        self.assertIsNone(coerce_bounds({"left": 0, "top": 0, "right": -5, "bottom": -5}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
