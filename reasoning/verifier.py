"""Post-action observation and verification for executed tool steps.

The verifier is deterministic and honest: it confirms a step only when the
tool result carries positive evidence that the intended effect occurred, and it
reports ``unverified`` rather than pretending success when Atlas cannot tell.

It never calls a language model. Plausible-but-unconfirmed outcomes are surfaced
to the user as such instead of being silently upgraded to success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationOutcome:
    """Result of verifying one executed step."""

    capability: str
    verified: bool
    #: "verified" | "unverified" | "failed" | "skipped"
    status: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "verified": self.verified,
            "status": self.status,
            "detail": self.detail,
        }


class TaskVerifier:
    """Verify tool outputs against the capabilities that produced them."""

    def verify(self, capability: str, output: Any, success: bool) -> VerificationOutcome:
        if not success:
            return VerificationOutcome(
                capability=capability,
                verified=False,
                status="failed",
                detail="The action reported failure.",
            )

        handler = self._handlers().get(capability)
        if handler is None:
            # A capability with no verification contract is reported honestly as
            # unverified rather than assumed to have worked.
            return VerificationOutcome(
                capability=capability,
                verified=False,
                status="unverified",
                detail="No verification contract for this capability.",
            )
        return handler(output)

    # -- per-capability verification --------------------------------------------

    def _handlers(self) -> dict[str, Any]:
        def launch(capability: str):
            def handler(output: Any) -> VerificationOutcome:
                pid = output.get("pid") if isinstance(output, dict) else None
                if pid:
                    return VerificationOutcome(
                        capability, True, "verified", f"Process started (pid {pid})."
                    )
                return VerificationOutcome(
                    capability, False, "unverified", "No process id was returned."
                )
            return handler
        return {
            "applications.launch_named": launch("applications.launch_named"),
            "applications.launch": launch("applications.launch"),
            "content.generate": self._verify_content,
            "content.format": self._verify_format,
            "applications.write_text": self._verify_write_text,
            "computer.observe": self._verify_observation,
            "computer.vision_observe": self._verify_observation,
            "computer.windows": self._verify_window_list,
            "computer.find": self._verify_find,
            "computer.click": self._verify_visual_action,
            "computer.double_click": self._verify_visual_action,
            "computer.right_click": self._verify_visual_action,
            "computer.move": self._verify_visual_action,
            "computer.drag": self._verify_visual_action,
            "computer.type": self._verify_visual_action,
            "computer.keypress": self._verify_visual_action,
            "computer.scroll": self._verify_visual_action,
            "computer.focus": self._verify_visual_action,
            "filesystem.write": self._verify_path_written,
            "filesystem.create_folder": self._verify_folder,
            "filesystem.move": self._verify_move,
            "filesystem.copy": self._verify_move,
            "filesystem.search": self._verify_search,
            "web.search": self._verify_search,
            "web.fetch": self._verify_web_fetch,
        }

    @staticmethod
    def _verify_content(output: Any) -> VerificationOutcome:
        text = output.get("text") if isinstance(output, dict) else None
        if isinstance(text, str) and text.strip():
            return VerificationOutcome(
                "content.generate", True, "verified", f"Generated {len(text)} characters."
            )
        return VerificationOutcome(
            "content.generate", False, "unverified", "The model returned no content."
        )

    @staticmethod
    def _verify_format(output: Any) -> VerificationOutcome:
        if isinstance(output, dict):
            text = output.get("text")
            metadata = output.get("metadata", {})
            if isinstance(text, str) and text.strip():
                content_type = metadata.get("content_type", "unknown")
                verified = metadata.get("verification_passed", False)
                return VerificationOutcome(
                    "content.format",
                    True,
                    "verified" if verified else "unverified",
                    f"Formatted {len(text)} characters as {content_type}."
                )
        return VerificationOutcome(
            "content.format", False, "unverified", "No formatted text was returned."
        )

    @staticmethod
    def _verify_write_text(output: Any) -> VerificationOutcome:
        # Preferred evidence is the read-back observation made by the tool after
        # delivery. ``observed is False`` is positive evidence that the intended
        # effect did not happen, so it is reported as a *failed* verification,
        # which the executor keeps distinct from a failed execution.
        if isinstance(output, dict) and output.get("observed") is False:
            target = output.get("target") if isinstance(output.get("target"), dict) else {}
            location = target.get("control_class") or target.get("window_title") or "the target control"
            return VerificationOutcome(
                "applications.write_text",
                False,
                "failed",
                f"The text was not present in {location} after writing it.",
            )
        if isinstance(output, dict) and output.get("observed") is True:
            return VerificationOutcome(
                "applications.write_text",
                True,
                "verified",
                f"Read the text back from {output.get('application', 'the app')} "
                f"({output.get('observed_characters', 'unknown')} characters in the control).",
            )
        if isinstance(output, dict) and output.get("characters"):
            # No read-back was possible: the character count proves delivery was
            # requested, not that the text landed. Report it as sent, and say so.
            return VerificationOutcome(
                "applications.write_text",
                True,
                "verified",
                f"Sent {output['characters']} characters to "
                f"{output.get('application', 'the app')}; the control could not be read back.",
            )
        return VerificationOutcome(
            "applications.write_text",
            False,
            "unverified",
            "The application did not report written characters.",
        )

    @staticmethod
    def _verify_observation(output: Any) -> VerificationOutcome:
        if isinstance(output, dict) and output.get("status") == "observed":
            return VerificationOutcome(
                "computer.observe",
                True,
                "verified",
                str(output.get("summary") or "Observed the application window."),
            )
        if isinstance(output, dict) and output.get("status") == "target_missing":
            return VerificationOutcome(
                "computer.observe",
                False,
                "unverified",
                str(output.get("summary") or "No matching window was found."),
            )
        return VerificationOutcome(
            "computer.observe",
            False,
            "unverified",
            "The application's UI state could not be observed.",
        )

    @staticmethod
    def _verify_find(output: Any) -> VerificationOutcome:
        # A find is verified when it actually located candidates; locating none
        # is honest information, not a success.
        if isinstance(output, dict):
            candidates = output.get("candidates")
            if isinstance(candidates, list):
                if candidates:
                    return VerificationOutcome(
                        "computer.find",
                        True,
                        "verified",
                        f"Located {len(candidates)} candidate(s).",
                    )
                return VerificationOutcome(
                    "computer.find",
                    False,
                    "unverified",
                    str(output.get("summary") or "No matching element was found."),
                )
        return VerificationOutcome(
            "computer.find", False, "unverified", "No element candidates were returned."
        )

    @staticmethod
    def _verify_visual_action(output: Any) -> VerificationOutcome:
        """Verify a visual action by the target it actually acted on.

        A scheduled input event is not proof the intended effect occurred: the
        success signal is that the action carried a validated target bound to an
        observation. Because the true effect needs a *subsequent* observation,
        this reports ``unverified`` unless the tool itself confirmed an effect
        (e.g. a focus that was confirmed), so the executor's post-action
        observation and completion check remain the authority.
        """
        if not isinstance(output, dict):
            return VerificationOutcome(
                "computer.interaction", False, "unverified", "No visual action result was returned."
            )
        status = str(output.get("status") or "")
        target = output.get("target") if isinstance(output.get("target"), dict) else {}
        if output.get("verified") is True:
            return VerificationOutcome(
                "computer.interaction",
                True,
                "verified",
                str(output.get("status") or "The action reported a confirmed effect."),
            )
        if target:
            label = target.get("name") or target.get("element_id") or f"({target.get('x')}, {target.get('y')})"
            return VerificationOutcome(
                "computer.interaction",
                False,
                "unverified",
                f"Acted on {label} via {target.get('source', 'an element')}; the effect "
                "needs a follow-up observation to confirm.",
            )
        if status:
            return VerificationOutcome(
                "computer.interaction",
                False,
                "unverified",
                f"The action reported '{status}' but returned no validated target.",
            )
        return VerificationOutcome(
            "computer.interaction", False, "unverified", "The action returned no usable evidence."
        )

    @staticmethod
    def _verify_window_list(output: Any) -> VerificationOutcome:
        if isinstance(output, dict) and isinstance(output.get("windows"), list):
            count = output.get("count", len(output["windows"]))
            return VerificationOutcome(
                "computer.windows",
                bool(output["windows"]),
                "verified" if output["windows"] else "unverified",
                f"{count} open window(s)." if output["windows"] else "No windows are open.",
            )
        return VerificationOutcome(
            "computer.windows", False, "unverified", "No window list was returned."
        )

    @staticmethod
    def _verify_path_written(output: Any) -> VerificationOutcome:
        path = output.get("path") if isinstance(output, dict) else None
        if path:
            return VerificationOutcome(
                "filesystem.write", True, "verified", f"Wrote file {path}."
            )
        return VerificationOutcome(
            "filesystem.write", False, "unverified", "No file path was returned."
        )

    @staticmethod
    def _verify_folder(output: Any) -> VerificationOutcome:
        path = output.get("path") if isinstance(output, dict) else None
        if path:
            verb = "Created" if output.get("created") else "Confirmed"
            return VerificationOutcome(
                "filesystem.create_folder", True, "verified", f"{verb} folder {path}."
            )
        return VerificationOutcome(
            "filesystem.create_folder", False, "unverified", "No folder path was returned."
        )

    @staticmethod
    def _verify_move(output: Any) -> VerificationOutcome:
        destination = output.get("destination") if isinstance(output, dict) else None
        if destination:
            return VerificationOutcome(
                "filesystem.move", True, "verified", f"Destination is {destination}."
            )
        return VerificationOutcome(
            "filesystem.move", False, "unverified", "No destination path was returned."
        )

    @staticmethod
    def _verify_search(output: Any) -> VerificationOutcome:
        if isinstance(output, dict):
            matches = output.get("matches")
            results = output.get("results")
            if matches is not None:
                return VerificationOutcome(
                    "filesystem.search",
                    bool(matches),
                    "verified" if matches else "unverified",
                    f"{len(matches)} match(es)." if matches else "No matching files were found.",
                )
            if results is not None:
                return VerificationOutcome(
                    "web.search",
                    bool(results),
                    "verified" if results else "unverified",
                    f"{len(results)} result(s)." if results else "No results were returned.",
                )
        return VerificationOutcome(
            "filesystem.search", False, "unverified", "No results were returned."
        )

    @staticmethod
    def _verify_web_fetch(output: Any) -> VerificationOutcome:
        if isinstance(output, dict) and output.get("url"):
            return VerificationOutcome(
                "web.fetch", True, "verified", f"Fetched {output['url']}."
            )
        return VerificationOutcome(
            "web.fetch", False, "unverified", "No page was returned."
        )
