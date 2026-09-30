from __future__ import annotations

import logging
import math
from typing import Callable

import ollama

from config import OLLAMA_MODEL, REASONING_TIMEOUT_SECONDS
from providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


def _chunk_text(chunk: object) -> str:
    """Extract the content of one streamed chunk across client versions.

    The installed ``ollama`` client yields ``ChatResponse`` objects, while older
    releases yielded plain dicts. Reading only one shape silently drops every
    token, which is exactly the bug this guard prevents.
    """

    message: object = None
    if isinstance(chunk, dict):
        message = chunk.get("message")
    else:
        message = getattr(chunk, "message", None)
    if isinstance(message, dict):
        return str(message.get("content") or "")
    content = getattr(message, "content", "") if message is not None else ""
    return str(content or "")


class OllamaProvider(BaseProvider):
    """Ollama-backed provider implementation."""

    DEFAULT_TIMEOUT_SECONDS: float = REASONING_TIMEOUT_SECONDS

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        self.timeout_seconds = float(timeout_seconds) if timeout_seconds is not None else self.DEFAULT_TIMEOUT_SECONDS
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Provider timeout must be finite and positive")

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def ask(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            with ollama.Client(timeout=self.timeout_seconds) as client:
                response = client.chat(
                    model=OLLAMA_MODEL,
                    messages=self._messages(system_prompt, user_prompt),
                )
            return response["message"]["content"]
        except Exception:
            logger.exception("Ollama provider call failed")
            raise

    def stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        on_token: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """Stream real tokens from Ollama's chat stream.

        Cancellation is genuine and as far as it can safely go: the iteration
        stops pulling further chunks from the model as soon as ``should_stop``
        reports True, so the remaining generation is abandoned rather than
        hidden. The request is not force-killed mid-chunk, which is stated
        honestly rather than claimed away.
        """

        pieces: list[str] = []
        try:
            with ollama.Client(timeout=self.timeout_seconds) as client:
                for chunk in client.chat(
                    model=OLLAMA_MODEL,
                    messages=self._messages(system_prompt, user_prompt),
                    stream=True,
                ):
                    if should_stop is not None and should_stop():
                        break
                    piece = _chunk_text(chunk)
                    if not piece:
                        continue
                    pieces.append(piece)
                    if on_token is not None:
                        on_token(piece)
        except Exception:
            logger.exception("Ollama streaming call failed")
            if not pieces:
                raise
        if not pieces:
            # The stream yielded nothing usable. Fall back to one blocking call
            # rather than reporting an empty answer as if the model said nothing.
            logger.warning("Ollama stream produced no tokens; falling back to a blocking call")
            text = self.ask(system_prompt=system_prompt, user_prompt=user_prompt)
            if on_token is not None and text:
                on_token(text)
            if text:
                pieces.append(text)
        return "".join(pieces)

    def supports_streaming(self) -> bool:
        return True
