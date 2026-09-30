"""LLM (Ollama) client.

ONE responsibility: communicate with Ollama.
"""

from __future__ import annotations

import logging
from typing import Callable

from providers.ollama_provider import OllamaProvider


logger = logging.getLogger(__name__)


def ask(*, system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""

    provider = OllamaProvider()
    try:
        return provider.ask(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception:
        logger.exception("LLM call failed")
        raise


def stream(
    *,
    system_prompt: str,
    user_prompt: str,
    on_token: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """Stream a prompt to Ollama, invoking ``on_token`` for each real token.

    Cancellation is cooperative: iteration stops pulling model chunks as soon as
    ``should_stop`` reports True. The provider boundary exposes this so the
    conversation runtime never has to fake progress.
    """

    provider = OllamaProvider()
    try:
        return provider.stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            on_token=on_token,
            should_stop=should_stop,
        )
    except Exception:
        logger.exception("LLM streaming call failed")
        raise



class _ModuleStreamer:
    """The module-level streaming boundary, exposed as a ``stream`` object.

    ``AnswerGenerator.stream`` expects an object exposing ``stream`` (the same
    shape as a provider). Wrapping the module function in a small object keeps
    that contract explicit instead of relying on a bare callable being invoked
    with a call signature it does not implement.
    """

    def stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        on_token: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        return stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            on_token=on_token,
            should_stop=should_stop,
        )

    def supports_streaming(self) -> bool:
        return True


#: The streaming boundary object used by the conversation runtime.
STREAMING_BOUNDARY = _ModuleStreamer()
