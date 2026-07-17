"""LLM (Ollama) client.

ONE responsibility: communicate with Ollama.
"""

from __future__ import annotations

import logging

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

