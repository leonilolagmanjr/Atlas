"""LLM (Ollama) client.

ONE responsibility: communicate with Ollama.
"""

from __future__ import annotations

import logging

import ollama

from config import OLLAMA_MODEL

logger = logging.getLogger(__name__)


def ask(*, system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]
    except Exception:
        logger.exception("Ollama call failed")
        # Keep caller behavior deterministic.
        raise
