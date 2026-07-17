from __future__ import annotations

import logging

import ollama

from config import OLLAMA_MODEL
from providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Ollama-backed provider implementation."""

    def ask(self, *, system_prompt: str, user_prompt: str) -> str:
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
            logger.exception("Ollama provider call failed")
            raise

