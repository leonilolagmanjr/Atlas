from __future__ import annotations

import logging
import math

import ollama

from config import OLLAMA_MODEL, REASONING_TIMEOUT_SECONDS
from providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Ollama-backed provider implementation."""

    DEFAULT_TIMEOUT_SECONDS: float = REASONING_TIMEOUT_SECONDS

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        self.timeout_seconds = float(timeout_seconds) if timeout_seconds is not None else self.DEFAULT_TIMEOUT_SECONDS
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Provider timeout must be finite and positive")

    def ask(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            with ollama.Client(timeout=self.timeout_seconds) as client:
                response = client.chat(
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
