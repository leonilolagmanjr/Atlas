from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Abstraction for interchangeable LLM reasoning providers."""

    @abstractmethod
    def ask(self, *, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError

