from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class BaseProvider(ABC):
    """Abstraction for interchangeable LLM reasoning providers."""

    @abstractmethod
    def ask(self, *, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        on_token: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """Stream a response, invoking ``on_token`` for each real token.

        The default implementation is honest rather than fake: a provider that
        cannot stream falls back to one blocking call and reports the whole
        answer as a single token. Callers therefore never see synthetic
        per-word progress that did not come from the model.
        """

        text = self.ask(system_prompt=system_prompt, user_prompt=user_prompt)
        if on_token is not None and text:
            on_token(text)
        return text

    def supports_streaming(self) -> bool:  # pragma: no cover - informational
        return False

