from __future__ import annotations

import logging
import math
from typing import Callable

import httpx
import ollama

from config import (
    OLLAMA_CONNECT_TIMEOUT_SECONDS,
    OLLAMA_MODEL,
    OLLAMA_READ_TIMEOUT_SECONDS,
    REASONING_TIMEOUT_SECONDS,
)
from providers.base_provider import BaseProvider
from providers.exceptions import (
    ProviderCancelled,
    ProviderConnectionFailure,
    ProviderError,
    ProviderModelError,
    ProviderTimeout,
)

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


def _make_timeout(connect: float, read: float) -> httpx.Timeout:
    """Build an httpx timeout distinguishing connect from read/generation."""

    return httpx.Timeout(
        timeout=read,
        connect=connect,
        read=read,
        write=read,
        pool=read,
    )


def _classify_ollama_exception(exc: BaseException) -> type[Exception]:
    """Map an httpx/ollama exception to the canonical provider error type.

    Timeouts are distinguished from connection failures and unexpected errors so
    that higher layers can recover from a timed-out model call without
    re-printing the traceback or treating it as a permanent failure.
    """

    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        return ProviderTimeout
    if isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError, httpx.NetworkError)):
        return ProviderConnectionFailure
    if isinstance(exc, httpx.HTTPStatusError):
        return ProviderModelError
    if isinstance(exc, ollama.ResponseError):
        return ProviderModelError
    return type(exc)


class OllamaProvider(BaseProvider):
    """Ollama-backed provider implementation."""

    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
    ) -> None:
        # When a single legacy timeout is passed, use it for read and fall back
        # to the connect default.
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(read_timeout) if read_timeout is not None
            else REASONING_TIMEOUT_SECONDS
        )
        self.connect_timeout = float(connect_timeout) if connect_timeout is not None else OLLAMA_CONNECT_TIMEOUT_SECONDS
        self.read_timeout = float(read_timeout) if read_timeout is not None else self.timeout_seconds
        for value in (self.timeout_seconds, self.connect_timeout, self.read_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Provider timeout must be finite and positive")

    def _timeout(self) -> httpx.Timeout:
        return _make_timeout(self.connect_timeout, self.read_timeout)

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def ask(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            with ollama.Client(timeout=self._timeout()) as client:
                response = client.chat(
                    model=OLLAMA_MODEL,
                    messages=self._messages(system_prompt, user_prompt),
                )
            return response["message"]["content"]
        except ProviderCancelled:
            raise
        except KeyboardInterrupt:
            raise ProviderCancelled("cancelled by user") from None
        except Exception as exc:
            self._log_and_translate(exc)

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
            with ollama.Client(timeout=self._timeout()) as client:
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
        except ProviderCancelled:
            if not pieces:
                raise
            return "".join(pieces)
        except KeyboardInterrupt:
            raise ProviderCancelled("cancelled by user") from None
        except Exception as exc:
            self._log_and_translate(exc, partial="".join(pieces))
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

    # -- internals ----------------------------------------------------------------

    def _log_and_translate(self, exc: BaseException, *, partial: str = "") -> None:
        """Log a provider-level failure once and translate to a typed exception.

        Only the provider layer logs the exception detail; higher layers
        receive a typed :class:`ProviderError` subclass and must not re-print
        the same traceback.
        """

        if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
            logger.warning(
                "Ollama request timed out (read timeout %.0fs); "
                "the model is alive but the response exceeded the deadline",
                self.read_timeout,
            )
            raise ProviderTimeout(str(exc)) from exc
        if isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError, httpx.NetworkError)):
            logger.warning("Ollama connection failed: %s", exc)
            raise ProviderConnectionFailure(str(exc)) from exc
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code if exc.response is not None else 0
            logger.warning("Ollama returned HTTP %s", status)
            raise ProviderModelError(str(exc)) from exc
        if isinstance(exc, ollama.ResponseError):
            status = getattr(exc, "status_code", None)
            if status == 404:
                logger.warning("Ollama model not found: %s", exc)
            else:
                logger.warning("Ollama model error: %s", exc)
            raise ProviderModelError(str(exc)) from exc
        logger.warning("Ollama provider call failed: %s", exc)
        raise exc
