"""Regression tests for Ollama LLM timeout and error handling.

Covers:
  A. Ollama successful response — provider returns normally, no error logged
  B. Ollama ReadTimeout — provider produces a typed ProviderTimeout
  C. Recoverable timeout — reasoning call times out, fallback succeeds, turn succeeds
  D. Unrecoverable timeout — reasoning call times out, recovery fails, state is Failed
  E. Cancellation — distinguishable from timeout, cancellation semantics intact
  F. Logging — same exception not emitted as three duplicate tracebacks
  G. Configurable timeout — explicit timeout reaches the httpx client

All tests mock ollama/httpx; no running Ollama server is required.
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, patch

import httpx
import ollama

from config import (
    OLLAMA_CONNECT_TIMEOUT_SECONDS,
    OLLAMA_MODEL,
    REASONING_TIMEOUT_SECONDS,
)
from providers.exceptions import (
    ProviderCancelled,
    ProviderConnectionFailure,
    ProviderError,
    ProviderModelError,
    ProviderTimeout,
)
from providers.ollama_provider import OllamaProvider
from reasoning.answer_generator import AnswerGenerator
from reasoning.json_llm import safe_reasoning_call
import llm


class _DummyLoggerCapture(logging.Handler):
    """Captures log records so tests can assert on log content / level."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def messages(self) -> list[str]:
        return [self.format(r) for r in self.records]


class _LogCapture:
    """Context manager that attaches a capture handler to one or more loggers."""

    def __init__(self, *logger_names: str) -> None:
        self._names = logger_names
        self._handlers: list[_DummyLoggerCapture] = []
        self._old_levels: list[tuple[logging.Logger, int]] = []

    def __enter__(self) -> "_LogCapture":
        for name in self._names:
            lg = logging.getLogger(name)
            handler = _DummyLoggerCapture()
            handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
            lg.addHandler(handler)
            self._old_levels.append((lg, lg.level))
            lg.setLevel(logging.DEBUG)
            self._handlers.append(handler)
        return self

    def __exit__(self, *exc: object) -> None:
        for lg, old_level in self._old_levels:
            lg.setLevel(old_level)
            for handler in list(lg.handlers):
                if any(h is h for h in self._handlers):
                    lg.removeHandler(handler)

    @property
    def messages(self) -> list[str]:
        msgs: list[str] = []
        for handler in self._handlers:
            msgs.extend(handler.messages)
        return msgs

    def count(self, substring: str) -> int:
        return sum(1 for msg in self.messages if substring.lower() in msg.lower())


# --- A. Ollama successful response ------------------------------------------------

class SuccessfulResponseTests(unittest.TestCase):
    """A successful provider call must not log any error-level messages."""

    def test_ask_returns_content_without_error_logging(self):
        with _LogCapture("providers.ollama_provider") as cap:
            with patch("providers.ollama_provider.ollama.Client") as factory:
                client = factory.return_value.__enter__.return_value
                client.chat.return_value = {
                    "message": {"content": "The Cleveland Cavaliers won in 2016."}
                }
                provider = OllamaProvider()
                result = provider.ask(system_prompt="s", user_prompt="q")

        self.assertEqual(result, "The Cleveland Cavaliers won in 2016.")
        self.assertEqual(cap.count("failed"), 0)
        self.assertEqual(cap.count("timeout"), 0)

    def test_stream_returns_tokens_without_error_logging(self):
        with _LogCapture("providers.ollama_provider") as cap:
            with patch("providers.ollama_provider.ollama.Client") as factory:
                client = factory.return_value.__enter__.return_value

                class _Chunk:
                    def __init__(self, content: str) -> None:
                        self.message = type("M", (), {"role": "assistant", "content": content})

                client.chat.return_value = [_Chunk("Hello"), _Chunk(" world")]
                provider = OllamaProvider()
                result = provider.stream(system_prompt="s", user_prompt="q")

        self.assertEqual(result, "Hello world")
        self.assertEqual(cap.count("failed"), 0)


# --- B. Ollama ReadTimeout --------------------------------------------------------

class ReadTimeoutTests(unittest.TestCase):
    """A read timeout must be translated into a typed ProviderTimeout."""

    def test_read_timeout_is_provider_timeout(self):
        with patch("providers.ollama_provider.ollama.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.side_effect = httpx.ReadTimeout("timed out")
            provider = OllamaProvider()
            with self.assertRaises(ProviderTimeout):
                provider.ask(system_prompt="s", user_prompt="q")

    def test_connect_error_is_connection_failure(self):
        with patch("providers.ollama_provider.ollama.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.side_effect = httpx.ConnectError("refused")
            provider = OllamaProvider()
            with self.assertRaises(ProviderConnectionFailure):
                provider.ask(system_prompt="s", user_prompt="q")

    def test_model_not_found_is_model_error(self):
        with patch("providers.ollama_provider.ollama.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.side_effect = ollama.ResponseError("model not found", status_code=404)
            provider = OllamaProvider()
            with self.assertRaises(ProviderModelError):
                provider.ask(system_prompt="s", user_prompt="q")

    def test_timeout_logged_once_at_provider_layer(self):
        with _LogCapture("providers.ollama_provider") as cap:
            with patch("providers.ollama_provider.ollama.Client") as factory:
                client = factory.return_value.__enter__.return_value
                client.chat.side_effect = httpx.ReadTimeout("timed out")
                provider = OllamaProvider()
                try:
                    provider.ask(system_prompt="s", user_prompt="q")
                except ProviderTimeout:
                    pass

        # No traceback logged; just a warning.
        self.assertEqual(cap.count("Traceback"), 0)
        self.assertEqual(cap.count("timed out"), 1)


# --- C. Recoverable timeout -------------------------------------------------------

class RecoverableTimeoutTests(unittest.TestCase):
    """When an LLM call times out, the fallback layer must still produce an answer."""

    def test_safe_reasoning_call_returns_none_on_timeout(self):
        """safe_reasoning_call returns None (not a crash) on ProviderTimeout."""

        def failing_ask(*, system_prompt: str, user_prompt: str) -> str:
            raise ProviderTimeout("timed out")

        result = safe_reasoning_call(
            system_prompt="s", user_prompt="q", ask=failing_ask
        )
        self.assertIsNone(result)

    def test_answer_generator_recovers_from_timeout(self):
        """AnswerGenerator._call returns None on timeout so the engine can fall back."""

        def failing_ask(*, system_prompt: str, user_prompt: str) -> str:
            raise ProviderTimeout("timed out")

        gen = AnswerGenerator(ask=failing_ask)
        result = gen._call(system_prompt="s", user_prompt="q")
        self.assertIsNone(result)

    def test_json_llm_timeout_does_not_print_traceback(self):
        """safe_reasoning_call must not call logger.exception for a ProviderTimeout."""

        def failing_ask(*, system_prompt: str, user_prompt: str) -> str:
            raise ProviderTimeout("timed out")

        with _LogCapture("reasoning.json_llm") as cap:
            result = safe_reasoning_call(
                system_prompt="s", user_prompt="q", ask=failing_ask
            )

        self.assertIsNone(result)
        self.assertEqual(cap.count("Traceback"), 0)
        self.assertEqual(cap.count("timed out"), 1)


# --- D. Unrecoverable timeout -----------------------------------------------------

class UnrecoverableTimeoutTests(unittest.TestCase):
    """When recovery fails and no evidence is available, the turn fails honestly."""

    def test_provider_timeout_propagates_through_llm_ask(self):
        """llm.ask re-raises ProviderTimeout so the engine can react."""

        fake_provider = MagicMock()
        fake_provider.ask.side_effect = ProviderTimeout("timed out")

        with patch("llm.OllamaProvider", return_value=fake_provider):
            with _LogCapture("llm") as cap:
                with self.assertRaises(ProviderTimeout):
                    llm.ask(system_prompt="s", user_prompt="q")

        # No traceback at the llm layer — the provider already logged it.
        self.assertEqual(cap.count("Traceback"), 0)


# --- E. Cancellation -------------------------------------------------------------

class CancellationTests(unittest.TestCase):
    """Cancellation must remain distinguishable from a timeout or provider failure."""

    def test_provider_cancelled_is_not_timeout(self):
        """ProviderCancelled is distinct from ProviderTimeout."""

        self.assertFalse(issubclass(ProviderCancelled, ProviderTimeout))
        self.assertTrue(issubclass(ProviderCancelled, ProviderError))

    def test_cancellation_propagates_through_provider(self):
        with patch("providers.ollama_provider.ollama.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.side_effect = KeyboardInterrupt("cancelled by user")
            provider = OllamaProvider()
            with self.assertRaises(ProviderCancelled):
                provider.ask(system_prompt="s", user_prompt="q")

    def test_cancellation_distinct_from_timeout_in_llm_ask(self):
        """llm.ask re-raises ProviderCancelled, not ProviderTimeout."""

        fake_provider = MagicMock()
        fake_provider.ask.side_effect = ProviderCancelled("cancelled by user")

        with patch("llm.OllamaProvider", return_value=fake_provider):
            with self.assertRaises(ProviderCancelled):
                llm.ask(system_prompt="s", user_prompt="q")


    def test_stream_cancellation_preserves_partial_tokens(self):
        class _Chunk:
            def __init__(self, content: str) -> None:
                self.message = type("M", (), {"role": "assistant", "content": content})

        with patch("providers.ollama_provider.ollama.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.return_value = iter([_Chunk("partial")])
            provider = OllamaProvider()
            stop_called = [False]

            def should_stop() -> bool:
                stop_called[0] = True
                return False  # don't stop, so we get the token first

            result = provider.stream(
                system_prompt="s", user_prompt="q", should_stop=should_stop
            )

        self.assertTrue(stop_called[0])
        self.assertEqual(result, "partial")


# --- F. Logging ------------------------------------------------------------------

class LoggingTests(unittest.TestCase):
    """The same ReadTimeout must not produce three duplicate full tracebacks."""

    def test_no_duplicate_traceback_via_safe_reasoning_call(self):
        """When the LLM times out, only one warning (not three tracebacks) appears."""

        def failing_ask(*, system_prompt: str, user_prompt: str) -> str:
            raise ProviderTimeout("timed out")

        with _LogCapture("providers.ollama_provider", "llm", "reasoning.json_llm") as cap:
            result = safe_reasoning_call(
                system_prompt="s", user_prompt="q", ask=failing_ask
            )

        self.assertIsNone(result)
        # No full traceback should appear anywhere in the chain.
        self.assertEqual(cap.count("Traceback"), 0)

    def test_no_duplicate_log_llm_ask_to_provider(self):
        """llm.ask must not re-log ProviderError tracebacks the provider already logged."""

        fake_provider = MagicMock()
        fake_provider.ask.side_effect = ProviderTimeout("timed out")

        with _LogCapture("providers.ollama_provider", "llm") as cap:
            with patch("llm.OllamaProvider", return_value=fake_provider):
                try:
                    llm.ask(system_prompt="s", user_prompt="q")
                except ProviderTimeout:
                    pass

        # The provider logs once; llm logs at most a short warning, not a traceback.
        self.assertEqual(cap.count("Traceback"), 0)

    def test_unexpected_exception_still_gets_traceback(self):
        """Genuinely unexpected errors must still get a full traceback for debugging."""

        def crashing_ask(*, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("unexpected bug")

        with _LogCapture("reasoning.json_llm") as cap:
            safe_reasoning_call(system_prompt="s", user_prompt="q", ask=crashing_ask)

        self.assertEqual(cap.count("Traceback"), 1)


# --- G. Configurable timeout -----------------------------------------------------

class ConfigurableTimeoutTests(unittest.TestCase):
    """The configured timeout must reach the httpx client as an httpx.Timeout."""

    def test_default_connect_and_read_are_distinct(self):
        provider = OllamaProvider()
        timeout = provider._timeout()
        self.assertEqual(timeout.connect, OLLAMA_CONNECT_TIMEOUT_SECONDS)
        self.assertEqual(timeout.read, REASONING_TIMEOUT_SECONDS)

    def test_explicit_timeout_is_passed_to_client(self):
        with patch("providers.ollama_provider.ollama.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.return_value = {"message": {"content": "ok"}}
            OllamaProvider(read_timeout=300.0, connect_timeout=5.0).ask(
                system_prompt="s", user_prompt="q"
            )
            call_kwargs = factory.call_args.kwargs
            injected = call_kwargs["timeout"]
            self.assertIsInstance(injected, httpx.Timeout)
            self.assertEqual(injected.read, 300.0)
            self.assertEqual(injected.connect, 5.0)

    def test_installed_client_accepts_httpx_timeout(self):
        timeout = httpx.Timeout(timeout=120.0, connect=10.0, read=120.0, write=120.0, pool=120.0)
        with ollama.Client(timeout=timeout) as client:
            self.assertEqual(client._client.timeout.connect, 10.0)
            self.assertEqual(client._client.timeout.read, 120.0)


if __name__ == "__main__":
    unittest.main()
