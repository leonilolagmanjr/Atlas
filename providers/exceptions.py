"""Exception types for LLM provider failures.

These exceptions let higher layers distinguish between a recoverable
timeout (the model is alive but slow), a connection failure (Ollama is
not running), a missing model, and an unexpected bug — without each
layer re-printing the same full traceback.
"""


class ProviderError(Exception):
    """Base class for all provider-layer failures."""


class ProviderTimeout(ProviderError):
    """The provider did not respond within the configured deadline."""


class ProviderConnectionFailure(ProviderError):
    """Could not establish a connection to the provider (e.g. Ollama not running)."""


class ProviderModelError(ProviderError):
    """The provider responded with a model-level error (model not found, bad request, etc.)."""


class ProviderCancelled(ProviderError):
    """The request was cancelled by the caller before completion."""
