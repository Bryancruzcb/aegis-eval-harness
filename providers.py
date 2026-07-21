"""Provider plumbing shared by the target and the judge.

Centralizes three concerns that were previously duplicated and inconsistent:
  * lazily-built, reused API clients (one per provider, not one per request),
  * a single retry policy for transient failures (429 / 5xx / timeouts), and
  * a clear boundary between *infrastructure* failures and *evaluation* results.

A ``ProviderError`` means "we could not get an answer from the model" — an
infrastructure problem — which the runner records as an ERROR, distinct from a
test the model actually failed.
"""
import logging

from google import genai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

import config

logger = logging.getLogger("AegisEval.Providers")


class ProviderError(Exception):
    """A model call could not be completed (missing key, or failed after retries)."""


# HTTP statuses worth retrying: rate limiting, request timeout, and server-side faults.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Substrings that signal a transient fault when no numeric status is available.
_RETRYABLE_MARKERS = (
    "rate limit", "ratelimit", "resource exhausted", "resource_exhausted",
    "timed out", "timeout", "deadline exceeded", "unavailable",
    "overloaded", "temporarily", "try again",
    "429", "500", "502", "503", "504",
)


def is_retryable_error(exc: BaseException) -> bool:
    """Decide whether an exception represents a transient fault worth retrying.

    A 429 or 5xx is transient; a 400/401/404 (bad request, bad key, wrong model)
    is not — retrying those just burns quota. Falls back to matching known
    transient phrases when the SDK error carries no numeric status code.
    """
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, bool):
            continue
        if isinstance(val, int):
            return val in _RETRYABLE_STATUS
        if isinstance(val, str) and val.isdigit():
            return int(val) in _RETRYABLE_STATUS

    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


# Shared retry policy: exponential backoff, only for transient errors, and
# re-raise the original exception (not tenacity's RetryError) once exhausted.
retryable = retry(
    retry=retry_if_exception(is_retryable_error),
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


# --- Lazily-constructed, reused clients (built once, on first use) ---

_gemini_client = None
_openai_client = None
_ollama_client = None


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not config.GEMINI_API_KEY:
            raise ProviderError("GEMINI_API_KEY is not set. Add it to your .env file.")
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        if not config.OPENAI_API_KEY:
            raise ProviderError("OPENAI_API_KEY is not set. Add it to your .env file.")
        _openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def get_ollama_client() -> AsyncOpenAI:
    global _ollama_client
    if _ollama_client is None:
        # Ollama exposes an OpenAI-compatible endpoint and ignores the API key.
        _ollama_client = AsyncOpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
    return _ollama_client
