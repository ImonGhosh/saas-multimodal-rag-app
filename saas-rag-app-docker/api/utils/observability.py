"""
Langfuse observability helpers.

Goals:
- Safe-by-default (configurable prompt/response capture)
- Works in async code via contextvars
- No-op when disabled/misconfigured
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_current_trace: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "langfuse_current_trace", default=None
)

_LANGFUSE_CLIENT: Any | None = None

# Converts strings like "yes", "1", "true" to a boolean (True or False)
def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

# Converts a string to a float, defaulting to a specified value if the conversion fails
def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default

# Checks if Langfuse tracking is enabled via an environment variable
def langfuse_enabled() -> bool:
    return _parse_bool(os.getenv("LANGFUSE_ENABLED"), default=False)

# Determines the rate at which data should be sampled (i.e., whether to log this event)
def langfuse_sample_rate() -> float:
    rate = _parse_float(os.getenv("LANGFUSE_SAMPLE_RATE"), default=1.0)
    return max(0.0, min(1.0, rate))

# Check if prompts (input text) should be stored or not
def store_prompts() -> bool:
    return _parse_bool(os.getenv("LANGFUSE_STORE_PROMPTS"), default=False)

# Check if responses (output text) should be stored or not
def store_responses() -> bool:
    return _parse_bool(os.getenv("LANGFUSE_STORE_RESPONSES"), default=False)

# Hashes (scrambles) text into a unique ID for privacy
def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

# A list of patterns to look for sensitive data like API keys or tokens (e.g., sk-****)
_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "sk-***REDACTED***"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9\-\._~\+/]+=*"), "Bearer ***REDACTED***"),
]

# Replaces sensitive data in the text with the word REDACTED and limits the length of the text to prevent over-exposure
def redact_text(text: str, max_chars: int = 8000) -> str:
    redacted = text
    for pattern, replacement in _REDACTIONS:
        redacted = pattern.sub(replacement, redacted) # search occurences of the "pattern" in "redacted" and replace with the "replacement" string
    if len(redacted) > max_chars:
        redacted = redacted[:max_chars] + "...[truncated]"
    return redacted


def text_payload(text: str, *, store: bool, max_chars: int = 8000) -> Dict[str, Any]:
    clean = redact_text(text, max_chars=max_chars)
    return {
        "text": clean if store else None,
        "len": len(text),
        "sha256": _hash_text(text),
    }

# checks if the Langfuse client should be initialized (based on environment variables for API keys and server settings). 
# It initializes Langfuse and sets the global _LANGFUSE_CLIENT
def _get_langfuse_client() -> Any | None:
    global _LANGFUSE_CLIENT
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    if not langfuse_enabled():
        _LANGFUSE_CLIENT = None
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")

    if not public_key or not secret_key:
        logger.warning("Langfuse enabled but keys missing; disabling Langfuse client.")
        _LANGFUSE_CLIENT = None
        return None

    try:
        from langfuse import Langfuse  # type: ignore
    except Exception as err:
        logger.warning("Langfuse SDK not importable (%s); disabling Langfuse.", err)
        _LANGFUSE_CLIENT = None
        return None

    try:
        kwargs: Dict[str, Any] = {"public_key": public_key, "secret_key": secret_key}
        if host:
            kwargs["host"] = host
        _LANGFUSE_CLIENT = Langfuse(**kwargs)
        return _LANGFUSE_CLIENT
    except Exception as err:
        logger.warning("Failed to initialize Langfuse client (%s); disabling.", err)
        _LANGFUSE_CLIENT = None
        return None

# Determines whether to sample this event based on the configured sampling rate. This is used to control how much data is sent to Langfuse for observability, allowing for a balance between insight and overhead.
def _should_sample() -> bool:
    rate = langfuse_sample_rate()
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    return random.random() < rate # if rate is between 0 and 1, randomly decide to sample based on the rate (e.g., if rate is 0.1, there's a 10% chance to sample this event)


@dataclass(frozen=True)
class TraceContext:
    trace: Any | None
    token: contextvars.Token[Any | None] | None

    def close(self) -> None:
        if self.token is not None:
            _current_trace.reset(self.token)


def start_trace(
    *,
    name: str,
    session_id: str,
    user_id: str | None = None,
    input: Any | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
) -> TraceContext:
    client = _get_langfuse_client()
    if client is None or not _should_sample():
        return TraceContext(trace=None, token=None)

    try:
        trace = client.trace(
            name=name,
            session_id=session_id,
            user_id=user_id,
            input=input,
            metadata=metadata or {},
            tags=tags or [],
        )
        token = _current_trace.set(trace)
        return TraceContext(trace=trace, token=token)
    except Exception as err:
        logger.debug("Failed to start Langfuse trace: %s", err)
        return TraceContext(trace=None, token=None)


def end_trace(
    trace_ctx: TraceContext,
    *,
    output: Any | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Exception | None = None,
) -> None:
    trace = trace_ctx.trace
    try:
        if trace is None:
            return

        combined_meta: Dict[str, Any] = {}
        if metadata:
            combined_meta.update(metadata)
        if error is not None:
            combined_meta["error_type"] = type(error).__name__
            combined_meta["error_message"] = redact_text(str(error), max_chars=1000)

        # Best-effort update/end; SDK methods vary slightly by version.
        try:
            trace.update(output=output, metadata=combined_meta or None)
        except Exception:
            pass
        try:
            trace.end(output=output, metadata=combined_meta or None)
        except Exception:
            try:
                trace.end()
            except Exception:
                pass
    finally:
        trace_ctx.close()


def get_current_trace() -> Any | None:
    return _current_trace.get()


def start_span(
    name: str,
    *,
    input: Any | None = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any | None:
    trace = get_current_trace()
    if trace is None:
        return None
    try:
        return trace.span(name=name, input=input, metadata=metadata or {})
    except Exception:
        return None


def end_span(
    span: Any | None,
    *,
    output: Any | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Exception | None = None,
) -> None:
    if span is None:
        return
    combined_meta: Dict[str, Any] = {}
    if metadata:
        combined_meta.update(metadata)
    if error is not None:
        combined_meta["error_type"] = type(error).__name__
        combined_meta["error_message"] = redact_text(str(error), max_chars=1000)
    try:
        span.end(output=output, metadata=combined_meta or None)
    except Exception:
        try:
            span.end()
        except Exception:
            pass

