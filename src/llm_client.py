"""Thin LLM provider adapter.

Everything in src/agent/ talks to *this* module, never to the Groq SDK
directly — swapping providers later (e.g. to OpenRouter) means changing
this file only, per the plan's provider-agnostic design.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, Optional, TypeVar

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)

load_dotenv()

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

_client: Optional[Groq] = None

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_MAX_RETRIES = 5
_BASE_DELAY_SECONDS = 2.0
# A suggested wait longer than this (e.g. Groq's daily-token-cap errors,
# which can ask for minutes) means a deliberate decision to burn that much
# time is needed -- surface it instead of silently blocking the caller.
_MAX_SINGLE_WAIT_SECONDS = 60.0
_RETRY_AFTER_RE = re.compile(r"try again in (?:(?P<minutes>\d+)m)?(?P<seconds>[\d.]+)s", re.IGNORECASE)

T = TypeVar("T")


def _suggested_wait_seconds(exc: Exception) -> Optional[float]:
    """Groq embeds an exact suggested wait in rate-limit error messages
    (e.g. "Please try again in 3m32.976s") -- honor it exactly instead of
    guessing with blind exponential backoff."""
    match = _RETRY_AFTER_RE.search(str(exc))
    if not match:
        return None
    minutes = float(match.group("minutes")) if match.group("minutes") else 0.0
    return minutes * 60 + float(match.group("seconds"))


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


def _call_with_retry(fn: Callable[[], T]) -> T:
    """Retry with exponential backoff on rate limits/transient errors —
    a full eval run makes hundreds of calls against the free tier, so
    occasional 429s are expected, not exceptional."""
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except _RETRYABLE as exc:
            suggested = _suggested_wait_seconds(exc) if isinstance(exc, RateLimitError) else None
            if suggested is not None and suggested > _MAX_SINGLE_WAIT_SECONDS:
                raise  # a hard cap (e.g. daily token limit) -- don't block silently
            if attempt == _MAX_RETRIES - 1:
                raise
            delay = suggested if suggested is not None else _BASE_DELAY_SECONDS * (2**attempt)
            print(f"  [retry {attempt + 1}/{_MAX_RETRIES}] {type(exc).__name__}, waiting {delay:.0f}s...")
            time.sleep(delay)
    raise AssertionError("unreachable")


def chat_with_tool(
    *,
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    parameters: dict,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> dict:
    """Force a single tool call and return its parsed JSON arguments.

    Used for extraction, where we need a structured, schema-conforming
    result rather than free text.
    """
    client = get_client()
    response = _call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": parameters,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
    )
    message = response.choices[0].message
    if not message.tool_calls:
        raise RuntimeError(f"Model did not call {tool_name}; got: {message.content!r}")
    return json.loads(message.tool_calls[0].function.arguments)


def chat_text(
    *, system: str, user: str, model: str = DEFAULT_MODEL, temperature: float = 0.3
) -> str:
    """Plain free-text completion, used for generating clarifying questions."""
    client = get_client()
    response = _call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    )
    return (response.choices[0].message.content or "").strip()
