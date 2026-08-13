"""Groq-backed implementations of the Extractor/Clarifier interfaces.

These match the exact callable signatures defined in src/agent/state.py, so
they drop into build_adaptive_graph() in place of the Step 1 scripted fakes
without any change to the graph's control flow.
"""

from __future__ import annotations

import json

from src.agent.prompts import (
    CLARIFY_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_TOOL_PARAMETERS,
    FIELD_DESCRIPTIONS,
    STRUCTURED_FIELDS,
)
from src.agent.state import Clarifier, ExtractResult, Extractor
from src.llm_client import DEFAULT_MODEL, chat_text, chat_with_tool


def make_groq_extractor(model: str = DEFAULT_MODEL) -> Extractor:
    def extractor(field: str, utterance: str, turn: int) -> ExtractResult:
        args = chat_with_tool(
            system=EXTRACTION_SYSTEM_PROMPT,
            user=(
                f"Field to extract: {field}\n"
                f"Field description: {FIELD_DESCRIPTIONS[field]}\n\n"
                f'Owner said: "{utterance}"'
            ),
            tool_name="record_extraction",
            tool_description="Record the extraction result for this field.",
            parameters=EXTRACTION_TOOL_PARAMETERS,
            model=model,
        )

        value = args.get("value")
        if value is not None and field in STRUCTURED_FIELDS:
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass  # fall back to the raw string rather than crashing the loop

        return ExtractResult(label=args["label"], value=value, note=args.get("note"))

    return extractor


def make_groq_clarifier(model: str = DEFAULT_MODEL) -> Clarifier:
    def clarifier(field: str, conversation: list[dict]) -> str:
        history = "\n".join(
            f"{turn['speaker']}: {turn['text']}" for turn in conversation[-6:]
        )
        user = (
            f"Field: {field}\n"
            f"What we need: {FIELD_DESCRIPTIONS[field]}\n\n"
            f"Conversation so far:\n{history or '(start of interview)'}\n\n"
            "Ask your next question about this field."
        )
        return chat_text(system=CLARIFY_SYSTEM_PROMPT, user=user, model=model)

    return clarifier
