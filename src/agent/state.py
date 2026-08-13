"""LangGraph dialogue state for the intake agent (spec Section 5)."""

from __future__ import annotations

from typing import Callable, Literal, Optional, TypedDict

from src.schema import FIELD_PRIORITY

FieldStatus = Literal["open", "confirmed", "inferred", "uncertain"]
ExtractionLabel = Literal["high", "plausible", "none"]

# How to handle "plausible but incomplete/ambiguous" extractions (spec 5.3) —
# the spec calls this out as a strategy worth ablating, so it's a knob, not
# a hardcoded branch.
AmbiguityStrategy = Literal["clarify", "infer"]


class FieldRecord(TypedDict):
    value: Optional[object]
    status: FieldStatus
    source_turn: Optional[int]
    note: Optional[str]
    clarify_attempts: int


class ExtractResult(TypedDict):
    label: ExtractionLabel
    value: Optional[object]
    note: Optional[str]


# Given (field_name, owner_utterance, turn) -> ExtractResult.
# Step 1: a hand-scripted fake. Step 2: swapped for a Groq-backed extractor.
Extractor = Callable[[str, str, int], ExtractResult]

# Given (field_name, conversation_so_far) -> a follow-up question string.
# Step 1: a hand-scripted fake. Step 2: swapped for an LLM-generated question.
Clarifier = Callable[[str, list[dict]], str]

# Given (field_name, attempt_index, question_text) -> the owner's reply for
# that field. In tests this is a scripted lookup; in a live CLI/UI it prints
# question_text and blocks for real input.
OwnerResponder = Callable[[str, int, str], str]


class DialogueState(TypedDict):
    fields: dict[str, FieldRecord]
    field_priority: list[str]
    pending_field: Optional[str]
    turn: int
    conversation: list[dict]
    max_clarify_attempts: int
    max_turn_budget: int
    ambiguity_strategy: AmbiguityStrategy
    done: bool
    # Transient, set by one node and consumed by the next.
    last_utterance: Optional[str]
    last_extraction: Optional[ExtractResult]


def new_dialogue_state(
    *,
    max_clarify_attempts: int = 2,
    max_turn_budget: int = 30,
    ambiguity_strategy: AmbiguityStrategy = "clarify",
) -> DialogueState:
    return DialogueState(
        fields={
            name: FieldRecord(
                value=None,
                status="open",
                source_turn=None,
                note=None,
                clarify_attempts=0,
            )
            for name in FIELD_PRIORITY
        },
        field_priority=list(FIELD_PRIORITY),
        pending_field=None,
        turn=0,
        conversation=[],
        max_clarify_attempts=max_clarify_attempts,
        max_turn_budget=max_turn_budget,
        ambiguity_strategy=ambiguity_strategy,
        done=False,
        last_utterance=None,
        last_extraction=None,
    )
