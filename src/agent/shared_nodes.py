"""Node logic shared between the adaptive graph and the fixed-form baseline.

Keeping this shared means the two agents differ only in dialogue *policy*
(how they route after an ambiguous/no-signal extraction), which is exactly
what the spec's Metric 2 ablation needs to isolate.
"""

from __future__ import annotations

from src.agent.state import DialogueState


def select_field(state: DialogueState) -> dict:
    if state["pending_field"] is not None:
        return {}
    for field in state["field_priority"]:
        if state["fields"][field]["status"] == "open":
            return {"pending_field": field}
    return {"done": True}


def route_after_select(state: DialogueState) -> str:
    return "finalize" if state["done"] else "ask"


def commit(state: DialogueState, status: str) -> dict:
    field = state["pending_field"]
    assert field is not None
    result = state["last_extraction"]
    fields = dict(state["fields"])
    fields[field] = {
        **fields[field],
        "value": result["value"],
        "status": status,
        "source_turn": state["turn"],
        "note": result.get("note"),
    }
    return {"fields": fields, "pending_field": None}


def finalize(state: DialogueState) -> dict:
    """Stop condition safety net: any field still open when we get here
    (turn budget exhausted) is flagged uncertain rather than left dangling."""
    fields = dict(state["fields"])
    for name, rec in fields.items():
        if rec["status"] == "open":
            fields[name] = {**rec, "status": "uncertain", "note": "turn budget exhausted"}
    return {"fields": fields, "done": True}
