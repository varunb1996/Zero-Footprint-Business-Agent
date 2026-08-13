"""Adaptive extract / clarify / flag-uncertain state machine (spec Section 5).

The dialogue policy itself lives entirely in this module, decoupled from any
LLM. `extractor`, `clarifier`, and `owner_responder` are injected callables —
Step 1 wires in hand-scripted fakes (see tests/test_graph_adaptive.py) so the
state machine's routing logic is provable with zero network calls. Step 2
swaps `extractor`/`clarifier` for Groq-backed implementations without
touching this file's control flow.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.shared_nodes import commit, finalize, route_after_select, select_field
from src.agent.state import Clarifier, DialogueState, Extractor, OwnerResponder


def _make_ask(clarifier: Clarifier, owner_responder: OwnerResponder):
    def _ask(state: DialogueState) -> dict:
        turn = state["turn"] + 1
        if turn > state["max_turn_budget"]:
            return {"turn": turn, "done": True}

        field = state["pending_field"]
        assert field is not None

        question = clarifier(field, state["conversation"])
        conversation = state["conversation"] + [
            {"turn": turn, "speaker": "agent", "field": field, "text": question}
        ]

        attempts = state["fields"][field]["clarify_attempts"]
        utterance = owner_responder(field, attempts, question)
        conversation = conversation + [
            {"turn": turn, "speaker": "owner", "field": field, "text": utterance}
        ]

        return {"turn": turn, "conversation": conversation, "last_utterance": utterance}

    return _ask


def _route_after_ask(state: DialogueState) -> str:
    return "finalize" if state["done"] else "classify"


def _make_classify(extractor: Extractor):
    def _classify(state: DialogueState) -> dict:
        field = state["pending_field"]
        assert field is not None
        result = extractor(field, state["last_utterance"] or "", state["turn"])
        return {"last_extraction": result}

    return _classify


def _route_after_classify(state: DialogueState) -> str:
    label = state["last_extraction"]["label"]
    if label == "high":
        return "commit_confirmed"
    if label == "plausible":
        return "commit_inferred" if state["ambiguity_strategy"] == "infer" else "clarify"
    return "leave_open"


def _commit_confirmed(state: DialogueState) -> dict:
    return commit(state, "confirmed")


def _commit_inferred(state: DialogueState) -> dict:
    return commit(state, "inferred")


def _clarify(state: DialogueState) -> dict:
    field = state["pending_field"]
    assert field is not None
    rec = dict(state["fields"][field])
    rec["clarify_attempts"] += 1
    fields = dict(state["fields"])
    fields[field] = rec
    return {"fields": fields}


def _route_after_clarify(state: DialogueState) -> str:
    field = state["pending_field"]
    assert field is not None
    attempts = state["fields"][field]["clarify_attempts"]
    return "flag_uncertain" if attempts >= state["max_clarify_attempts"] else "ask"


def _leave_open(state: DialogueState) -> dict:
    """No usable signal. Per spec 5.3: don't loop on this field immediately —
    deprioritize it behind other open fields, but still cap total attempts
    so it can't be silently skipped forever."""
    field = state["pending_field"]
    assert field is not None
    rec = dict(state["fields"][field])
    rec["clarify_attempts"] += 1
    fields = dict(state["fields"])

    if rec["clarify_attempts"] >= state["max_clarify_attempts"]:
        fields[field] = {
            **rec,
            "status": "uncertain",
            "source_turn": state["turn"],
            "note": "no usable signal after max attempts",
        }
        return {"fields": fields, "pending_field": None}

    fields[field] = rec
    priority = [f for f in state["field_priority"] if f != field] + [field]
    return {"fields": fields, "field_priority": priority, "pending_field": None}


def _flag_uncertain(state: DialogueState) -> dict:
    field = state["pending_field"]
    assert field is not None
    fields = dict(state["fields"])
    fields[field] = {
        **fields[field],
        "status": "uncertain",
        "source_turn": state["turn"],
        "note": "max clarify attempts reached",
    }
    return {"fields": fields, "pending_field": None}


def build_adaptive_graph(
    extractor: Extractor,
    clarifier: Clarifier,
    owner_responder: OwnerResponder,
):
    graph = StateGraph(DialogueState)

    graph.add_node("select_field", select_field)
    graph.add_node("ask", _make_ask(clarifier, owner_responder))
    graph.add_node("classify", _make_classify(extractor))
    graph.add_node("commit_confirmed", _commit_confirmed)
    graph.add_node("commit_inferred", _commit_inferred)
    graph.add_node("clarify", _clarify)
    graph.add_node("leave_open", _leave_open)
    graph.add_node("flag_uncertain", _flag_uncertain)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("select_field")

    graph.add_conditional_edges(
        "select_field", route_after_select, {"ask": "ask", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "ask", _route_after_ask, {"classify": "classify", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "classify",
        _route_after_classify,
        {
            "commit_confirmed": "commit_confirmed",
            "commit_inferred": "commit_inferred",
            "clarify": "clarify",
            "leave_open": "leave_open",
        },
    )
    graph.add_conditional_edges(
        "clarify", _route_after_clarify, {"ask": "ask", "flag_uncertain": "flag_uncertain"}
    )

    graph.add_edge("commit_confirmed", "select_field")
    graph.add_edge("commit_inferred", "select_field")
    graph.add_edge("leave_open", "select_field")
    graph.add_edge("flag_uncertain", "select_field")
    graph.add_edge("finalize", END)

    return graph.compile()
