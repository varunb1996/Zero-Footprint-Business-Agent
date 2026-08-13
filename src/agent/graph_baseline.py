"""Fixed-form baseline agent — the ablation control for spec Section 6.2's
clarification-efficiency metric.

Asks every schema field exactly once, in a fixed order, using a canned
question — no clarification loop, no re-asking, no reprioritization.
Ambiguous ("plausible") answers are committed as `inferred` immediately
instead of triggering a follow-up; "none" answers are flagged `uncertain`
immediately instead of being deprioritized and retried.

Reuses the same pluggable Extractor and shared node logic as
graph_adaptive.py (see shared_nodes.py), so turn count and field accuracy
differences between the two agents are attributable to dialogue *policy*,
not to a different extractor or a different schema walk order.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.prompts import FIXED_QUESTIONS
from src.agent.shared_nodes import commit, finalize, route_after_select, select_field
from src.agent.state import DialogueState, Extractor, OwnerResponder


def _make_ask(owner_responder: OwnerResponder):
    def _ask(state: DialogueState) -> dict:
        turn = state["turn"] + 1
        if turn > state["max_turn_budget"]:
            return {"turn": turn, "done": True}

        field = state["pending_field"]
        assert field is not None

        question = FIXED_QUESTIONS[field]
        conversation = state["conversation"] + [
            {"turn": turn, "speaker": "agent", "field": field, "text": question}
        ]
        utterance = owner_responder(field, 0, question)
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
        return "commit_inferred"
    return "commit_uncertain"


def _commit_confirmed(state: DialogueState) -> dict:
    return commit(state, "confirmed")


def _commit_inferred(state: DialogueState) -> dict:
    return commit(state, "inferred")


def _commit_uncertain(state: DialogueState) -> dict:
    """Unlike the adaptive graph's leave_open/flag_uncertain split, the
    baseline never retries — a "none" extraction is flagged uncertain on
    the spot."""
    field = state["pending_field"]
    assert field is not None
    result = state["last_extraction"]
    fields = dict(state["fields"])
    fields[field] = {
        **fields[field],
        "value": result.get("value"),
        "status": "uncertain",
        "source_turn": state["turn"],
        "note": result.get("note") or "no usable signal",
    }
    return {"fields": fields, "pending_field": None}


def build_baseline_graph(extractor: Extractor, owner_responder: OwnerResponder):
    graph = StateGraph(DialogueState)

    graph.add_node("select_field", select_field)
    graph.add_node("ask", _make_ask(owner_responder))
    graph.add_node("classify", _make_classify(extractor))
    graph.add_node("commit_confirmed", _commit_confirmed)
    graph.add_node("commit_inferred", _commit_inferred)
    graph.add_node("commit_uncertain", _commit_uncertain)
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
            "commit_uncertain": "commit_uncertain",
        },
    )

    graph.add_edge("commit_confirmed", "select_field")
    graph.add_edge("commit_inferred", "select_field")
    graph.add_edge("commit_uncertain", "select_field")
    graph.add_edge("finalize", END)

    return graph.compile()
