"""Fixed-form baseline verification, mirroring test_graph_adaptive.py's
style: scripted fakes, zero network calls.

Uses the same tailor-shop scenario and the same ambiguous/no-signal
utterances as the adaptive test, so the two test suites are directly
comparable. The point of this suite is to prove the baseline's defining
property: it never clarifies or retries — every field gets exactly one
question and is committed (confirmed/inferred/uncertain) on the spot, so
total turns are always exactly len(FIELD_PRIORITY) regardless of ambiguity.
"""

from src.agent.graph_baseline import build_baseline_graph
from src.agent.state import new_dialogue_state
from src.schema import FIELD_PRIORITY

OWNER_SCRIPT = {
    "name": "We're called Rajesh Tailors.",
    "category": "We do tailoring - stitching and alterations.",
    "hours": "Open most days, closed Sundays usually.",
    "location": "We're near the market.",
    "contact": "Just call me.",
    "products_or_services": (
        "Stitching starts at 200 rupees, alterations vary depending on the work."
    ),
    "policies": "No returns on custom orders, we do take advance payment for big jobs.",
    "free_text_notes": "We've been in business for 20 years in this neighborhood.",
}

EXTRACTION_SCRIPT = {
    "We're called Rajesh Tailors.": {
        "label": "high", "value": "Rajesh Tailors", "note": None,
    },
    "We do tailoring - stitching and alterations.": {
        "label": "high", "value": "tailor", "note": None,
    },
    "Open most days, closed Sundays usually.": {
        "label": "plausible", "value": "open most days, closed Sundays", "note": "days not fully specified",
    },
    "We're near the market.": {
        "label": "plausible", "value": "near the market", "note": "location too vague",
    },
    "Just call me.": {"label": "none", "value": None, "note": None},
    "Stitching starts at 200 rupees, alterations vary depending on the work.": {
        "label": "high",
        "value": [
            {"name": "stitching", "price": "starts at 200", "notes": None},
            {"name": "alterations", "price": "varies", "notes": None},
        ],
        "note": None,
    },
    "No returns on custom orders, we do take advance payment for big jobs.": {
        "label": "high",
        "value": {"returns": "no returns on custom orders", "advance_payment": "required for big jobs"},
        "note": None,
    },
    "We've been in business for 20 years in this neighborhood.": {
        "label": "high",
        "value": "In business 20 years in this neighborhood.",
        "note": None,
    },
}


def fake_owner_responder(field: str, attempts: int, question: str) -> str:
    return OWNER_SCRIPT[field]


def fake_extractor(field: str, utterance: str, turn: int) -> dict:
    return EXTRACTION_SCRIPT[utterance]


def run_scenario():
    graph = build_baseline_graph(extractor=fake_extractor, owner_responder=fake_owner_responder)
    initial = new_dialogue_state(max_clarify_attempts=2, max_turn_budget=30)
    return graph.invoke(initial, config={"recursion_limit": 200})


def test_unambiguous_fields_committed_confirmed():
    final = run_scenario()
    assert final["fields"]["name"]["status"] == "confirmed"
    assert final["fields"]["name"]["value"] == "Rajesh Tailors"
    assert final["fields"]["category"]["status"] == "confirmed"
    assert final["fields"]["products_or_services"]["status"] == "confirmed"
    assert final["fields"]["policies"]["status"] == "confirmed"
    assert final["fields"]["free_text_notes"]["status"] == "confirmed"


def test_ambiguous_fields_committed_inferred_without_clarification():
    """The defining baseline behavior: no follow-up question, ever."""
    final = run_scenario()
    hours = final["fields"]["hours"]
    assert hours["status"] == "inferred"
    assert hours["clarify_attempts"] == 0

    location = final["fields"]["location"]
    assert location["status"] == "inferred"
    assert location["clarify_attempts"] == 0


def test_no_signal_field_flagged_uncertain_immediately():
    final = run_scenario()
    contact = final["fields"]["contact"]
    assert contact["status"] == "uncertain"
    assert contact["clarify_attempts"] == 0
    assert contact["note"] == "no usable signal"


def test_turn_count_is_always_exactly_one_per_field():
    """Fixed-form: total turns never vary with ambiguity, unlike the
    adaptive agent's variable turn count on the same scenario."""
    final = run_scenario()
    assert final["turn"] == len(FIELD_PRIORITY)


def test_hallucination_guard_no_confirmed_value_without_extraction():
    final = run_scenario()
    for field, rec in final["fields"].items():
        if rec["status"] in ("confirmed", "inferred"):
            assert rec["source_turn"] is not None
            assert rec["value"] is not None


def test_dialogue_completes_without_exhausting_turn_budget():
    final = run_scenario()
    assert final["done"] is True
    assert all(rec["status"] != "open" for rec in final["fields"].values())
    assert final["turn"] < final["max_turn_budget"]
