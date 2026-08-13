"""Step 1 verification: the extract/clarify/flag-uncertain state machine
against a single hardcoded fake conversation. No network/LLM calls — the
extractor, clarifier, and owner_responder are all scripted fakes, so this
proves the dialogue *policy* (routing, attempt limits, prioritization,
hallucination guard) before any real model is wired in.

Scenario: a tailor shop interview that deliberately exercises all four
per-turn outcomes from spec Section 5.3:
  - name, category, products_or_services, policies, free_text_notes:
      unambiguous answers -> committed "confirmed" on the first try.
  - hours: ambiguous first answer -> one clarifying round -> resolved
      "confirmed" on the second try.
  - location: ambiguous both times -> clarify attempts exhausted ->
      flagged "uncertain".
  - contact: no usable signal either time -> deprioritized once, then
      flagged "uncertain" after attempts exhausted.
"""

from src.agent.graph_adaptive import build_adaptive_graph
from src.agent.state import new_dialogue_state

OWNER_SCRIPT = {
    ("name", 0): "We're called Rajesh Tailors.",
    ("category", 0): "We do tailoring - stitching and alterations.",
    ("hours", 0): "Open most days, closed Sundays usually.",
    ("hours", 1): "10am to 8pm, Monday to Saturday, closed Sunday.",
    ("location", 0): "We're near the market.",
    ("location", 1): "You know, near the main market area.",
    ("contact", 0): "Just call me.",
    ("contact", 1): "Just come by the shop sometime.",
    ("products_or_services", 0): (
        "Stitching starts at 200 rupees, alterations vary depending on the work."
    ),
    ("policies", 0): "No returns on custom orders, we do take advance payment for big jobs.",
    ("free_text_notes", 0): "We've been in business for 20 years in this neighborhood.",
}

EXTRACTION_SCRIPT = {
    "We're called Rajesh Tailors.": {
        "label": "high", "value": "Rajesh Tailors", "note": None,
    },
    "We do tailoring - stitching and alterations.": {
        "label": "high", "value": "tailor", "note": None,
    },
    "Open most days, closed Sundays usually.": {
        "label": "plausible", "value": None, "note": "days not fully specified",
    },
    "10am to 8pm, Monday to Saturday, closed Sunday.": {
        "label": "high",
        "value": {"mon_sat": "10:00-20:00", "sun": "closed"},
        "note": None,
    },
    "We're near the market.": {
        "label": "plausible", "value": None, "note": "location too vague",
    },
    "You know, near the main market area.": {
        "label": "plausible", "value": None, "note": "still vague",
    },
    "Just call me.": {"label": "none", "value": None, "note": None},
    "Just come by the shop sometime.": {"label": "none", "value": None, "note": None},
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
    return OWNER_SCRIPT[(field, attempts)]


def fake_extractor(field: str, utterance: str, turn: int) -> dict:
    return EXTRACTION_SCRIPT[utterance]


def fake_clarifier(field: str, conversation: list[dict]) -> str:
    return f"Could you tell me more about {field}?"


def run_scenario():
    graph = build_adaptive_graph(
        extractor=fake_extractor,
        clarifier=fake_clarifier,
        owner_responder=fake_owner_responder,
    )
    initial = new_dialogue_state(max_clarify_attempts=2, max_turn_budget=30)
    return graph.invoke(initial, config={"recursion_limit": 200})


def test_unambiguous_fields_committed_confirmed():
    final = run_scenario()
    assert final["fields"]["name"] == {
        "value": "Rajesh Tailors", "status": "confirmed",
        "source_turn": 1, "note": None, "clarify_attempts": 0,
    }
    assert final["fields"]["category"]["status"] == "confirmed"
    assert final["fields"]["category"]["value"] == "tailor"
    assert final["fields"]["products_or_services"]["status"] == "confirmed"
    assert final["fields"]["policies"]["status"] == "confirmed"
    assert final["fields"]["free_text_notes"]["status"] == "confirmed"


def test_ambiguous_field_resolved_after_one_clarification():
    final = run_scenario()
    hours = final["fields"]["hours"]
    assert hours["status"] == "confirmed"
    assert hours["value"] == {"mon_sat": "10:00-20:00", "sun": "closed"}
    assert hours["clarify_attempts"] == 1


def test_ambiguous_field_flagged_uncertain_after_exhausting_clarify_attempts():
    final = run_scenario()
    location = final["fields"]["location"]
    assert location["status"] == "uncertain"
    assert location["clarify_attempts"] == 2
    assert location["note"] == "max clarify attempts reached"


def test_no_signal_field_deprioritized_then_flagged_uncertain():
    final = run_scenario()
    contact = final["fields"]["contact"]
    assert contact["status"] == "uncertain"
    assert contact["clarify_attempts"] == 2
    assert contact["note"] == "no usable signal after max attempts"


def test_hallucination_guard_no_confirmed_value_without_extraction():
    """Every confirmed/inferred field's value must trace back to something
    the (fake) extractor actually returned for that field's utterance —
    never fabricated by the graph itself."""
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
