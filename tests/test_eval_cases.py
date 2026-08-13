"""Structural sanity checks for the eval set itself (spec Section 6.1) —
catches malformed/incomplete cases before Step 5's run_eval.py relies on
them. Does not test extraction correctness; that's what run_eval.py is for.
"""

from eval.case_schema import load_cases
from src.schema import FIELD_PRIORITY

REQUIRED_CATEGORIES = {"tailor", "pharmacy", "restaurant", "tuition"}


def test_at_least_fifteen_cases():
    cases = load_cases()
    assert len(cases) >= 15


def test_case_ids_are_unique():
    cases = load_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))


def test_all_required_categories_present():
    cases = load_cases()
    categories = {c["category"] for c in cases}
    assert REQUIRED_CATEGORIES <= categories


def test_every_case_covers_every_schema_field():
    cases = load_cases()
    for case in cases:
        assert set(case["owner_script"].keys()) == set(FIELD_PRIORITY)
        assert set(case["ground_truth"].keys()) == set(FIELD_PRIORITY)


def test_owner_script_has_two_attempts_per_field():
    cases = load_cases()
    for case in cases:
        for field, replies in case["owner_script"].items():
            assert len(replies) == 2, f"{case['id']}.{field} needs 2 scripted replies"


def test_dataset_includes_genuinely_uncertain_examples():
    """The hallucination-guard metric needs cases where the correct answer
    is 'uncertain', not a fabricated confirmed value."""
    cases = load_cases()
    uncertain_count = sum(
        1
        for case in cases
        for truth in case["ground_truth"].values()
        if truth["status"] == "uncertain"
    )
    assert uncertain_count >= 5


def test_dataset_includes_ambiguous_then_resolved_examples():
    """Fields where attempt 0 and attempt 1 differ meaningfully — these are
    what should separate the adaptive agent's turn efficiency from the
    baseline's in Step 5's Metric 2."""
    cases = load_cases()
    resolved_count = sum(
        1
        for case in cases
        for field, replies in case["owner_script"].items()
        if replies[0] != replies[1] and case["ground_truth"][field]["status"] == "confirmed"
    )
    assert resolved_count >= 5


def test_dataset_includes_hinglish_examples():
    cases = load_cases()
    assert any(c["language_style"] == "hinglish" for c in cases)
