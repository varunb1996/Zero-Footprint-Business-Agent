"""Unit tests for eval/metrics.py's classification logic — synthetic data
only, no API calls. Verifying this before the live eval run matters: a bug
here would silently skew every reported number from Step 5's real run.
"""

from eval.metrics import classify_field, summarize, values_match


def _field(status: str, value=None) -> dict:
    return {"status": status, "value": value}


def test_values_match_lenient_overlap():
    assert values_match("Rajesh Tailors", "Rajesh Tailors")
    assert values_match("Near the market, Sector 12", "Sector 12 market area")
    assert not values_match("9876543210", "9123456780")


def test_values_match_none_only_matches_none():
    assert values_match(None, None)
    assert not values_match(None, "something")
    assert not values_match("something", None)


def test_classify_true_positive():
    gt = _field("confirmed", "Rajesh Tailors")
    agent_fields = {"name": _field("confirmed", "Rajesh Tailors")}
    outcome = classify_field("c1", "name", gt, agent_fields)
    assert outcome.outcome == "tp"


def test_classify_wrong_value():
    gt = _field("confirmed", "Rajesh Tailors")
    agent_fields = {"name": _field("confirmed", "Totally Different Shop")}
    outcome = classify_field("c1", "name", gt, agent_fields)
    assert outcome.outcome == "fp_wrong_value"


def test_classify_hallucination():
    """Ground truth says uncertain, but the agent committed a value anyway."""
    gt = _field("uncertain", None)
    agent_fields = {"contact": _field("confirmed", "9876543210")}
    outcome = classify_field("c1", "contact", gt, agent_fields)
    assert outcome.outcome == "fp_hallucination"


def test_classify_false_negative():
    gt = _field("confirmed", "Rajesh Tailors")
    agent_fields = {"name": _field("uncertain", None)}
    outcome = classify_field("c1", "name", gt, agent_fields)
    assert outcome.outcome == "fn"


def test_classify_true_negative():
    gt = _field("uncertain", None)
    agent_fields = {"contact": _field("uncertain", None)}
    outcome = classify_field("c1", "contact", gt, agent_fields)
    assert outcome.outcome == "tn"


def test_summarize_precision_recall_and_hallucination_rate():
    gt_confirmed = _field("confirmed", "X")
    gt_uncertain = _field("uncertain", None)

    outcomes = [
        classify_field("c1", "f1", gt_confirmed, {"f1": _field("confirmed", "X")}),  # tp
        classify_field("c2", "f1", gt_confirmed, {"f1": _field("confirmed", "Y")}),  # fp_wrong_value
        classify_field("c3", "f1", gt_uncertain, {"f1": _field("confirmed", "Z")}),  # fp_hallucination
        classify_field("c4", "f1", gt_confirmed, {"f1": _field("uncertain", None)}),  # fn
        classify_field("c5", "f1", gt_uncertain, {"f1": _field("uncertain", None)}),  # tn
    ]
    s = summarize(outcomes)

    assert s["tp"] == 1
    assert s["fp_wrong_value"] == 1
    assert s["fp_hallucination"] == 1
    assert s["fn"] == 1
    assert s["tn"] == 1
    assert s["precision"] == 1 / 3  # tp / (tp + fp_wrong + fp_hallucination)
    assert s["recall"] == 1 / 2  # tp / (tp + fn)
    assert s["hallucination_rate"] == 1 / 2  # fp_hallucination / (fp_hallucination + tn)
