"""Metrics 1 and 3 from spec Section 6.2: field-level extraction accuracy
and hallucination rate under ambiguity. (Metric 2, clarification efficiency,
is just a turn count comparison and is computed directly in run_eval.py.
Metric 4, downstream usability, is answered by src/whatsapp/rag.py but
isn't wired into this module as a batch-graded computation.)

Value matching uses a normalized-token-overlap heuristic rather than exact
equality or an LLM-judge call — real extraction output will never exactly
match hand-authored ground truth JSON, and an LLM judge would roughly
double the API calls in a run that's already several hundred calls against
a free tier. This is a documented limitation: the heuristic is lenient by
design (see `values_match`), not a precise semantic comparator.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "to", "for", "of", "in",
    "on", "at", "we", "our", "with", "per", "all", "not", "no", "yes",
}


def _tokenize(value: object) -> set[str]:
    if value is None:
        return set()
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def values_match(gt_value: object, agent_value: object, threshold: float = 0.3) -> bool:
    """True if a lenient-but-meaningful fraction of the ground truth's
    tokens show up in the agent's extracted value."""
    if gt_value is None or agent_value is None:
        return gt_value == agent_value

    gt_tokens = _tokenize(gt_value)
    agent_tokens = _tokenize(agent_value)
    if not gt_tokens or not agent_tokens:
        # Too short to tokenize meaningfully (e.g. single characters) —
        # fall back to a direct normalized string comparison rather than
        # treating "both tokenized to nothing" as a match.
        return str(gt_value).strip().lower() == str(agent_value).strip().lower()

    overlap = gt_tokens & agent_tokens
    return (len(overlap) / len(gt_tokens)) >= threshold


@dataclass
class FieldOutcome:
    case_id: str
    field: str
    gt_status: str
    agent_status: str
    outcome: str  # "tp" | "fp_wrong_value" | "fp_hallucination" | "fn" | "tn"


def classify_field(case_id: str, field: str, gt: dict, agent_fields: dict) -> FieldOutcome:
    """Classify one (case, field) prediction against ground truth.

    - tp: both agree there's a value, and it matches.
    - fp_wrong_value: both agree there's a value, but it's wrong.
    - fp_hallucination: ground truth says uncertain, but the agent
      committed a value anyway — this is exactly Metric 3's numerator.
    - fn: ground truth has a value, but the agent flagged uncertain.
    - tn: both agree the field is genuinely uncertain.
    """
    gt_has_value = gt["status"] != "uncertain"
    agent_rec = agent_fields[field]
    agent_has_value = agent_rec["status"] != "uncertain"

    if gt_has_value and agent_has_value:
        outcome = "tp" if values_match(gt["value"], agent_rec["value"]) else "fp_wrong_value"
    elif not gt_has_value and agent_has_value:
        outcome = "fp_hallucination"
    elif gt_has_value and not agent_has_value:
        outcome = "fn"
    else:
        outcome = "tn"

    return FieldOutcome(case_id, field, gt["status"], agent_rec["status"], outcome)


def summarize(outcomes: list[FieldOutcome]) -> dict:
    counts = {k: 0 for k in ("tp", "fp_wrong_value", "fp_hallucination", "fn", "tn")}
    for o in outcomes:
        counts[o.outcome] += 1

    fp_total = counts["fp_wrong_value"] + counts["fp_hallucination"]
    tp = counts["tp"]
    fn = counts["fn"]
    uncertain_total = counts["fp_hallucination"] + counts["tn"]

    precision: Optional[float] = tp / (tp + fp_total) if (tp + fp_total) else None
    recall: Optional[float] = tp / (tp + fn) if (tp + fn) else None
    hallucination_rate: Optional[float] = (
        counts["fp_hallucination"] / uncertain_total if uncertain_total else None
    )

    return {
        "n": len(outcomes),
        **counts,
        "precision": precision,
        "recall": recall,
        "hallucination_rate": hallucination_rate,
    }


def summarize_by_field(outcomes: list[FieldOutcome]) -> dict[str, dict]:
    grouped: dict[str, list[FieldOutcome]] = defaultdict(list)
    for o in outcomes:
        grouped[o.field].append(o)
    return {field: summarize(field_outcomes) for field, field_outcomes in grouped.items()}
