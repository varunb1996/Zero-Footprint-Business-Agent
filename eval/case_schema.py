"""Schema and loader for the evaluation set (spec Section 6.1).

Each case is one JSON file under eval/cases/ combining the simulated
owner's script with its ground-truth schema fill, so a case's script and
its expected answer can never drift apart into mismatched files.

owner_script[field] is always a 2-element list: [reply_if_asked_once,
reply_if_asked_again]. Fields designed to resolve after one clarification
have a vague first reply and a clear second reply; fields designed to stay
genuinely unclear repeat a vague/non-answer in both slots; fields with an
unambiguous answer just repeat it in both slots (attempt 2 is never
consumed for these in practice, but must exist since a real LLM may rate a
clear-sounding answer as merely "plausible" and ask again anyway).

ground_truth[field] is the answer a *correctly behaving* agent should
reach — not necessarily what every agent will produce. The baseline is
expected to under-perform on ambiguous-resolved fields by design, since it
never gets a second attempt; that gap is exactly what Metric 2
(clarification efficiency) measures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TypedDict

from src.schema import FIELD_PRIORITY

CASES_DIR = Path(__file__).parent / "cases"

VALID_STATUSES = {"confirmed", "inferred", "uncertain"}


class FieldTruth(TypedDict):
    status: str
    value: Optional[object]


class EvalCase(TypedDict):
    id: str
    category: str
    language_style: str
    description: str
    owner_script: dict[str, list[str]]
    ground_truth: dict[str, FieldTruth]


def validate_case(case: dict) -> list[str]:
    """Return validation error strings; empty list means the case is well-formed."""
    errors: list[str] = []
    for key in ("id", "category", "owner_script", "ground_truth"):
        if key not in case:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return errors

    case_id = case["id"]
    for field in FIELD_PRIORITY:
        script = case["owner_script"].get(field)
        if not script or not isinstance(script, list) or len(script) < 2:
            errors.append(
                f"{case_id}: owner_script['{field}'] must be a list of 2 replies"
            )

        truth = case["ground_truth"].get(field)
        if truth is None:
            errors.append(f"{case_id}: ground_truth missing field '{field}'")
            continue
        if truth["status"] not in VALID_STATUSES:
            errors.append(f"{case_id}: invalid status '{truth['status']}' for '{field}'")
        if truth["status"] != "uncertain" and truth.get("value") is None:
            errors.append(
                f"{case_id}: field '{field}' status={truth['status']} but value is null"
            )
    return errors


def load_cases(cases_dir: Path = CASES_DIR) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for path in sorted(cases_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_case(case)
        if errors:
            raise ValueError(f"Invalid eval case {path.name}: " + "; ".join(errors))
        cases.append(case)
    return cases
