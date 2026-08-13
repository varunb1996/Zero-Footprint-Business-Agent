"""Target schema for the business intake agent (spec Section 4).

The field list below is the single source of truth for what the agent is
trying to fill in. Runtime state uses the dict-based FieldRecord
(src/agent/state.py) rather than a fixed Pydantic model, since real
extracted values are inherently variable-shaped (see the eval cases in
eval/cases/ for examples of the actual value shapes per field).
"""

from __future__ import annotations

# Top-level fields, in priority order (spec Section 5, step 4).
# name/category/hours/location before nice-to-have policy details.
FIELD_PRIORITY: list[str] = [
    "name",
    "category",
    "hours",
    "location",
    "contact",
    "products_or_services",
    "policies",
    "free_text_notes",
]
