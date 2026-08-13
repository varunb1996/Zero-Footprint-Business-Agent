"""Persist a completed DialogueState's fields into both KB stores in one call."""

from __future__ import annotations

from src.kb import store_sql, store_vector


def save_dialogue_state(business_id: str, agent_type: str, fields: dict) -> None:
    store_sql.save_business(business_id, agent_type, fields)
    store_vector.index_business(business_id, fields)
