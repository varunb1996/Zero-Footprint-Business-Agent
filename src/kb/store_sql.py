"""Structured KB store (spec Component B) — SQLite, deliberately simple.

One row per (business_id, field), carrying the same confidence/source_turn
metadata the dialogue state's FieldRecord already tracks (src/agent/state.py),
so a completed DialogueState maps onto this table with no lossy conversion.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "kb.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS businesses (
    business_id TEXT PRIMARY KEY,
    agent_type TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS business_fields (
    business_id TEXT NOT NULL,
    field TEXT NOT NULL,
    value_json TEXT,
    status TEXT NOT NULL,
    source_turn INTEGER,
    note TEXT,
    PRIMARY KEY (business_id, field),
    FOREIGN KEY (business_id) REFERENCES businesses(business_id)
);
"""


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def save_business(
    business_id: str,
    agent_type: str,
    fields: dict,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO businesses (business_id, agent_type, created_at) "
                "VALUES (?, ?, ?)",
                (business_id, agent_type, datetime.now(timezone.utc).isoformat()),
            )
            for field, rec in fields.items():
                conn.execute(
                    """INSERT OR REPLACE INTO business_fields
                       (business_id, field, value_json, status, source_turn, note)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        business_id,
                        field,
                        json.dumps(rec["value"]),
                        rec["status"],
                        rec["source_turn"],
                        rec["note"],
                    ),
                )
    finally:
        conn.close()


def load_business(business_id: str, db_path: Path = DEFAULT_DB_PATH) -> Optional[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT field, value_json, status, source_turn, note "
            "FROM business_fields WHERE business_id = ?",
            (business_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    return {
        field: {
            "value": json.loads(value_json) if value_json is not None else None,
            "status": status,
            "source_turn": source_turn,
            "note": note,
        }
        for field, value_json, status, source_turn, note in rows
    }
