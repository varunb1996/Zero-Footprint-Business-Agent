"""SQLite KB store round-trip tests — synthetic field data, no API calls."""

from src.kb import store_sql

SAMPLE_FIELDS = {
    "name": {"value": "Rajesh Tailors", "status": "confirmed", "source_turn": 1, "note": None},
    "category": {"value": "tailor", "status": "confirmed", "source_turn": 2, "note": None},
    "hours": {
        "value": {"mon_sat": "10:00-20:00", "sun": "closed"},
        "status": "confirmed",
        "source_turn": 3,
        "note": None,
    },
    "contact": {"value": None, "status": "uncertain", "source_turn": 4, "note": "no signal"},
}


def _db_path(tmp_path):
    return tmp_path / "kb.sqlite3"


def test_save_and_load_round_trip(tmp_path):
    db_path = _db_path(tmp_path)
    store_sql.save_business("biz1", "adaptive", SAMPLE_FIELDS, db_path=db_path)

    loaded = store_sql.load_business("biz1", db_path=db_path)

    assert loaded["name"]["value"] == "Rajesh Tailors"
    assert loaded["name"]["status"] == "confirmed"
    assert loaded["hours"]["value"] == {"mon_sat": "10:00-20:00", "sun": "closed"}
    assert loaded["contact"]["status"] == "uncertain"
    assert loaded["contact"]["value"] is None


def test_load_missing_business_returns_none(tmp_path):
    db_path = _db_path(tmp_path)
    store_sql.get_connection(db_path).close()  # ensure schema exists, no rows
    assert store_sql.load_business("nonexistent", db_path=db_path) is None


def test_save_business_overwrites_previous_values(tmp_path):
    db_path = _db_path(tmp_path)
    store_sql.save_business("biz1", "adaptive", SAMPLE_FIELDS, db_path=db_path)

    updated_fields = dict(SAMPLE_FIELDS)
    updated_fields["name"] = {
        "value": "New Name Tailors", "status": "confirmed", "source_turn": 1, "note": None,
    }
    store_sql.save_business("biz1", "adaptive", updated_fields, db_path=db_path)

    loaded = store_sql.load_business("biz1", db_path=db_path)
    assert loaded["name"]["value"] == "New Name Tailors"


def test_different_businesses_do_not_collide(tmp_path):
    db_path = _db_path(tmp_path)
    store_sql.save_business("biz1", "adaptive", SAMPLE_FIELDS, db_path=db_path)
    store_sql.save_business(
        "biz2",
        "baseline",
        {"name": {"value": "Other Shop", "status": "confirmed", "source_turn": 1, "note": None}},
        db_path=db_path,
    )

    biz1 = store_sql.load_business("biz1", db_path=db_path)
    biz2 = store_sql.load_business("biz2", db_path=db_path)

    assert biz1["name"]["value"] == "Rajesh Tailors"
    assert biz2["name"]["value"] == "Other Shop"
    assert "hours" not in biz2
