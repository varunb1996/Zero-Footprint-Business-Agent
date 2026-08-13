"""Chroma semantic store tests — local embeddings only, no LLM API calls."""

import chromadb

from src.kb import store_vector

SAMPLE_FIELDS = {
    "products_or_services": {
        "value": [{"name": "stitching", "price": "200"}, {"name": "alterations", "price": "varies"}],
        "status": "confirmed",
        "source_turn": 5,
        "note": None,
    },
    "policies": {
        "value": {"returns": "no returns on custom orders", "advance_payment": "50% for big orders"},
        "status": "confirmed",
        "source_turn": 6,
        "note": None,
    },
    "free_text_notes": {
        "value": "In business over 20 years, also does embroidery on request",
        "status": "confirmed",
        "source_turn": 7,
        "note": None,
    },
    "contact": {"value": "9876543210", "status": "confirmed", "source_turn": 4, "note": None},
    "policies_uncertain_example": {"value": None, "status": "uncertain", "source_turn": 8, "note": None},
}


def _fresh_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return client.get_or_create_collection("test_business_kb")


def test_index_only_stores_semantic_fields(tmp_path):
    collection = _fresh_collection(tmp_path)
    store_vector.index_business("biz1", SAMPLE_FIELDS, collection=collection)

    # 3 semantic fields indexed; contact and the uncertain field are skipped.
    assert collection.count() == 3


def test_search_returns_relevant_field(tmp_path):
    collection = _fresh_collection(tmp_path)
    store_vector.index_business("biz1", SAMPLE_FIELDS, collection=collection)

    hits = store_vector.search("biz1", "do you accept returns on custom orders?", collection=collection)

    assert len(hits) > 0
    assert any(hit["field"] == "policies" for hit in hits)


def test_search_scoped_to_business_id(tmp_path):
    collection = _fresh_collection(tmp_path)
    store_vector.index_business("biz1", SAMPLE_FIELDS, collection=collection)
    store_vector.index_business(
        "biz2",
        {"free_text_notes": {"value": "Completely unrelated shop", "status": "confirmed", "source_turn": 1, "note": None}},
        collection=collection,
    )

    hits = store_vector.search("biz2", "embroidery", collection=collection)

    assert all(h["text"] != SAMPLE_FIELDS["free_text_notes"]["value"] for h in hits)


def test_search_on_empty_collection_returns_empty_list(tmp_path):
    collection = _fresh_collection(tmp_path)
    assert store_vector.search("biz1", "anything", collection=collection) == []
