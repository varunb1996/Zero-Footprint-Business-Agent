"""Free-text semantic store (spec Component B) — Chroma, local, free.

Only the free-form fields (products_or_services, policies, free_text_notes)
go here for semantic retrieval; short structured fields (name, hours,
location, contact, category) are looked up directly from SQLite instead —
embedding "9876543210" for semantic search would be pointless.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import chromadb

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chroma"

SEMANTIC_FIELDS = ("products_or_services", "policies", "free_text_notes")


def get_collection(persist_dir: Path = DEFAULT_PERSIST_DIR):
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection("business_kb")


def index_business(business_id: str, fields: dict, collection=None) -> None:
    collection = collection or get_collection()
    ids, documents, metadatas = [], [], []
    for field in SEMANTIC_FIELDS:
        rec = fields.get(field)
        if not rec or rec["status"] == "uncertain" or rec["value"] is None:
            continue
        text = rec["value"] if isinstance(rec["value"], str) else json.dumps(rec["value"])
        ids.append(f"{business_id}:{field}")
        documents.append(text)
        metadatas.append({"business_id": business_id, "field": field, "status": rec["status"]})

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def search(
    business_id: str, query: str, n_results: int = 3, collection=None
) -> list[dict]:
    collection = collection or get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"business_id": business_id},
    )
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    return [{"field": meta["field"], "text": doc} for doc, meta in zip(documents, metadatas)]
