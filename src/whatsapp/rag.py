"""Thin RAG layer (spec Component C): customer question -> retrieve
relevant KB fields/chunks -> LLM answers using only retrieved context.

Intentionally minimal — this exists to prove the KB is queryable and
useful, not to be a polished product. Short structured fields are looked
up directly from SQLite; free-form fields go through Chroma's semantic
search. One business per call, per spec's own non-goal of multi-tenancy.
"""

from __future__ import annotations

from src.kb import store_sql, store_vector
from src.llm_client import chat_text

RAG_SYSTEM_PROMPT = """\
You are answering a customer's question about a small business, using ONLY \
the retrieved information below. If the retrieved context doesn't contain \
the answer, say plainly that you don't have that information -- never \
guess or invent details. Keep answers short and friendly, matching the \
customer's language (English/Hindi/Hinglish).
"""

# Short fields looked up directly -- cheap, no semantic search needed.
DIRECT_FIELDS = ("name", "category", "hours", "location", "contact")


def _format_context(structured: dict, semantic_hits: list[dict]) -> str:
    lines = []
    for field in DIRECT_FIELDS:
        rec = structured.get(field)
        if rec and rec["status"] != "uncertain" and rec["value"] is not None:
            lines.append(f"{field}: {rec['value']}")
    for hit in semantic_hits:
        lines.append(f"{hit['field']}: {hit['text']}")
    return "\n".join(lines) if lines else "(no information available)"


def answer_question(business_id: str, question: str) -> str:
    structured = store_sql.load_business(business_id)
    if not structured:
        return "Sorry, I don't have information about this business yet."

    semantic_hits = store_vector.search(business_id, question)
    context = _format_context(structured, semantic_hits)

    user = f"Business context:\n{context}\n\nCustomer question: {question}"
    return chat_text(system=RAG_SYSTEM_PROMPT, user=user)
