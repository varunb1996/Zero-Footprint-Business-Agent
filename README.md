# Conversational Business Intake Agent

An agent that interviews a small business owner in natural conversation
(English/Hindi/Hinglish) and converts it into a structured, queryable
knowledge base — deciding per turn whether to **extract** a value,
**clarify** an ambiguous answer, or **flag it uncertain** rather than
guessing. See [`capstone_spec.md`](../capstone_spec.md) for the full
research spec this implements.

The core contribution is the dialogue policy (`src/agent/`), evaluated
against a naive fixed-form baseline (`graph_baseline.py`) using a
20-case eval set (`eval/cases/`). A thin KB store + RAG layer
(`src/kb/`, `src/whatsapp/`) exists to prove the generated knowledge base
is actually queryable.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt     # macOS/Linux

cp .env.example .env
# then edit .env and set GROQ_API_KEY (free tier: console.groq.com/keys)
```

`.env` variables:

| Variable | Required for | Notes |
|---|---|---|
| `GROQ_API_KEY` | everything except unit tests | free tier, no credit card |
| `LLM_MODEL` | — | defaults to `openai/gpt-oss-120b` |
| `WHATSAPP_*` | live WhatsApp webhook only | not required — see `src/whatsapp/webhook.py` docstring |

## Running things

**Unit + integration tests** (no API calls, all scripted/mocked):

```bash
python -m pytest tests/ -v
```

**Interactive conversation with the real agent** (uses your Groq quota):

```bash
python scripts/chat_cli.py
```

**Evaluation harness** — runs both agents against `eval/cases/` through
the real Groq API and prints Metrics 1–3 (field accuracy, clarification
efficiency, hallucination rate). Metric 4 (downstream usability) is
answered separately via the RAG CLI below.

```bash
python eval/run_eval.py                                    # full set
python eval/run_eval.py --limit 3                           # smoke test
python eval/run_eval.py --cases tailor_01,pharmacy_02        # specific cases
python eval/run_eval.py --resume <run_id>                    # continue after a quota interruption,
                                                               # skipping cases already cached in eval/results/<run_id>/
```

Groq's free tier has a **daily token cap**, not just a per-minute rate
limit — a full 20-case run makes 500+ calls and can hit it. `--resume`
picks up exactly where a run left off using the saved per-case traces in
`eval/results/`, at zero extra API cost for already-completed cases.

**RAG demo (KB + WhatsApp layer)** — no live WhatsApp connection needed:

```bash
# seed the KB from an already-generated eval trace
python scripts/seed_kb_from_trace.py eval/results/<run_id>/tailor_01.json adaptive demo_tailor

# ask it a customer question
python scripts/rag_cli.py demo_tailor "What are your business hours?"
```

**FastAPI server** (health check + WhatsApp webhook route):

```bash
uvicorn src.app:app --reload
```

The webhook logic can be dry-run locally with `curl` against
`GET/POST /webhook` without any Meta account — see `tests/test_webhook.py`
for the exact payload shapes it expects.

## Project layout

```
src/
  schema.py            # target schema field list (FIELD_PRIORITY)
  llm_client.py         # Groq adapter (retry/backoff, swappable provider)
  agent/
    state.py            # DialogueState + Extractor/Clarifier/OwnerResponder interfaces
    shared_nodes.py      # node logic shared by both graphs
    graph_adaptive.py    # extract/clarify/flag-uncertain state machine
    graph_baseline.py    # fixed-form ablation control
    llm_policy.py        # Groq-backed Extractor/Clarifier
    prompts.py           # all LLM prompts + tool schemas
  kb/
    store_sql.py         # SQLite structured store
    store_vector.py       # Chroma semantic store
    loader.py             # writes a completed DialogueState to both
  whatsapp/
    rag.py                # retrieve + answer (Component C)
    webhook.py             # Meta Cloud API sandbox webhook
  app.py                  # FastAPI entrypoint
eval/
  case_schema.py          # eval case format + loader/validator
  cases/                  # 20 hand-authored cases (tailor/pharmacy/restaurant/tuition)
  metrics.py               # precision/recall/hallucination-rate computation
  run_eval.py               # the harness itself
scripts/
  chat_cli.py              # interactive terminal conversation
  rag_cli.py                # ask the RAG layer a question
  seed_kb_from_trace.py     # populate the KB from a saved eval trace
tests/                     # all scripted/mocked, no API calls except where noted
```
