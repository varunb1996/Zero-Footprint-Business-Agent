# Project: Conversational Business Intake Agent
### ("Digital Presence in One Conversation")

## 1. What we are building — plain summary

Most small Indian businesses (a tailor, a local pharmacy, an electrician, a tiffin service) have **no structured digital record of their own business information**. It exists only in the owner's head — hours, prices, policies, what they sell. Building them a chatbot or website is useless without that data existing somewhere structured first, and owners will not fill out a form.

We are building an **agent that interviews a small business owner in natural conversation** (text first, voice later — Hindi/English/Hinglish) and converts that messy, incomplete, code-switched conversation into a **clean, structured, queryable knowledge base** — while correctly knowing when to ask a follow-up question versus when to flag a field as "uncertain" rather than guessing.

A thin secondary layer (WhatsApp-based Q&A bot) sits on top of the generated knowledge base purely to *prove it's usable* — that is a demo, not the core research contribution.

**The core research contribution is the intake agent's dialogue policy**: extraction accuracy under ambiguity, adaptive clarification efficiency, and honest uncertainty-flagging (not hallucinating clean answers from messy ones).

We are explicitly **not** building "yet another WhatsApp FAQ bot" — that space is commoditized (Meta shipped native Business AI for WhatsApp in May 2026; WATI, AiSensy, and others already do this well). Our novelty is entirely in the **input side**: turning unstructured human conversation into structured, trustworthy business data.

---

## 2. System Architecture

Three components, in order of research weight (most effort/novelty first):

### Component A — Intake Agent (core contribution, ~60% of effort)
- Conversational agent that interviews the business owner.
- Works from a **target schema** (see Section 4) it is trying to fill in.
- Per turn, it must decide one of three actions:
  1. **Extract** — confidently pull structured value(s) from what was just said, move to next open field.
  2. **Clarify** — the answer was ambiguous/incomplete; ask a targeted follow-up (not a generic "can you elaborate").
  3. **Flag uncertain** — do not fabricate a clean value; mark the field explicitly as unconfirmed/needs-owner-review, and move on rather than looping forever.
- Should track **dialogue state**: which fields are filled, which are uncertain, which are still open, and use that state to decide what to ask next (not a fixed linear script).
- Handles Hindi/English/Hinglish code-switching in later phase.
- Voice input (via Whisper) is a stretch goal — build text-first, add voice once the core dialogue policy works.

### Component B — Knowledge Base Store (~15% of effort, intentionally simple)
- Structured fields (hours, location, category, policies, contact info) → SQLite or plain JSON. Keep this boring and reliable.
- Free-text / product-description fields → embed and store in a vector index (Chroma, local, free) for semantic retrieval later.
- Each field carries metadata: `value`, `confidence` (`confirmed` / `inferred` / `uncertain`), `source_turn` (which conversation turn it came from, for traceability/debugging).

### Component C — Thin WhatsApp Demo Layer (~10% of effort, proof of usability only)
- Simple RAG: customer question → retrieve relevant KB fields/chunks → LLM answers using only retrieved context.
- Deploy via **Meta WhatsApp Cloud API sandbox** (free, unmetered for testing/service replies within the 24-hour window — confirmed suitable for a capstone demo, not for production scale).
- Do not over-invest here. It exists to show the KB is queryable and useful, not to be a polished product.

### Component D — Evaluation Harness (~15% of effort, this is what makes the capstone credible)
- See Section 6. Build this in parallel with Component A, not as an afterthought.

---

## 3. Tech Stack (100% free)

| Layer | Choice | Why |
|---|---|---|
| LLM (dev/local) | **Ollama** running **Qwen3** (8B or 14B) or **GLM-4.7-Flash** | Fully free, self-hosted, no per-token cost, native tool-calling support for agentic loops |
| LLM (if local hardware is too weak) | **Groq free tier** (Llama/Qwen models, ~30 RPM, 100K–500K tokens/day, no credit card) | Free hosted fallback with fast inference |
| Speech-to-text | **Whisper (open-source, local)** | Free, runs offline, good multilingual support including Hindi |
| Agent orchestration | **LangGraph** (You already use this at work — reuse the pattern) | Explicit state machine fits the "extract / clarify / flag-uncertain" per-turn decision structure well |
| Vector store | **Chroma** (local, free) — you already use FAISS/Chroma at work | Simple, no hosting cost |
| Structured store | **SQLite** | Zero setup, free, sufficient for this scale |
| WhatsApp layer | **Meta WhatsApp Cloud API — sandbox/dev mode** | Confirmed free and unmetered for testing |
| Backend | **FastAPI** (your existing stack) | Reuse what you already know |
| Frontend/demo | Simple Streamlit or plain HTML for the intake chat interface; WhatsApp sandbox for the customer-facing demo | Minimal surface area |

**Do not use any paid API by default.** If you hit local hardware limits, fall back to Groq's free tier before considering any paid tier.

---

## 4. Target Schema (v1 — text-only, English first)

Design this as a Pydantic model / structured schema the agent is trying to fill:

```
Business:
  name: str
  category: str                      # e.g. "tailor", "pharmacy", "tiffin service"
  hours: dict                        # per day, with explicit handling for irregular hours
  location: str
  contact: str
  products_or_services: list[Item]   # each with name, price (optional), notes
  policies: dict                     # returns, custom orders, delivery, advance payment, etc.
  free_text_notes: str               # anything that doesn't fit structured fields — goes to vector store

Item:
  name: str
  price: str | None                  # allow "varies", "ask in store" as valid values — do not force a number
  notes: str | None

FieldMeta (attached to every field):
  confidence: "confirmed" | "inferred" | "uncertain"
  source_turn: int
```

Keep v1 schema **deliberately small** (6-8 top-level fields). Expand only after the core dialogue policy is working and evaluated. Scope creep here is the biggest risk to your 2-month timeline.

---

## 5. Dialogue Policy — the actual research problem

This is the heart of the build. The agent must implement a per-turn decision loop, roughly:

1. Parse the owner's latest message.
2. Attempt structured extraction against currently-open schema fields.
3. For each candidate extraction, classify confidence:
   - **High confidence, unambiguous** → commit as `confirmed`, mark field closed.
   - **Plausible but incomplete/ambiguous** (e.g. "open most days, closed Sundays usually") → either commit as `inferred` with a note, OR trigger one targeted clarifying question — pick based on whether the ambiguity materially affects usefulness downstream (test both strategies, this is a good ablation).
   - **No usable signal** → leave field open, do not ask about it yet if higher-priority fields remain unfilled.
4. Decide next question based on: which fields are still open, prioritized by importance (name/category/hours/location before nice-to-have policy details).
5. Stop condition: all high-priority fields are `confirmed` or `inferred`, OR a max-turn budget is hit — remaining fields get flagged `uncertain` rather than looping indefinitely. **Never let the agent stall the owner indefinitely chasing one field.**

Explicitly build in a **hallucination guard**: the agent must never write a `confirmed` value it did not actually extract from owner text. This is testable and citable in your write-up.

---

## 6. Evaluation Methodology — build this alongside the agent, not after

This is what separates a demo from a capstone with real research weight.

### 6.1 Test set construction
- Build **15–20 simulated shop-owner interviews** yourself (role-play both sides, deliberately messy/incomplete/Hinglish where relevant).
- For each, write a **ground-truth structured answer** (the correct schema fill, including which fields *should* legitimately be marked uncertain because the simulated owner never gave a clear answer).

### 6.2 Metrics
1. **Field-level extraction accuracy** — precision/recall against ground truth, per field type.
2. **Clarification efficiency** — number of turns to reach a complete-enough KB. Compare your adaptive agent against a **naive fixed-form baseline** (asks every question regardless of context) as your ablation/control. This is your cleanest, most defensible experiment.
3. **Hallucination rate under ambiguity** — % of deliberately ambiguous test inputs where the agent wrongly committed a `confirmed` value instead of correctly flagging `uncertain`/asking a follow-up. Report this explicitly — it's your responsible-AI angle.
4. **Downstream usability** — a small set (~10-15) of customer-style questions run against each generated KB through the WhatsApp RAG demo layer, graded correct/incorrect/unanswerable.

Log everything (which strategy, which turn, which decision) — this traceability is what makes the results defensible and also gives you good qualitative examples for your report/demo.

---

## 7. Explicit Non-Goals (say this clearly in your proposal to preempt scope creep)

- Not building a general-purpose WhatsApp chatbot platform (that market is commoditized — Meta Business AI, WATI, AiSensy already do this).
- Not building multi-business, multi-tenant infrastructure — one business profile per demo run is enough.
- Not doing voice input in v1 — text-first, add Whisper only if time permits after core evaluation is solid.
- Not fine-tuning any model — pure prompting/agentic orchestration, per your own stated preference.
- Not handling every possible business category exhaustively — pick 3-4 representative categories (e.g., tailor, pharmacy, small restaurant, tuition/coaching class) for your test set and be upfront that generalization beyond these is future work.

---

## 8. Build Order (suggested, matches the 8-week plan already discussed)

1. Schema + LangGraph state machine skeleton (extract/clarify/flag-uncertain loop), text-only, English, single hardcoded test conversation.
2. Real LLM integration (Ollama/Qwen local) driving the extraction + decision logic.
3. Build the fixed-form baseline agent (for your ablation comparison) — reuse most of the same code, just remove the adaptive logic.
4. Build the 15-20 item evaluation set + ground truth.
5. Run both agents against the eval set, compute the four metrics above, iterate on failure cases.
6. Add Hinglish/Hindi handling if time and eval results justify it.
7. Build KB store (SQLite + Chroma) properly, wire up the thin WhatsApp RAG demo layer.
8. Polish, write up, prepare demo with 1-2 real (or realistic simulated) end-to-end examples.

---

## 9. What to tell Claude Code when starting the build

When you start building, give Claude Code:
- This full spec document.
- Confirmation of your exact local setup (Ollama installed? which model pulled? Python version?).
- Ask it to start with **Section 8, step 1** only — the LangGraph state machine skeleton with a single hardcoded fake conversation, before touching any real LLM calls. Get the state machine logic right and testable first, then swap in real model calls.
