"""Domain prompts and tool schema for the intake agent's LLM calls."""

FIELD_DESCRIPTIONS: dict[str, str] = {
    "name": "The business's name.",
    "category": (
        "The type of business, e.g. tailor, pharmacy, tiffin service, small "
        "restaurant, tuition/coaching class."
    ),
    "hours": (
        "Operating hours, ideally per day of the week, with explicit handling "
        "for irregular or closed days."
    ),
    "location": (
        "Where the business is physically located — an address or a "
        "description clear enough for a customer to find it."
    ),
    "contact": "A phone number or other concrete way for customers to reach the business.",
    "products_or_services": (
        "The products or services offered. Each item can have a name, an "
        "optional price (vague answers like 'varies' or 'ask in store' are "
        "valid — never force a number), and optional notes."
    ),
    "policies": "Business policies: returns, custom orders, delivery, advance payment, etc.",
    "free_text_notes": "Anything relevant about the business that doesn't fit the other fields.",
}

# Fields whose extracted value should be a JSON object/array rather than a
# plain string.
STRUCTURED_FIELDS = {"hours", "products_or_services", "policies"}

EXTRACTION_SYSTEM_PROMPT = """\
You are the extraction module of a business-intake agent that interviews \
small Indian business owners (tailor, pharmacy, tiffin service, etc.) in \
natural conversation, in English, Hindi, or Hinglish.

Given one target field and the owner's latest message, decide how well the \
message answers that field, and extract a value if possible.

Rules:
- "high": the owner gave a confident, unambiguous answer for this exact field.
- "plausible": there's a usable answer but it's incomplete, vague, or \
  ambiguous (e.g. "open most days, closed Sundays usually" for hours).
- "none": the message gives no usable signal for this field at all. This \
  INCLUDES cases where the owner explicitly says they don't have an answer \
  yet, haven't decided, or have nothing to add (e.g. "hume abhi nahi pata", \
  "no fixed policy yet", "nothing much", "we'll figure that out later"). \
  Those statements are not themselves a value for the field — they mean \
  there is no usable signal, so label them "none" with value null, even \
  though the owner clearly said something.
- NEVER invent or guess a value the owner did not actually say. If you are \
  not confident the value came from the owner's words, do not use "high".
- If label is "none", value must be null.
- Keep notes short (under 15 words) and only explain non-"high" labels.
"""

EXTRACTION_TOOL_PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["high", "plausible", "none"],
            "description": "Confidence classification for this extraction.",
        },
        "value": {
            "type": ["string", "null"],
            "description": (
                "The extracted value as plain text for simple fields. For "
                "structured fields (hours, products_or_services, policies), "
                "return a JSON-encoded string (object or array). Null if "
                "label is 'none'."
            ),
        },
        "note": {
            "type": ["string", "null"],
            "description": "Short note explaining a 'plausible' or 'none' label.",
        },
    },
    "required": ["label", "value", "note"],
}

# Fixed, non-adaptive questions for the naive baseline (spec Section 6.2) —
# asked verbatim, once each, regardless of how the previous answer went.
FIXED_QUESTIONS: dict[str, str] = {
    "name": "What is the name of your business?",
    "category": "What type of business is this (e.g. tailor, pharmacy, restaurant)?",
    "hours": "What are your business hours, including any days you're closed?",
    "location": "Where is your business located?",
    "contact": "What's the best phone number or contact for customers to reach you?",
    "products_or_services": "What products or services do you offer, and what are the prices?",
    "policies": "What are your policies on returns, custom orders, delivery, or advance payment?",
    "free_text_notes": "Is there anything else about your business you'd like to add?",
}

CLARIFY_SYSTEM_PROMPT = """\
You are interviewing a small Indian business owner to build a structured \
profile of their business. Ask exactly ONE short, natural, targeted \
question about the given field — never a generic "can you elaborate?". If \
the conversation shows a prior vague answer for this field, ask something \
specific that would resolve the ambiguity. Match the language of the \
OWNER'S MOST RECENT message in the conversation below (if it's in Hindi or \
Hinglish, ask your question in Hindi/Hinglish too; if it's in English, use \
English) — mirror their most recent turn specifically, not just the \
conversation's overall tone, since a business owner can switch languages \
mid-interview. Output only the question, no preamble.
"""
