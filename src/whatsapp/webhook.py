"""Meta WhatsApp Cloud API sandbox webhook (spec Component C).

Single-business demo only, per the spec's own non-goal of multi-tenant
infrastructure -- which business's KB to answer from is fixed via
WHATSAPP_BUSINESS_ID, no per-conversation routing.

Requires a Meta developer app with the Cloud API sandbox configured:
WHATSAPP_VERIFY_TOKEN (arbitrary string you choose, entered in the Meta
app's webhook config), WHATSAPP_ACCESS_TOKEN (temporary or system-user
token from the app), WHATSAPP_PHONE_NUMBER_ID (from the Cloud API sandbox
dashboard). See .env.example.
"""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter, Request, Response

from src.whatsapp.rag import answer_question

logger = logging.getLogger(__name__)
router = APIRouter()

GRAPH_API_URL = "https://graph.facebook.com/v20.0"


def _config() -> dict:
    return {
        "verify_token": os.environ.get("WHATSAPP_VERIFY_TOKEN", ""),
        "access_token": os.environ.get("WHATSAPP_ACCESS_TOKEN", ""),
        "phone_number_id": os.environ.get("WHATSAPP_PHONE_NUMBER_ID", ""),
        "business_id": os.environ.get("WHATSAPP_BUSINESS_ID", ""),
    }


@router.get("/webhook")
def verify_webhook(request: Request) -> Response:
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge", "")

    if mode == "subscribe" and token == _config()["verify_token"]:
        return Response(content=challenge, media_type="text/plain")
    return Response(status_code=403)


@router.post("/webhook")
async def receive_message(request: Request) -> dict:
    config = _config()
    payload = await request.json()

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            for message in change.get("value", {}).get("messages", []):
                text = message.get("text", {}).get("body")
                sender = message.get("from")
                if not text or not sender:
                    continue
                try:
                    answer = answer_question(config["business_id"], text)
                    logger.info("RAG answer for %r: %r", text, answer)
                    _send_reply(sender, answer, config)
                except Exception:
                    # Meta retries the whole webhook delivery on a non-2xx
                    # response, which would re-trigger the RAG call for
                    # every message in the batch -- log and move on instead.
                    logger.exception("Failed to answer/send reply to %s", sender)

    return {"status": "ok"}


def _send_reply(to: str, text: str, config: dict) -> None:
    url = f"{GRAPH_API_URL}/{config['phone_number_id']}/messages"
    headers = {"Authorization": f"Bearer {config['access_token']}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": text},
    }
    response = httpx.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
