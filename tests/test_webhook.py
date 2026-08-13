"""Webhook request-parsing tests — mocks out the RAG answer and outbound
HTTP call, so this only verifies Meta's Cloud API webhook contract
(verification handshake, incoming message payload shape), no API calls.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)

SAMPLE_INCOMING_PAYLOAD = {
    "entry": [
        {
            "changes": [
                {
                    "value": {
                        "messages": [
                            {"from": "919876543210", "text": {"body": "What are your hours?"}}
                        ]
                    }
                }
            ]
        }
    ]
}


def test_verify_webhook_succeeds_with_matching_token(monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "my-secret-token")
    response = client.get(
        "/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "my-secret-token", "hub.challenge": "12345"},
    )
    assert response.status_code == 200
    assert response.text == "12345"


def test_verify_webhook_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "my-secret-token")
    response = client.get(
        "/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "wrong-token", "hub.challenge": "12345"},
    )
    assert response.status_code == 403


def test_receive_message_calls_rag_and_sends_reply(monkeypatch):
    monkeypatch.setenv("WHATSAPP_BUSINESS_ID", "biz1")
    with patch("src.whatsapp.webhook.answer_question", return_value="We're open 9 to 6.") as mock_answer, \
         patch("src.whatsapp.webhook.httpx.post") as mock_post:
        mock_post.return_value.raise_for_status.return_value = None

        response = client.post("/webhook", json=SAMPLE_INCOMING_PAYLOAD)

        assert response.status_code == 200
        mock_answer.assert_called_once_with("biz1", "What are your hours?")
        mock_post.assert_called_once()
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["to"] == "919876543210"
        assert sent_payload["text"]["body"] == "We're open 9 to 6."


def test_receive_message_ignores_payload_with_no_text():
    empty_payload = {"entry": [{"changes": [{"value": {"messages": []}}]}]}
    response = client.post("/webhook", json=empty_payload)
    assert response.status_code == 200
