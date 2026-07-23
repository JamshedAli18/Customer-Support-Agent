"""
Tests main.py — validates the FastAPI endpoints using TestClient.
Calls the app in-process, no running server required.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient
from main import app, SESSIONS

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("PASS: /api/health returns ok")


def test_voice_endpoint_rejects_neither_file_nor_text():
    response = client.post("/api/voice", data={})
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print("PASS: /api/voice correctly rejects request with neither file nor text")


def test_voice_endpoint_text_path():
    response = client.post("/api/voice", data={"text": "What is the IPX rating on these earbuds?"})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data["transcript"] == "What is the IPX rating on these earbuds?"
    assert data["response_text"], "Expected non-empty response_text"
    assert "IPX" in data["response_text"], f"Expected IPX mention, got: {data['response_text']}"
    assert data["response_audio_base64"], "Expected non-empty audio"
    assert data["intent"] == "inquiry"
    assert data["session_id"], "Expected a session_id to be returned"

    print(f"PASS: /api/voice text path works (session_id: {data['session_id']})")
    return data["session_id"]


def test_voice_endpoint_session_persists_across_calls():
    """Confirms the in-memory session store actually persists state across two separate calls."""
    first = client.post("/api/voice", data={"text": "What is the IPX rating?"})
    session_id = first.json()["session_id"]

    second = client.post("/api/voice", data={"text": "What about the battery life?", "session_id": session_id})
    second_data = second.json()

    assert second_data["session_id"] == session_id, "Session ID should remain the same"
    assert len(second_data["sentiment_history"]) == 2, (
        f"Expected 2 sentiment_history entries after 2 calls with same session, got {len(second_data['sentiment_history'])}"
    )
    print(f"PASS: session state persists correctly across separate /api/voice calls (session: {session_id})")


def test_voice_endpoint_escalation_over_three_turns():
    """
    Full HTTP-layer proof: 3 consecutive negative-sentiment text turns
    via the actual API trigger escalation, matching the 3-strike rule.
    """
    r1 = client.post("/api/voice", data={"text": "One of my earbuds stopped working completely, it's so annoying"})
    session_id = r1.json()["session_id"]
    assert r1.json()["escalated"] is False, "Should not escalate after 1 negative turn"

    r2 = client.post("/api/voice", data={"text": "I already tried that, still broken, very frustrating", "session_id": session_id})
    assert r2.json()["escalated"] is False, "Should not escalate after 2 negative turns"

    r3 = client.post("/api/voice", data={"text": "This is the third time, I'm really annoyed now", "session_id": session_id})
    r3_data = r3.json()
    assert r3_data["escalated"] is True, "Expected escalation after 3 consecutive negative turns"
    assert r3_data["ticket_id"], "Expected a ticket_id after escalation"
    assert "support@shopnest.com" in r3_data["response_text"], (
        f"Expected handoff message to mention support email, got: {r3_data['response_text']}"
    )

    print(f"PASS: 3-turn escalation works through the actual API (ticket: {r3_data['ticket_id']})")


def test_text_stream_endpoint_returns_sse():
    response = client.post(
        "/api/text/stream",
        data={"text": "How long is the return window?"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", ""), (
        f"Expected text/event-stream content type, got: {response.headers.get('content-type')}"
    )

    body = response.text
    assert "event: meta" in body, "Expected a 'meta' event in the stream"
    assert "event: token" in body, "Expected at least one 'token' event in the stream"
    assert "event: done" in body, "Expected a 'done' event in the stream"

    print("PASS: /api/text/stream returns a valid SSE stream with meta/token/done events")


def test_session_reset_endpoint():
    create = client.post("/api/voice", data={"text": "What is the IPX rating?"})
    session_id = create.json()["session_id"]
    assert session_id in SESSIONS, "Session should exist in memory after creation"

    response = client.delete(f"/api/session/{session_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "session reset"
    assert session_id not in SESSIONS, "Session should be removed from memory after reset"

    print("PASS: /api/session/{id} correctly resets a session")


def test_session_reset_nonexistent_session():
    response = client.delete("/api/session/does-not-exist-12345")
    assert response.status_code == 200
    assert response.json()["status"] == "session not found"
    print("PASS: resetting a nonexistent session returns a graceful response, not an error")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING main.py (FastAPI endpoints via TestClient)")
    print("=" * 60)

    test_health_check()
    test_voice_endpoint_rejects_neither_file_nor_text()
    test_voice_endpoint_text_path()
    test_voice_endpoint_session_persists_across_calls()
    test_voice_endpoint_escalation_over_three_turns()
    test_text_stream_endpoint_returns_sse()
    test_session_reset_endpoint()
    test_session_reset_nonexistent_session()

    print("\nAll main.py (API) tests passed.")