"""
FastAPI app exposing the VoiceCart voice agent.

/api/voice         - accepts EITHER an audio file OR text, returns full JSON response
/api/text/stream    - text-only input, streams the answer token-by-token via SSE
                      for inquiry/complaint intents; other/stock_check/warranty_claim/
                      order_booking/order_tracking/order_cancel/escalation-email/
                      warranty-email intents return their full response in a single
                      token event since they don't benefit from streaming.
                      Synthesizes and returns speech either way.
/api/health         - health check
/api/session/{id}   - reset a session
"""

import os
import sys
import uuid
import base64
import json
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app", "graph"))
from db import sessions_collection
from voice_pipeline import run_voice_turn, synthesize_speech
from graph import graph
from nodes import (
    classify_and_retrieve,
    stream_answer,
    empathetic_response_node,
    escalate_handoff_node,
    escalation_followup_node,
    other_node,
    check_stock_node,
    warranty_claim_node,
    warranty_email_node,
    order_booking_node,
    order_tracking_node,
    order_cancel_node,
    manual_escalation_followup_node,
    _normalize_spoken_email,
)

app = FastAPI(title="VoiceCart Support Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, dict] = {}


def _load_session_from_db(session_id: str) -> Optional[dict]:
    """Falls back to MongoDB if a session isn't in memory (e.g. after a server restart)."""
    doc = sessions_collection.find_one({"session_id": session_id})
    if doc:
        doc.pop("_id", None)
        doc.pop("session_id", None)
        return doc
    return None


def _save_session_to_db(session_id: str, state: dict) -> None:
    """Persists the current session state so it survives a server restart."""
    try:
        doc = dict(state)
        doc["session_id"] = session_id
        sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": doc},
            upsert=True,
        )
    except Exception as e:
        print(f"[_save_session_to_db] Failed to persist session: {e}")


class VoiceResponse(BaseModel):
    session_id: str
    transcript: str
    response_text: str
    response_audio_base64: str
    intent: Optional[str] = None
    sentiment: Optional[str] = None
    sentiment_history: list[str] = []
    escalated: bool = False
    is_handoff_turn: bool = False
    ticket_id: Optional[str] = None


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/voice", response_model=VoiceResponse)
async def voice_endpoint(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    if not file and not text:
        raise HTTPException(status_code=400, detail="Provide either 'file' (audio) or 'text', not neither.")
    if file and text:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'text', not both.")

    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in SESSIONS:
        restored = _load_session_from_db(session_id)
        SESSIONS[session_id] = restored or {
            "messages": [],
            "sentiment_history": [],
            "turn_count": 0,
        }

    state = SESSIONS[session_id]

    if file:
        suffix = os.path.splitext(file.filename)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            contents = await file.read()
            tmp_in.write(contents)
            input_path = tmp_in.name

        output_path = input_path.replace(suffix, "_response.wav")

        try:
            state = run_voice_turn(input_path, state, output_audio_path=output_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Voice pipeline error: {str(e)}")
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    else:
        text = _normalize_spoken_email(text)
        state["messages"].append({"role": "user", "content": text})

        try:
            state = graph.invoke(state)
        except Exception as e:
            print(f"[voice_endpoint] Graph error: {e}")
            state["response"] = "I'm having trouble processing that right now. Please try again in a moment."

        response_text = state.get("response", "I'm sorry, I didn't catch that.")

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        try:
            synthesize_speech(response_text, output_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

        state["messages"].append({"role": "assistant", "content": response_text})
        state["transcript"] = text

    SESSIONS[session_id] = state
    _save_session_to_db(session_id, state)

    with open(output_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    if os.path.exists(output_path):
        os.remove(output_path)

    return VoiceResponse(
        session_id=session_id,
        transcript=state.get("transcript", ""),
        response_text=state.get("response", ""),
        response_audio_base64=audio_base64,
        intent=state.get("intent"),
        sentiment=state.get("sentiment"),
        sentiment_history=state.get("sentiment_history", []),
        escalated=state.get("escalated", False),
        is_handoff_turn=state.get("is_handoff_turn", False),
        ticket_id=state.get("ticket_id"),
    )


def _check_pending_short_circuit(state: dict) -> Optional[str]:
    """
    Mirrors graph.py's route_at_start priority order: checks whether this
    session is mid a multi-turn sub-dialogue (email consent/collection,
    warranty claim collection, order booking collection) that should
    bypass classification entirely. Returns the node name to route to,
    or None for normal classify flow.
    """
    if state.get("escalated") and not state.get("escalation_acknowledged"):
        return "post_escalation"

    escalation_email = state.get("escalation_email") or {}
    if escalation_email.get("status") in ("asking_consent", "collecting_email"):
        return "escalation_followup"

    warranty_email = state.get("warranty_email") or {}
    if warranty_email.get("status") in ("asking_consent", "collecting_email"):
        return "warranty_email"

    warranty_claim = state.get("warranty_claim") or {}
    if warranty_claim.get("status") == "collecting":
        return "warranty_claim"

    order_booking = state.get("order_booking") or {}
    if order_booking.get("status") in ("collecting", "confirming"):
        return "order_booking"

    if state.get("offered_manual_escalation"):
        return "manual_escalation_followup"

    return None


POST_ESCALATION_RESPONSE = (
    "Your conversation has already been sent to our support team, and they'll be in touch soon. "
    "Is there anything else I can help you with in the meantime?"
)


@app.post("/api/text/stream")
async def text_stream_endpoint(text: str = Form(...), session_id: Optional[str] = Form(None)):
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in SESSIONS:
        restored = _load_session_from_db(session_id)
        SESSIONS[session_id] = restored or {"messages": [], "sentiment_history": [], "turn_count": 0}

    state = SESSIONS[session_id]
    pending_route = _check_pending_short_circuit(state)

    text = _normalize_spoken_email(text)
    state["messages"].append({"role": "user", "content": text})

    def event_generator():
        nonlocal state

        # --- Short-circuit paths: skip classification entirely ---
        if pending_route == "post_escalation":
            full_response = POST_ESCALATION_RESPONSE
            state["is_handoff_turn"] = False
            state["escalation_acknowledged"] = True

            yield f"event: meta\ndata: {json.dumps({'session_id': session_id, 'intent': None, 'sentiment': state.get('sentiment')})}\n\n"
            yield f"event: token\ndata: {json.dumps({'text': full_response})}\n\n"

            state["messages"].append({"role": "assistant", "content": full_response})
            SESSIONS[session_id] = state
            _save_session_to_db(session_id, state)

            audio_base64 = _synthesize_and_encode(full_response)
            yield f"event: done\ndata: {json.dumps({'escalated': state.get('escalated', False), 'is_handoff_turn': False, 'ticket_id': state.get('ticket_id'), 'audio_base64': audio_base64})}\n\n"
            return

        if pending_route in ("escalation_followup", "warranty_email", "warranty_claim", "order_booking", "manual_escalation_followup"):
            node_fn = {
                "escalation_followup": escalation_followup_node,
                "warranty_email": warranty_email_node,
                "warranty_claim": warranty_claim_node,
                "order_booking": order_booking_node,
                "manual_escalation_followup": manual_escalation_followup_node,
            }[pending_route]

            updated_state = node_fn(state)
            full_response = updated_state["response"]
            state.update(updated_state)

            yield f"event: meta\ndata: {json.dumps({'session_id': session_id, 'intent': state.get('intent'), 'sentiment': state.get('sentiment')})}\n\n"
            yield f"event: token\ndata: {json.dumps({'text': full_response})}\n\n"

            state["messages"].append({"role": "assistant", "content": full_response})
            SESSIONS[session_id] = state
            _save_session_to_db(session_id, state)

            audio_base64 = _synthesize_and_encode(full_response)
            yield f"event: done\ndata: {json.dumps({'escalated': state.get('escalated', False), 'is_handoff_turn': state.get('is_handoff_turn', False), 'ticket_id': state.get('ticket_id'), 'audio_base64': audio_base64})}\n\n"
            return

        # --- Normal flow: classify first ---
        try:
            state_after_classify = classify_and_retrieve(state)
        except Exception as e:
            print(f"[text_stream_endpoint] classify_and_retrieve error: {e}")
            state_after_classify = dict(state)
            state_after_classify["intent"] = "other"
            state_after_classify["context"] = None

        intent = state_after_classify.get("intent")

        yield f"event: meta\ndata: {json.dumps({'session_id': session_id, 'intent': intent, 'sentiment': state_after_classify.get('sentiment')})}\n\n"

        should_start_warranty_claim = (
            intent == "complaint"
            and state_after_classify.get("warranty_eligible")
        )

        explicit_escalation = state_after_classify.get("explicit_escalation_request")

        full_response = ""

        if explicit_escalation:
            escalated_state = escalate_handoff_node(state_after_classify)
            full_response = escalated_state["response"]
            yield f"event: token\ndata: {json.dumps({'text': full_response})}\n\n"
            state_after_classify.update(escalated_state)

        elif should_start_warranty_claim:
            claim_state = warranty_claim_node(state_after_classify)
            full_response = claim_state["response"]
            state_after_classify.update(claim_state)
            yield f"event: token\ndata: {json.dumps({'text': full_response})}\n\n"

            # warranty_claim_node may set escalate=True on its own (retry-limit
            # bail-out) — if so, immediately follow through to escalate_handoff_node,
            # same as graph.py's conditional edge does.
            if state_after_classify.get("escalate"):
                escalated_state = escalate_handoff_node(state_after_classify)
                full_response = escalated_state["response"]
                yield f"event: token\ndata: {json.dumps({'text': full_response})}\n\n"
                state_after_classify.update(escalated_state)

        elif intent == "other":
            other_state = other_node(state_after_classify)
            full_response = other_state["response"]
            state_after_classify.update(other_state)
            yield f"event: token\ndata: {json.dumps({'text': full_response})}\n\n"

        elif intent == "stock_check":
            stock_state = check_stock_node(state_after_classify)
            full_response = stock_state["response"]
            state_after_classify.update(stock_state)
            yield f"event: token\ndata: {json.dumps({'text': full_response})}\n\n"

        elif intent == "order_booking":
            order_state = order_booking_node(state_after_classify)
            full_response = order_state["response"]
            state_after_classify.update(order_state)
            yield f"event: token\ndata: {json.dumps({'text': full_response})}\n\n"

        elif intent == "order_tracking":
            tracking_state = order_tracking_node(state_after_classify)
            full_response = tracking_state["response"]
            state_after_classify.update(tracking_state)
            yield f"event: token\ndata: {json.dumps({'text': full_response})}\n\n"

        elif intent == "order_cancel":
            cancel_state = order_cancel_node(state_after_classify)
            full_response = cancel_state["response"]
            state_after_classify.update(cancel_state)
            yield f"event: token\ndata: {json.dumps({'text': full_response})}\n\n"

        elif intent == "complaint":
            empathetic_state = empathetic_response_node(state_after_classify)
            full_response = empathetic_state["response"]
            state_after_classify.update(empathetic_state)
            yield f"event: token\ndata: {json.dumps({'text': full_response})}\n\n"

        else:
            for chunk in stream_answer(state_after_classify):
                full_response += chunk
                yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"
            state_after_classify["response"] = full_response

        state_after_classify["messages"].append({"role": "assistant", "content": full_response})
        SESSIONS[session_id] = state_after_classify
        _save_session_to_db(session_id, state_after_classify)

        audio_base64 = _synthesize_and_encode(full_response)

        yield f"event: done\ndata: {json.dumps({'escalated': state_after_classify.get('escalated', False), 'is_handoff_turn': state_after_classify.get('is_handoff_turn', False), 'ticket_id': state_after_classify.get('ticket_id'), 'audio_base64': audio_base64})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _synthesize_and_encode(text: str) -> str:
    """Synthesizes speech for the given text and returns it as a base64 string."""
    try:
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        synthesize_speech(text, tmp_path)
        with open(tmp_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        os.remove(tmp_path)
        return audio_base64
    except Exception as e:
        print(f"[text_stream] TTS error: {e}")
        return ""


@app.delete("/api/session/{session_id}")
def reset_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    sessions_collection.delete_one({"session_id": session_id})
    return {"status": "session reset"}