"""
Tests app/graph/nodes.py — validates classify_node, inquiry_node,
empathetic_response_node, escalate_handoff_node, check_stock_node, and
the streaming helpers. Since LLM outputs vary slightly run-to-run, these
tests check structural correctness (valid categories, non-empty
responses, correct routing/escalation/tool-call behavior) rather than
exact wording.
"""

import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app", "graph"))
from nodes import (
    classify_node,
    inquiry_node,
    empathetic_response_node,
    escalate_handoff_node,
    classify_and_retrieve,
    stream_answer,
    check_stock,
    check_stock_node,
    FAKE_INVENTORY,
    TICKETS_DIR,
)

VALID_INTENTS = {"inquiry", "complaint", "stock_check", "other"}
VALID_SENTIMENTS = {"neutral", "frustrated", "angry"}
VALID_NAMESPACES = {"product-info", "usage-guidance", "troubleshooting", "policies", "limitations", None}


def fresh_state(message: str, **overrides) -> dict:
    state = {"messages": [{"role": "user", "content": message}], "sentiment_history": [], "turn_count": 0}
    state.update(overrides)
    return state


# ---------------------------------------------------------------------
# classify_node
# ---------------------------------------------------------------------

def test_classify_returns_valid_categories():
    test_messages = [
        "What is the IPX rating on these?",
        "My earbud stopped working, this is ridiculous",
        "Hello, can you help me?",
    ]
    for msg in test_messages:
        state = classify_node(fresh_state(msg))
        assert state["intent"] in VALID_INTENTS, f"Invalid intent '{state['intent']}' for '{msg}'"
        assert state["sentiment"] in VALID_SENTIMENTS, f"Invalid sentiment '{state['sentiment']}' for '{msg}'"
        assert state["target_namespace"] in VALID_NAMESPACES, f"Invalid namespace '{state['target_namespace']}' for '{msg}'"
    print("PASS: classify_node returns valid intent/sentiment/namespace categories")


def test_classify_increments_turn_count_and_sentiment_history():
    state = fresh_state("What is the IPX rating?")
    state = classify_node(state)
    assert state["turn_count"] == 1, f"Expected turn_count=1, got {state['turn_count']}"
    assert len(state["sentiment_history"]) == 1, "Expected sentiment_history to have 1 entry"
    print("PASS: classify_node correctly updates turn_count and sentiment_history")


def test_classify_detects_clear_inquiry():
    state = classify_node(fresh_state("How long is the return window?"))
    assert state["intent"] == "inquiry", f"Expected 'inquiry', got '{state['intent']}'"
    assert state["target_namespace"] == "policies", f"Expected 'policies', got '{state['target_namespace']}'"
    print("PASS: classify_node correctly identifies a clear policy inquiry")


def test_classify_detects_clear_complaint():
    state = classify_node(fresh_state("My earbuds won't charge at all, very frustrating"))
    assert state["intent"] == "complaint", f"Expected 'complaint', got '{state['intent']}'"
    assert state["sentiment"] in {"frustrated", "angry"}, f"Expected negative sentiment, got '{state['sentiment']}'"
    print("PASS: classify_node correctly identifies a clear complaint with negative sentiment")


def test_classify_detects_stock_check_with_color():
    state = classify_node(fresh_state("Is the matte black one in stock?"))
    assert state["intent"] == "stock_check", f"Expected 'stock_check', got '{state['intent']}'"
    assert state["color"] == "Matte Black", f"Expected color='Matte Black', got '{state['color']}'"
    print("PASS: classify_node correctly identifies a stock_check intent and extracts the color")


def test_classify_stock_check_without_color():
    state = classify_node(fresh_state("Can I check if you have any in stock?"))
    assert state["intent"] == "stock_check", f"Expected 'stock_check', got '{state['intent']}'"
    assert state["color"] is None, f"Expected color=None when no color is mentioned, got '{state['color']}'"
    print("PASS: classify_node correctly leaves color=None when no color is mentioned")


# ---------------------------------------------------------------------
# inquiry_node
# ---------------------------------------------------------------------

def test_inquiry_node_grounded_answer():
    state = fresh_state("What is the IPX rating on these earbuds?", target_namespace="product-info")
    state = inquiry_node(state)
    assert state["response"], "inquiry_node returned an empty response"
    assert "IPX5" in state["response"] or "IPX" in state["response"], (
        f"Expected response to mention the IPX rating, got: {state['response']}"
    )
    print("PASS: inquiry_node produces a grounded, correct answer")


def test_inquiry_node_fallback_on_no_namespace():
    state = fresh_state("Random unrelated message", target_namespace=None)
    state = inquiry_node(state)
    assert state["response"], "inquiry_node returned an empty response"
    assert state["retrieval_score"] is None, "Expected retrieval_score to be None when no namespace given"
    print("PASS: inquiry_node falls back gracefully with no namespace")


def test_inquiry_node_adversarial_no_hallucination():
    """Forces an irrelevant question into a namespace; must not fabricate an answer."""
    state = fresh_state("What's the capital of France?", target_namespace="product-info")
    state = inquiry_node(state)
    assert state["retrieval_score"] < 0.22, (
        f"Expected low retrieval score for adversarial query, got {state['retrieval_score']}"
    )
    assert "don't have" in state["response"].lower() or "recommend reaching out" in state["response"].lower(), (
        f"Expected an honest fallback response, got: {state['response']}"
    )
    print("PASS: inquiry_node correctly avoids hallucinating on an out-of-scope question")


# ---------------------------------------------------------------------
# empathetic_response_node
# ---------------------------------------------------------------------

def test_empathetic_first_negative_turn_gives_fix():
    state = fresh_state(
        "One of my earbuds won't charge, this is annoying",
        target_namespace="troubleshooting",
        sentiment_history=["frustrated"],
    )
    state = empathetic_response_node(state)
    assert state["escalate"] is False, "Expected escalate=False on first negative turn"
    assert state["response"], "empathetic_response_node returned an empty response"
    print("PASS: empathetic_response_node gives a grounded fix on first negative turn")


def test_empathetic_escalates_on_third_negative_turn():
    state = fresh_state(
        "It's still not charging, I've tried everything",
        target_namespace="troubleshooting",
        sentiment_history=["frustrated", "angry", "angry"],
    )
    state = empathetic_response_node(state)
    assert state["escalate"] is True, "Expected escalate=True on 3rd consecutive negative turn"
    print("PASS: empathetic_response_node correctly flags escalation on 3rd negative turn")


def test_empathetic_does_not_escalate_on_mixed_history():
    """One neutral turn between two negatives should NOT count as consecutive."""
    state = fresh_state(
        "It's still broken",
        target_namespace="troubleshooting",
        sentiment_history=["angry", "neutral"],
    )
    state = empathetic_response_node(state)
    assert state["escalate"] is False, (
        "Expected escalate=False since the last two entries are not both negative"
    )
    print("PASS: empathetic_response_node does not escalate on non-consecutive negative turns")


# ---------------------------------------------------------------------
# escalate_handoff_node
# ---------------------------------------------------------------------

def test_escalate_handoff_logs_ticket_and_responds():
    state = fresh_state(
        "It's still not charging",
        sentiment_history=["frustrated", "angry", "angry"],
        turn_count=3,
        intent="complaint",
        target_namespace="troubleshooting",
    )
    state["messages"] = [
        {"role": "user", "content": "One of my earbuds won't charge, this is annoying"},
        {"role": "assistant", "content": "Let's try a few troubleshooting steps..."},
        {"role": "user", "content": "It's still not charging"},
    ]

    state = escalate_handoff_node(state)

    assert state["escalated"] is True, "Expected escalated=True"
    assert state["response"], "escalate_handoff_node returned an empty response"
    assert state["ticket_id"], "escalate_handoff_node did not set a ticket_id"
    assert "support@shopnest.com" in state["response"], "Expected handoff message to mention support email"

    ticket_path = os.path.join(TICKETS_DIR, f"{state['ticket_id']}.json")
    assert os.path.exists(ticket_path), f"Ticket file not found at {ticket_path}"

    with open(ticket_path, "r", encoding="utf-8") as f:
        ticket = json.load(f)
    assert ticket["sentiment_history"] == ["frustrated", "angry", "angry"], "Ticket sentiment_history mismatch"
    assert len(ticket["conversation"]) == 3, "Ticket conversation length mismatch"
    assert ticket["assigned_to"] == "support@shopnest.com", "Ticket should record assigned support contact"

    print(f"PASS: escalate_handoff_node logs a correct ticket at {ticket_path}")

    os.remove(ticket_path)


# ---------------------------------------------------------------------
# check_stock tool + check_stock_node
# ---------------------------------------------------------------------

def test_check_stock_tool_known_color_in_stock():
    result = check_stock("Matte Black")
    assert result["found"] is True
    assert result["in_stock"] is True
    print("PASS: check_stock tool correctly reports Matte Black as in stock")


def test_check_stock_tool_known_color_out_of_stock():
    result = check_stock("Pearl White")
    assert result["found"] is True
    assert result["in_stock"] is False
    print("PASS: check_stock tool correctly reports Pearl White as out of stock")


def test_check_stock_tool_unknown_color():
    result = check_stock("Galaxy Purple")
    assert result["found"] is False
    assert result["in_stock"] is None
    print("PASS: check_stock tool correctly handles an unknown color")


def test_check_stock_node_in_stock_response():
    state = fresh_state("Is the black one in stock?", color="Matte Black")
    state = check_stock_node(state)
    assert state["response"], "Expected a non-empty response"
    assert "black" in state["response"].lower() or "matte" in state["response"].lower(), (
        f"Expected response to mention the color, got: {state['response']}"
    )
    print("PASS: check_stock_node produces a correct in-stock response")


def test_check_stock_node_out_of_stock_only_mentions_real_colors():
    """
    Regression test for the hallucination bug: the LLM once invented
    fictional alternative colors ('Black Onyx', 'Midnight Blue') instead
    of using the real inventory. This confirms the grounding fix holds.
    """
    state = fresh_state("Do you have these in pearl white?", color="Pearl White")
    state = check_stock_node(state)

    response_lower = state["response"].lower()

    fabricated_indicators = ["onyx", "midnight", "galaxy", "crimson", "emerald"]
    for fake_term in fabricated_indicators:
        assert fake_term not in response_lower, (
            f"Response contains a fabricated color term '{fake_term}': {state['response']}"
        )

    other_in_stock = [c for c, in_stock in FAKE_INVENTORY.items() if in_stock and c != "Pearl White"]
    mentioned_real_alternative = any(c.lower() in response_lower for c in other_in_stock)
    assert mentioned_real_alternative, (
        f"Expected response to mention a real in-stock alternative {other_in_stock}, got: {state['response']}"
    )

    print("PASS: check_stock_node out-of-stock response only mentions real inventory colors")


def test_check_stock_node_no_color_asks_for_clarification():
    state = fresh_state("Can I check if you have any in stock?", color=None)
    state = check_stock_node(state)
    assert state["response"], "Expected a non-empty response"
    assert "which" in state["response"].lower() or "color" in state["response"].lower(), (
        f"Expected a clarifying question about color, got: {state['response']}"
    )
    print("PASS: check_stock_node asks for clarification when no color is given")


def test_check_stock_node_unrecognized_color():
    state = fresh_state("Do you have it in neon green?", color="Neon Green")
    state = check_stock_node(state)
    assert state["response"], "Expected a non-empty response"
    print("PASS: check_stock_node handles an unrecognized color gracefully")


# ---------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------

def test_classify_and_retrieve_sets_context():
    state = classify_and_retrieve(fresh_state("What is the IPX rating on these earbuds?"))
    assert state["intent"] == "inquiry"
    assert state["context"] is not None, "Expected context to be set for a clear inquiry"
    assert "IPX5" in state["context"], "Expected retrieved context to mention IPX5"
    print("PASS: classify_and_retrieve correctly sets context for streaming path")


def test_stream_answer_yields_text():
    state = classify_and_retrieve(fresh_state("How long is the return window?"))
    chunks = list(stream_answer(state))
    full_text = "".join(chunks)
    assert len(chunks) > 0, "stream_answer yielded no chunks"
    assert full_text.strip(), "stream_answer produced empty combined text"
    print(f"PASS: stream_answer yielded {len(chunks)} chunks, combined length {len(full_text)} chars")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING nodes.py")
    print("=" * 60)

    test_classify_returns_valid_categories()
    test_classify_increments_turn_count_and_sentiment_history()
    test_classify_detects_clear_inquiry()
    test_classify_detects_clear_complaint()
    test_classify_detects_stock_check_with_color()
    test_classify_stock_check_without_color()

    test_inquiry_node_grounded_answer()
    test_inquiry_node_fallback_on_no_namespace()
    test_inquiry_node_adversarial_no_hallucination()

    test_empathetic_first_negative_turn_gives_fix()
    test_empathetic_escalates_on_third_negative_turn()
    test_empathetic_does_not_escalate_on_mixed_history()

    test_escalate_handoff_logs_ticket_and_responds()

    test_check_stock_tool_known_color_in_stock()
    test_check_stock_tool_known_color_out_of_stock()
    test_check_stock_tool_unknown_color()
    test_check_stock_node_in_stock_response()
    test_check_stock_node_out_of_stock_only_mentions_real_colors()
    test_check_stock_node_no_color_asks_for_clarification()
    test_check_stock_node_unrecognized_color()

    test_classify_and_retrieve_sets_context()
    test_stream_answer_yields_text()

    print("\nAll nodes.py tests passed.")