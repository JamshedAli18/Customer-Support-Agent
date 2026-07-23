"""
Tests app/graph/graph.py — validates the COMPILED graph's conditional
routing end-to-end. nodes.py tests proved each node works in isolation;
this proves the graph wires them together correctly via graph.invoke().

Escalation rule: 3 consecutive negative (frustrated/angry) turns.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app", "graph"))
from graph import graph


def fresh_state() -> dict:
    return {"messages": [], "sentiment_history": [], "turn_count": 0}


def send_turn(state: dict, message: str) -> dict:
    state["messages"].append({"role": "user", "content": message})
    return graph.invoke(state)


def test_inquiry_routes_to_inquiry_node():
    state = send_turn(fresh_state(), "What is the IPX rating on these earbuds?")
    assert state["intent"] == "inquiry", f"Expected intent='inquiry', got '{state['intent']}'"
    assert state["response"], "Expected a non-empty response"
    assert state.get("escalated", False) is False, "Should not be escalated on a simple inquiry"
    print("PASS: inquiry correctly routes through inquiry_node")


def test_greeting_routes_to_other_node():
    state = send_turn(fresh_state(), "Hello, can you help me?")
    assert state["intent"] == "other", f"Expected intent='other', got '{state['intent']}'"
    assert state["response"], "Expected a non-empty response from other_node"
    print("PASS: greeting correctly routes through other_node")


def test_first_complaint_routes_to_empathetic_not_escalate():
    state = send_turn(fresh_state(), "One of my earbuds stopped working, this is annoying")
    assert state["intent"] == "complaint", f"Expected intent='complaint', got '{state['intent']}'"
    assert state.get("escalated", False) is False, "Should NOT escalate on the first negative turn"
    assert state["response"], "Expected a non-empty empathetic response"
    print("PASS: first complaint routes through empathetic_response_node without escalating")


def test_three_consecutive_complaints_escalate():
    """
    Full end-to-end proof: 3 consecutive negative turns through the
    ACTUAL compiled graph (not just the node function directly) results
    in escalation, a logged ticket, and a handoff response. The 1st and
    2nd negative turns should NOT escalate.
    """
    state = fresh_state()

    state = send_turn(state, "One of my earbuds stopped working completely, it's so annoying")
    assert state.get("escalated", False) is False, "Should not escalate after only 1 negative turn"

    state = send_turn(state, "I already tried that, it's still not working, this is so frustrating")
    assert state.get("escalated", False) is False, "Should not escalate after only 2 negative turns"

    state = send_turn(state, "This is the third time, I'm really annoyed now")
    assert state.get("escalated", False) is True, "Expected escalation after 3 consecutive negative turns"
    assert state.get("ticket_id"), "Expected a ticket_id to be set after escalation"
    assert len(state["sentiment_history"]) == 3, (
        f"Expected sentiment_history length 3, got {len(state['sentiment_history'])}"
    )

    print(f"PASS: 3 consecutive complaints escalate through the full graph (ticket: {state['ticket_id']})")


def test_state_accumulates_across_turns():
    """Confirms messages, turn_count, and sentiment_history all grow correctly turn over turn."""
    state = fresh_state()
    state = send_turn(state, "What is the IPX rating?")
    state = send_turn(state, "What about the battery life?")

    assert state["turn_count"] == 2, f"Expected turn_count=2, got {state['turn_count']}"
    assert len(state["sentiment_history"]) == 2, "Expected 2 sentiment_history entries"
    user_messages = [m for m in state["messages"] if m["role"] == "user"]
    assert len(user_messages) == 2, f"Expected 2 user messages, got {len(user_messages)}"
    print("PASS: state correctly accumulates turn_count and sentiment_history across turns")


def test_adversarial_question_does_not_hallucinate_through_graph():
    state = send_turn(fresh_state(), "What's the capital of France?")
    assert state["response"], "Expected a non-empty response"
    assert "paris" not in state["response"].lower(), (
        f"Response should not contain fabricated geography facts: {state['response']}"
    )
    print("PASS: adversarial off-topic question does not produce a hallucinated answer through the graph")


def test_stock_check_routes_correctly():
    state = send_turn(fresh_state(), "Is the matte black one available?")
    assert state["intent"] == "stock_check", f"Expected intent='stock_check', got '{state['intent']}'"
    assert state["response"], "Expected a non-empty response"
    assert "black" in state["response"].lower() or "matte" in state["response"].lower(), (
        f"Expected response to mention the color, got: {state['response']}"
    )
    print("PASS: stock_check intent correctly routes through check_stock_node")


def test_stock_check_color_persists_in_graph_state():
    """
    Regression test for the TypedDict schema bug: 'color' was being
    silently dropped between classify_node and check_stock_node because
    it wasn't declared in VoiceCartState. Confirms the fix holds.
    """
    state = send_turn(fresh_state(), "Do you have these in slate blue?")
    assert state.get("color") == "Slate Blue", (
        f"Expected color='Slate Blue' to persist through the graph, got '{state.get('color')}'"
    )
    print("PASS: color field correctly persists across graph nodes (TypedDict schema regression test)")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING graph.py (compiled graph, end-to-end routing)")
    print("=" * 60)

    test_inquiry_routes_to_inquiry_node()
    test_greeting_routes_to_other_node()
    test_first_complaint_routes_to_empathetic_not_escalate()
    test_three_consecutive_complaints_escalate()
    test_state_accumulates_across_turns()
    test_adversarial_question_does_not_hallucinate_through_graph()
    test_stock_check_routes_correctly()
    test_stock_check_color_persists_in_graph_state()

    print("\nAll graph.py tests passed.")