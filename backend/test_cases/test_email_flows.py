"""
Tests the escalation and warranty claim email notification flows,
including consent handling, email collection, implicit email detection,
and the warranty claim retry-limit escalation fallback.

NOTE: These tests send REAL emails via Resend and write REAL documents
to MongoDB (tickets/warranty_claims collections). Run sparingly, not
in a tight loop, to avoid filling your inbox/database with test noise.
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


def test_escalation_with_explicit_yes_then_email():
    state = fresh_state()
    state = send_turn(state, "One of my earbuds stopped working completely, it's so annoying")
    state = send_turn(state, "I already tried that, it's still not working, this is so frustrating")
    state = send_turn(state, "This is the third time, I'm really annoyed now")

    assert state.get("escalation_email", {}).get("status") == "asking_consent", (
        f"Expected asking_consent after 3rd strike, got: {state.get('escalation_email')}"
    )

    state = send_turn(state, "yes")
    assert state.get("escalation_email", {}).get("status") == "collecting_email"

    state = send_turn(state, "jamshed@example.com")
    assert state.get("escalation_email", {}).get("status") == "sent"
    assert state.get("escalation_email", {}).get("email") == "jamshed@example.com"
    assert state.get("escalated") is True
    assert state.get("ticket_id"), "Expected a ticket_id to be set"

    print(f"PASS: escalation email flow (explicit yes -> email) works end-to-end, ticket: {state['ticket_id']}")


def test_escalation_with_explicit_yes_then_email():
    state = fresh_state()
    state = send_turn(state, "One of my earbuds stopped working completely, it's so annoying")
    state = send_turn(state, "I already tried that, it's still not working, this is so frustrating")
    state = send_turn(state, "This is the third time, I'm really annoyed now")
    state = send_turn(state, "still not fixed")  # <-- ADD THIS: triggers 3rd attempt -> bail-out

    assert state.get("escalation_email", {}).get("status") == "asking_consent", (
        f"Expected asking_consent after 3rd strike, got: {state.get('escalation_email')}"
    )

    state = send_turn(state, "yes")
    assert state.get("escalation_email", {}).get("status") == "collecting_email"

    state = send_turn(state, "jamshed@example.com")
    assert state.get("escalation_email", {}).get("status") == "sent"
    assert state.get("escalation_email", {}).get("email") == "jamshed@example.com"
    assert state.get("escalated") is True
    assert state.get("ticket_id"), "Expected a ticket_id to be set"

    print(f"PASS: escalation email flow (explicit yes -> email) works end-to-end, ticket: {state['ticket_id']}")


def test_escalation_declined_finalizes_without_email():
    state = fresh_state()
    state = send_turn(state, "One earbud stopped working, so annoying")
    state = send_turn(state, "Still broken after trying stuff, frustrating")
    state = send_turn(state, "Third time, really annoyed")

    assert state.get("escalation_email", {}).get("status") == "asking_consent"

    state = send_turn(state, "no thanks")

    assert state.get("escalated") is True, "Expected escalation to finalize even when email is declined"
    assert state.get("escalation_email", {}).get("status") == "declined"
    assert state.get("escalation_email", {}).get("email") is None
    assert state.get("ticket_id"), "Expected a ticket_id even without email"
    print(f"PASS: declining email still finalizes the ticket, ticket: {state['ticket_id']}")


def test_warranty_claim_retry_limit_escalates():
    """Regression test: if the user never provides real claim details,
    warranty_claim_node should give up after 3 attempts and escalate,
    rather than looping forever."""
    state = fresh_state()
    state = send_turn(state, "My earbud completely died, this is so frustrating")
    state = send_turn(state, "I already tried everything, still broken")
    state = send_turn(state, "This is ridiculous, so annoyed")

    warranty_claim = state.get("warranty_claim", {})
    assert warranty_claim.get("status") == "abandoned", (
        f"Expected warranty_claim status 'abandoned' after 3 failed attempts, got: {warranty_claim}"
    )
    assert warranty_claim.get("attempts") == 3

    assert state.get("escalation_email", {}).get("status") == "asking_consent", (
        "Expected the retry-limit bail-out to hand off into the escalation consent flow"
    )

    print("PASS: warranty claim collection correctly bails out to escalation after 3 failed attempts")


def test_warranty_claim_success_with_email_consent():
    state = fresh_state()
    state = send_turn(state, "My earbud died completely after two weeks, so frustrating")
    state = send_turn(state, "It's the right one, won't turn on, had it 2 weeks")

    warranty_claim = state.get("warranty_claim", {})
    assert warranty_claim.get("status") == "filed", f"Expected claim filed, got: {warranty_claim}"
    assert warranty_claim.get("earbud_affected") == "right"

    assert state.get("warranty_email", {}).get("status") == "asking_consent"

    state = send_turn(state, "yes")
    assert state.get("warranty_email", {}).get("status") == "collecting_email"

    state = send_turn(state, "claimtest@example.com")
    assert state.get("warranty_email", {}).get("status") == "sent"
    assert state.get("warranty_email", {}).get("email") == "claimtest@example.com"

    print(f"PASS: warranty claim + email consent flow works end-to-end, claim: {warranty_claim.get('claim_id') or 'see MongoDB'}")


def test_warranty_claim_declined_email_still_confirms():
    state = fresh_state()
    state = send_turn(state, "My left earbud died after a week, this is frustrating")
    state = send_turn(state, "Left one, won't power on at all, had it about a week")

    assert state.get("warranty_claim", {}).get("status") == "filed"
    assert state.get("warranty_email", {}).get("status") == "asking_consent"

    state = send_turn(state, "no")
    assert state.get("warranty_email", {}).get("status") == "declined"
    assert state.get("warranty_email", {}).get("email") is None
    print("PASS: declining warranty email still confirms the claim was filed")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING escalation + warranty email notification flows")
    print("=" * 60)
    print("WARNING: this sends real emails and writes real MongoDB documents.\n")

    test_escalation_with_explicit_yes_then_email()
    test_escalation_with_implicit_email_skips_yes_no()
    test_escalation_declined_finalizes_without_email()
    test_warranty_claim_retry_limit_escalates()
    test_warranty_claim_success_with_email_consent()
    test_warranty_claim_declined_email_still_confirms()

    print("\nAll email flow tests passed.")