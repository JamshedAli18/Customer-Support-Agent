"""
Wires the nodes into a LangGraph StateGraph with conditional routing.

Flow:
  START -> (escalated already?) -> post_escalation
         -> (mid escalation-email consent/collection?) -> escalation_followup
         -> (mid warranty-email consent/collection?) -> warranty_email
         -> (warranty claim collecting?) -> warranty_claim
         -> (order booking collecting?) -> order_booking
         -> classify
  classify -> (router) -> inquiry | empathetic | check_stock | warranty_claim |
              order_booking | order_tracking | order_cancel | other
  inquiry -> END
  empathetic -> (escalate flagged?) -> escalate | END
  escalate -> END (asks email consent, ends turn)
  escalation_followup -> END (handles consent + email collection)
  other -> END
  check_stock -> END
  warranty_claim -> END (asks email consent after filing, ends turn)
  warranty_email -> END (handles consent + email collection)
  order_booking -> END
  order_tracking -> END
  order_cancel -> END
  post_escalation -> END
"""

from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END

from nodes import (
    classify_node,
    inquiry_node,
    empathetic_response_node,
    escalate_handoff_node,
    escalation_followup_node,
    check_stock_node,
    post_escalation_node,
    other_node,
    warranty_claim_node,
    warranty_email_node,
    order_booking_node,
    order_tracking_node,
    order_cancel_node,
)


class VoiceCartState(TypedDict, total=False):
    messages: list[dict]
    intent: str
    sentiment: str
    target_namespace: Optional[str]
    color: Optional[str]
    quantity: Optional[int]
    order_id_mentioned: Optional[str]
    order_booking: Optional[dict]
    last_shipping_address: Optional[str]
    user_name: Optional[str]
    warranty_eligible: bool
    warranty_claim: Optional[dict]
    warranty_email: Optional[dict]
    escalation_email: Optional[dict]
    _pending_ticket: Optional[dict]
    sentiment_history: list[str]
    turn_count: int
    response: str
    retrieval_score: Optional[float]
    escalate: bool
    escalated: bool
    escalation_acknowledged: bool
    is_handoff_turn: bool
    ticket_id: Optional[str]


def route_after_classify(state: dict) -> Literal[
    "inquiry", "empathetic", "check_stock", "warranty_claim",
    "order_booking", "order_tracking", "order_cancel", "other"
]:
    """Routes based on classify_node's intent output."""
    intent = state.get("intent", "other")
    if intent == "inquiry":
        return "inquiry"
    if intent == "stock_check":
        return "check_stock"
    if intent == "order_booking":
        return "order_booking"
    if intent == "order_tracking":
        return "order_tracking"
    if intent == "order_cancel":
        return "order_cancel"
    if intent == "complaint":
        warranty_claim = state.get("warranty_claim") or {}
        terminal_statuses = {"filed", "abandoned", "declined_out_of_warranty"}
        already_filed = warranty_claim.get("status") in terminal_statuses
        if state.get("warranty_eligible") and not already_filed:
            return "warranty_claim"
        return "empathetic"
    return "other"


def route_after_empathetic(state: dict) -> Literal["escalate", "end"]:
    """Routes to escalation if empathetic_response_node flagged it."""
    if state.get("escalate"):
        return "escalate"
    return "end"


def route_at_start(state: dict) -> Literal[
    "post_escalation", "escalation_followup", "warranty_email", "warranty_claim", "order_booking", "classify"
]:
    """
    Checks session status before running classification, in priority order:
    - just escalated, not yet acknowledged -> post_escalation (shows ONCE)
    - mid escalation-email consent/collection -> escalation_followup
    - mid warranty-email consent/collection -> warranty_email
    - mid warranty-claim detail collection -> warranty_claim
    - mid order-booking detail collection -> order_booking
    - otherwise -> normal classify flow (even if escalated earlier — conversation
      should continue normally after the one-time acknowledgment)
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
    if order_booking.get("status") == "collecting":
        return "order_booking"

    return "classify"


def build_graph():
    builder = StateGraph(VoiceCartState)

    builder.add_node("classify", classify_node)
    builder.add_node("inquiry", inquiry_node)
    builder.add_node("empathetic", empathetic_response_node)
    builder.add_node("escalate", escalate_handoff_node)
    builder.add_node("escalation_followup", escalation_followup_node)
    builder.add_node("other", other_node)
    builder.add_node("check_stock", check_stock_node)
    builder.add_node("post_escalation", post_escalation_node)
    builder.add_node("warranty_claim", warranty_claim_node)
    builder.add_node("warranty_email", warranty_email_node)
    builder.add_node("order_booking", order_booking_node)
    builder.add_node("order_tracking", order_tracking_node)
    builder.add_node("order_cancel", order_cancel_node)

    builder.add_conditional_edges(
        START,
        route_at_start,
        {
            "post_escalation": "post_escalation",
            "escalation_followup": "escalation_followup",
            "warranty_email": "warranty_email",
            "warranty_claim": "warranty_claim",
            "order_booking": "order_booking",
            "classify": "classify",
        },
    )

    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "inquiry": "inquiry",
            "empathetic": "empathetic",
            "check_stock": "check_stock",
            "warranty_claim": "warranty_claim",
            "order_booking": "order_booking",
            "order_tracking": "order_tracking",
            "order_cancel": "order_cancel",
            "other": "other",
        },
    )

    builder.add_conditional_edges(
        "empathetic",
        route_after_empathetic,
        {"escalate": "escalate", "end": END},
    )

    builder.add_edge("inquiry", END)
    builder.add_edge("escalate", END)
    builder.add_edge("escalation_followup", END)
    builder.add_edge("other", END)
    builder.add_edge("check_stock", END)
    builder.add_conditional_edges(
        "warranty_claim",
        lambda state: "escalate" if state.get("escalate") else "end",
        {"escalate": "escalate", "end": END},
    )
    builder.add_edge("warranty_email", END)
    builder.add_edge("post_escalation", END)
    builder.add_edge("order_booking", END)
    builder.add_edge("order_tracking", END)
    builder.add_edge("order_cancel", END)

    return builder.compile()


graph = build_graph()


if __name__ == "__main__":
    test_conversations = [
        # Escalation with email consent YES
        [
            "One of my earbuds stopped working completely, it's so annoying",
            "I already tried that, it's still not working, this is so frustrating",
            "This is the third time, I'm really annoyed now",
            "yes",
            "jamshed@example.com",
        ],
        # Order booking, then cancel it (still processing -> should succeed)
        [
            "I want to buy 2 red ones, ship to 111 First St, email cancel1@example.com",
        ],
        # Stock check via MongoDB
        ["Is the matte black one in stock?"],
    ]

    for i, turns in enumerate(test_conversations):
        print(f"\n{'=' * 60}")
        print(f"CONVERSATION {i + 1}")
        print(f"{'=' * 60}")

        state = {"messages": [], "sentiment_history": [], "turn_count": 0}

        for turn in turns:
            state["messages"].append({"role": "user", "content": turn})
            print(f"\nUser: {turn}")
            state = graph.invoke(state)
            print(f"Assistant: {state.get('response')}")

        print(f"\n[final state] intent={state.get('intent')}, order_booking={state.get('order_booking')}")