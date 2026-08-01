"""
LangGraph nodes for VoiceCart's customer support agent.
"""

import os
import sys
import re
import json
from typing import Optional
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime, timezone
from langsmith import traceable

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
from retriever import VoiceCartRetriever

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db import inventory_collection, tickets_collection, warranty_claims_collection, orders_collection
from email_service import send_support_email, send_customer_email
from slack_service import notify_support_ticket, notify_warranty_claim, notify_new_order, notify_order_cancelled

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)
retriever = VoiceCartRetriever(alpha=0.75)

SCORE_THRESHOLD = 0.25
EMPATHETIC_SCORE_THRESHOLD = 0.35
PRODUCT_NAME = "ShopNest Pulse"
VALID_NAMESPACES = {"product-info", "usage-guidance", "troubleshooting", "policies", "limitations"}


EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def get_recent_history(messages: list[dict], limit: int = 10) -> list[dict]:
    return messages[-limit:]


def format_history_for_prompt(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        speaker = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def _get_valid_colors() -> list[str]:
    """Fetches the current list of valid colors from MongoDB inventory."""
    docs = inventory_collection.find({"product": PRODUCT_NAME})
    return [d["color"] for d in docs]


def _interpret_yes_no(message: str) -> Optional[bool]:
    """Interprets a yes/no answer. Fast keyword path, LLM fallback for ambiguous replies."""
    text = message.strip().lower().rstrip(".!")
    yes_words = {"yes", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead", "yup", "affirmative", "please do"}
    no_words = {"no", "nope", "nah", "not now", "don't", "do not", "skip", "no thanks"}

    if text in yes_words:
        return True
    if text in no_words:
        return False

    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": (
                f'A user was asked a yes/no question. Their reply was: "{message}". '
                f'Does this reply clearly mean "yes" or clearly mean "no"? '
                f'If the reply is a completely different question, an unrelated topic, or doesn\'t '
                f'actually answer yes or no at all, respond "unclear" — do NOT guess "no" just because '
                f'it isn\'t a "yes". Respond with exactly one word: yes, no, or unclear.'
            ),
        }],
        model="llama-3.1-8b-instant",
        temperature=0,
    )
    answer = response.choices[0].message.content.strip().lower()
    if re.search(r"\byes\b", answer):
        return True
    if re.search(r"\bno\b", answer):
        return False
    return None


def _normalize_spoken_email(message: str) -> str:
    """
    Converts common spoken-email patterns (from voice transcription) into
    real email syntax before regex extraction. E.g. "jamshed at gmail dot
    com" -> "jamshed@gmail.com". Safe to run on typed text too, since it
    only fires on the specific "word at word dot word" pattern.
    """
    text = message

    # Whisper sometimes transcribes a spoken "@" as a literal @ symbol but
    # with spaces around it (e.g. "jamshed @ gmail.com") — collapse that
    # first, since EMAIL_REGEX requires no whitespace around the @.
    text = re.sub(r'\s*@\s*', '@', text)

    # "word at word dot/." -> word@word.word
    pattern = re.compile(
        r'\b([a-zA-Z0-9._%+-]+)\s+at\s+([a-zA-Z0-9-]+)(?:\s+dot\s+|\.)([a-zA-Z]{2,})\b',
        re.IGNORECASE,
    )
    text = pattern.sub(lambda m: f"{m.group(1)}@{m.group(2)}.{m.group(3)}", text)

    # Standalone " at " / " dot " inside something that already looks
    # like it's forming an email attempt (has an @ or a recognizable
    # domain word nearby) — normalize remaining loose "dot"/"at" words.
    text = re.sub(r'\s+at\s+', '@', text) if '@' not in text and re.search(r'\bat\b.*(?:\bdot\b|\.[a-zA-Z]{2,})', text, re.IGNORECASE) else text
    text = re.sub(r'\s+dot\s+', '.', text, flags=re.IGNORECASE)

    return text


def _extract_email(message: str) -> Optional[str]:
    """Extracts an email address via regex — deterministic, no LLM guessing.
    Normalizes spoken patterns (e.g. 'at'/'dot') first, since voice
    transcription doesn't produce literal @ and . symbols."""
    normalized = _normalize_spoken_email(message)
    match = EMAIL_REGEX.search(normalized)
    return match.group(0) if match else None


# ---------------------------------------------------------------------
# classify_node
# ---------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = """You are a classifier for a customer support voice agent for ShopNest Pulse wireless earbuds.

You will be given the recent conversation history, ending with the user's latest message. Use the history to correctly interpret short or ambiguous replies (e.g. "ok", "no", "yes") in context — classify based on what the latest message means GIVEN the conversation so far, not in isolation.

Classify the LATEST user message into exactly one intent, one sentiment, and (if intent is "inquiry" or "stock_check") supporting fields.

INTENT options:
- "inquiry": any GENERAL QUESTION answerable from product knowledge — specs, setup, shipping, returns, payment, how something works, recommendations, or guidance requests. This does NOT include a message where the user reports their OWN specific product problem and/or explicitly wants to file a warranty claim for it — that is always "complaint", even if the word "warranty" appears in the message.
- "complaint": user reports a problem, defect, or dissatisfaction with THEIR OWN product or order — including messages that describe a specific issue (e.g. "right earbud dead") AND/OR explicitly ask to file/claim a warranty in the same message (e.g. "I want to claim warranty, right earbud dead, bought 2 weeks ago"). Any message reporting a personal defect combined with wanting warranty coverage is ALWAYS "complaint", never "inquiry" — regardless of whether ownership duration or other warranty-policy-sounding details are also mentioned in the same sentence.
- "stock_check": user is asking about color/stock availability — a SPECIFIC color ("is red in stock?"), a general "what colors do you have" question, OR a bulk quantity question about current stock levels ("how many of each do you have", "what's currently available", "quantity of each")
- "order_booking": user wants to buy/order/purchase the earbuds (e.g. "I want to buy this", "I'd like to order the black one", "can I purchase these")
- "order_tracking": user is asking about the status of an existing order (e.g. "where's my order", "track my order", "what's the status of order ORD-...")
- "order_cancel": user wants to cancel an existing order (e.g. "cancel my order", "cancel order ORD-...", "I want to cancel this order")
- "other": anything that doesn't fit the above (greetings, casual chat, stating their name, thanks, unclear requests, off-topic, or a short acknowledgment like "ok"/"no"/"good" that doesn't need a namespace or color lookup). This INCLUDES clearly off-topic requests unrelated to ShopNest Pulse earbuds at all (e.g. "recommend a podcast", "what's the weather") — do NOT classify these as "inquiry", since there is no product knowledge to retrieve for them. This ALSO INCLUDES the user asking about THEIR OWN information, status, or conversation history already established earlier in this session — e.g. "what's my name", "what did I tell you earlier", "have you sent my issue to your team", "did you escalate this", "did you email them", "what's my order status" (when no order ID is given, this is order_tracking instead — but a vague status check about escalation/email specifically is "other"). None of these are product-knowledge inquiries needing retrieval — they must use intent "other" with target_namespace null, never "inquiry".

SENTIMENT options:
- "neutral": calm, straightforward tone
- "frustrated": mild annoyance, disappointment, or impatience
- "angry": strong negative emotion, harsh language, demands for refund/escalation, repeated complaints

TARGET_NAMESPACE options (only required if intent is "inquiry" or a fixable "complaint", otherwise use null):
- "product-info": specs, what's in the box, pairing, touch controls, overview, colors, pricing, package weight, firmware, voice assistant compatibility
- "usage-guidance": ANC/transparency explanation, fit/comfort, usage by activity (calls/gym/gaming/commute), battery aging, buying advice, recommendations, color choice
- "troubleshooting": something isn't working right and the user wants a fix (won't pair, won't charge, sound cutting out, falling out during exercise, echo on calls, one-sided volume, LED not working, overheating, voice assistant not responding, audio delay, stuck pairing mode, case lid not closing)
- "policies": shipping, returns, refunds, warranty, payment methods, COD, order cancellation, order tracking, warranty claim process, international shipping, bulk discounts, gift orders
- "limitations": questions OR complaints about something the product explicitly cannot do or a known expected behavior (full silence, swimming, competitive gaming, battery capacity naturally decreasing over time/years of use, drop damage, loss/theft, hearing protection, dust exposure, multipoint device limit, wireless charging, wired listening, voice assistant needing phone signal, extreme temperatures, case battery aging separately) — use this whenever the complaint matches a known limitation, even if warranty_eligible is false

SECONDARY_NAMESPACE (only if intent is "inquiry" AND the user's message clearly asks about a SECOND distinct topic from a different namespace in the same message, e.g. "what's your return policy and also how do I pair these" — otherwise use null):
- Use the same namespace options as above for whichever second topic is present.
- Do NOT set this just because a question is broad — only when there are genuinely two separate questions bundled into one message.

COLOR (only required if intent is "stock_check" or "order_booking", otherwise use null):
- Extract the color the user is asking about.
- Valid colors: {valid_colors}
- If the user's wording doesn't clearly map to one of these colors, use null.

QUANTITY (only relevant if intent is "order_booking", otherwise use null):
- Extract the number of units the user wants to order, as an integer. Default to null if not mentioned (will be asked separately).

ORDER_ID (only relevant if intent is "order_tracking" or "order_cancel", otherwise use null):
- Extract the order ID if the user provides one (format looks like "ORD-XXXXXXXX-XXXXXX"). Use null if not mentioned.

WARRANTY_ELIGIBLE (only relevant if intent is "complaint", otherwise use false):
- true if the user EXPLICITLY asks to start/file a warranty claim in THIS message OR earlier in the conversation history if a claim collection is clearly already in progress — using clear phrasing like "warranty claim," "file a claim," "claim my warranty," "I want to claim warranty," "can I get this warrantied." A plain description of a problem, no matter how severe it sounds ("won't turn on," "completely dead," "stopped working"), is NOT enough on its own — the user must explicitly ask for the warranty process by name.
- Once true, stay true — do NOT second-guess or reroute based on how old the product sounds, how it broke, or whether it resembles a known limitation. Ownership age and coverage validity are checked separately and deterministically after details are collected, not by you. Your only job here is detecting the explicit request itself.
- NEVER set this to false just because an EARLIER warranty claim in this conversation was declined (e.g. for being out of the 12-month window) — a new explicit request is always a fresh, independent attempt (possibly for a different product or issue) and must be given true here. The actual 12-month eligibility check happens later, deterministically, once real ownership-duration details are collected — that is NOT your job to pre-judge from conversation history.
- In every other case — including severe-sounding complaints without an explicit warranty request — set this to false. Route to normal troubleshooting/empathetic handling instead.

ISSUE_TOPIC (only relevant if intent is "complaint", otherwise use null):
- Write a short 2-5 word tag identifying the SPECIFIC problem being reported, e.g. "earbuds falling out", "echo on calls", "case not charging", "won't pair", "battery draining fast". This should be specific enough to distinguish this problem from a different problem in the same namespace.

SAME_ISSUE_AS_BEFORE (only relevant if intent is "complaint", otherwise use false):
- Compare the CURRENT complaint's issue against the PREVIOUS ISSUE TOPIC given below. Set this to true ONLY if the user is still describing the SAME underlying problem (e.g. following up on the same unresolved issue, even if worded differently) — set to false if this is a NEW, different problem, even if it's in the same general namespace (e.g. "falling out" then "echo on calls" are DIFFERENT issues, both troubleshooting, but not the same issue).
- PREVIOUS ISSUE TOPIC: {previous_issue_topic}

EXHAUSTED_TROUBLESHOOTING (relevant for any message, otherwise use false):
- true if the user signals they've already tried everything and it isn't working — e.g. "tried everything, nothing works", "still not working after all that", "this isn't helping", "I've done all of that already", "nothing is fixing it". This is different from an explicit escalation request — the user isn't asking to talk to a human directly, they're expressing that troubleshooting has genuinely failed.
- false otherwise, including a first-time complaint or a complaint where the user hasn't indicated that prior attempts failed.

USER_NAME (extract whenever the user states or mentions their own name, e.g. "my name is Jamshed", "I'm Jamshed", "this is Jamshed" — otherwise use null):
- Only extract a name if the user is clearly telling you THEIR OWN name.
- Use null if no name is stated in this message.

Note: if intent is "complaint", target_namespace should be "troubleshooting" if the complaint sounds like a fixable technical issue, or "limitations" if it relates to a known product limitation (including normal wear like battery aging after extended use — this is a limitations topic, NOT a fixable issue), or null only if neither applies (e.g. a refund demand with no technical detail). A non-warranty-eligible complaint should almost always map to "limitations" since it's describing expected product behavior, not a fixable defect.

EXPLICIT_ESCALATION_REQUEST (relevant for any message, otherwise use false):
- true ONLY if the user explicitly asks to escalate, talk to a human, speak to a real person, or get connected to support in general — e.g. "escalate this", "escalate the issue", "I want to talk to a human", "get me support", "send this to your team", "connect me with someone".
- false if the user mentions a WARRANTY CLAIM specifically — e.g. "file a warranty claim for this", "I want to claim warranty for that issue", "claim warranty for this" — even though this also involves getting help from the team. Warranty requests are a completely separate, dedicated flow (see WARRANTY_ELIGIBLE below) and must NEVER be treated as a generic escalation request, regardless of phrasing similarity.
- false otherwise, including ordinary complaints that don't explicitly ask for escalation/a human.

Respond ONLY with valid JSON in this exact format, no other text:
{{"intent": "...", "sentiment": "...", "target_namespace": "..." or null, "secondary_namespace": "..." or null, "color": "..." or null, "quantity": 0 or null, "order_id": "..." or null, "warranty_eligible": true or false, "issue_topic": "..." or null, "same_issue_as_before": true or false, "explicit_escalation_request": true or false, "exhausted_troubleshooting": true or false, "user_name": "..." or null, "reason": "brief one-sentence explanation"}}
"""


@traceable(name="classify_node")
def classify_node(state: dict) -> dict:
    messages = state["messages"]
    history = get_recent_history(messages, limit=10)
    history_text = format_history_for_prompt(history)
    valid_colors_str = ", ".join(f'"{c}"' for c in _get_valid_colors())
    previous_issue_topic = state.get("last_issue_topic") or "none yet"
    system_prompt = CLASSIFY_SYSTEM_PROMPT.format(
        valid_colors=valid_colors_str,
        previous_issue_topic=previous_issue_topic,
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONVERSATION HISTORY:\n{history_text}\n\nClassify the LATEST user message above."},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
        response_format={"type": "json_object"},
    )

    parsed = json.loads(response.choices[0].message.content)

    intent = parsed.get("intent", "other")
    sentiment = parsed.get("sentiment", "neutral")
    target_namespace = parsed.get("target_namespace")
    secondary_namespace = parsed.get("secondary_namespace")
    color = parsed.get("color")
    quantity = parsed.get("quantity")
    order_id_mentioned = parsed.get("order_id")
    warranty_eligible = parsed.get("warranty_eligible", False)
    issue_topic = parsed.get("issue_topic")
    same_issue_as_before = parsed.get("same_issue_as_before", False)
    explicit_escalation_request = parsed.get("explicit_escalation_request", False)
    exhausted_troubleshooting = parsed.get("exhausted_troubleshooting", False)
    extracted_name = parsed.get("user_name")
    reason = parsed.get("reason", "")

    state["intent"] = intent
    state["sentiment"] = sentiment
    state["target_namespace"] = target_namespace
    state["secondary_namespace"] = secondary_namespace
    state["color"] = color
    state["quantity"] = quantity
    state["order_id_mentioned"] = order_id_mentioned
    state["warranty_eligible"] = warranty_eligible
    state["explicit_escalation_request"] = explicit_escalation_request
    state["exhausted_troubleshooting"] = exhausted_troubleshooting
    state["sentiment_history"] = state.get("sentiment_history", []) + [sentiment]
    state["turn_count"] = state.get("turn_count", 0) + 1
    state["is_handoff_turn"] = False

    if extracted_name:
        state["user_name"] = extracted_name

    # Track a same-issue streak based on the LLM's explicit same-issue
    # judgment (not just namespace continuity, which was too coarse —
    # different problems in the same namespace were wrongly counted as
    # the same unresolved issue).
    negative = {"frustrated", "angry"}
    if intent == "complaint" and sentiment in negative:
        if same_issue_as_before:
            state["same_issue_streak"] = state.get("same_issue_streak", 0) + 1
        else:
            state["same_issue_streak"] = 1
        if issue_topic:
            state["last_issue_topic"] = issue_topic
    elif intent == "complaint":
        state["same_issue_streak"] = 0
        if issue_topic:
            state["last_issue_topic"] = issue_topic
    elif intent == "other":
        # Short acknowledgments/small talk don't mean the user abandoned
        # the issue — leave the streak untouched rather than resetting it.
        pass
    else:
        # A genuinely different substantive intent (inquiry, order, stock
        # check, etc.) means the user has moved on — reset the streak.
        state["same_issue_streak"] = 0

    print(f"[classify_node] intent={intent} sentiment={sentiment} namespace={target_namespace} secondary_namespace={secondary_namespace} color={color} quantity={quantity} order_id_mentioned={order_id_mentioned} warranty_eligible={warranty_eligible} issue_topic={issue_topic} same_issue_as_before={same_issue_as_before} same_issue_streak={state.get('same_issue_streak', 0)} user_name={state.get('user_name')} reason={reason}")

    return state


# ---------------------------------------------------------------------
# inquiry_node
# ---------------------------------------------------------------------

GROUNDED_ANSWER_PROMPT = """You are a customer support voice agent for ShopNest Pulse wireless earbuds.

Recent conversation history (for context on follow-up questions):
{history}

Answer the user's LATEST question using ONLY the context provided below. Do not use any outside knowledge or make up information not present in the context.

If the context clearly states a fact, limit, or number (e.g. a maximum device count, a price, a policy detail), state it plainly and confidently — do NOT hedge with phrases like "I'm not sure if" or "I don't have information about" when the context directly answers the question. Only express uncertainty when the context genuinely does not address the question.

If the context doesn't fully answer the question, say what you do know from the context and acknowledge what you're unsure about — do not guess or invent details.

Keep your answer conversational and concise, since this will be spoken aloud to the user. Avoid bullet points or lists; speak in natural sentences.

CONTEXT:
{context}

USER'S LATEST QUESTION:
{question}
"""

NO_ANSWER_RESPONSE = (
    "I don't have specific information about that in what I can check right now. "
    "I'd recommend reaching out to our support team directly so they can look into it for you."
)


def _retrieve_context(user_question: str, namespace: str) -> tuple[Optional[str], float]:
    """Shared retrieval helper: returns (context_string_or_None, top_score)."""
    if namespace not in VALID_NAMESPACES:
        print(f"[_retrieve_context] Invalid namespace '{namespace}' from classifier, treating as no context")
        return None, 0.0
        state["warranty_claim_boundary_index"] = len(state["messages"])
    results = retriever.retrieve(user_question, namespace, top_k=5)
    if not results or results[0]["score"] < SCORE_THRESHOLD:
        return None, (results[0]["score"] if results else 0.0)
    context = "\n\n".join(
        f"[{r['title']}]: {r['text']}" for r in results if r["score"] >= SCORE_THRESHOLD * 0.7
    )
    return context, results[0]["score"]


@traceable(name="inquiry_node")
def inquiry_node(state: dict) -> dict:
    messages = state["messages"]
    user_question = messages[-1]["content"]
    namespace = state.get("target_namespace")
    secondary_namespace = state.get("secondary_namespace")
    history = get_recent_history(messages, limit=10)
    history_text = format_history_for_prompt(history)

    if not namespace:
        state["response"] = NO_ANSWER_RESPONSE
        state["retrieval_score"] = None
        print(f"[inquiry_node] No namespace provided, returning fallback")
        return state

    context, top_score = _retrieve_context(user_question, namespace)

    if context is None:
        state["response"] = NO_ANSWER_RESPONSE
        state["retrieval_score"] = top_score
        print(f"[inquiry_node] Top score {top_score:.4f} below threshold {SCORE_THRESHOLD}, returning fallback")
        return state

    # If the message bundled a second distinct topic, retrieve that too and
    # combine both contexts so the answer can address both parts at once.
    if secondary_namespace and secondary_namespace != namespace:
        secondary_context, _ = _retrieve_context(user_question, secondary_namespace)
        if secondary_context:
            context = context + "\n\n" + secondary_context
            print(f"[inquiry_node] Combined context from secondary namespace '{secondary_namespace}'")

    prompt = GROUNDED_ANSWER_PROMPT.format(context=context, question=user_question, history=history_text)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    state["response"] = answer
    state["retrieval_score"] = top_score

    print(f"[inquiry_node] Top score: {top_score:.4f}")
    print(f"[inquiry_node] Response: {answer}")

    return state


# ---------------------------------------------------------------------
# empathetic_response_node
# ---------------------------------------------------------------------

EMPATHETIC_ANSWER_PROMPT = """You are a customer support voice agent for ShopNest Pulse wireless earbuds.

Recent conversation history (for context):
{history}

The user has reported a problem and sounds {sentiment}. Respond with brief, genuine empathy first — acknowledge their frustration in one short sentence, do not over-apologize or grovel — then clearly explain the fix using ONLY the context provided below. Do not use any outside knowledge or invent details not in the context.

If the context clearly states a fact or limit, state it plainly and confidently rather than hedging.

IMPORTANT: Only reference "earlier" troubleshooting steps if the conversation history above ACTUALLY shows you suggesting them for THIS SAME issue. If this is the first time discussing this specific issue, do not claim anything was "already discussed" or "already tried" — just give the troubleshooting steps directly. Never fabricate a reference to a prior exchange that isn't in the history shown above.

Keep the whole response conversational and concise since it will be spoken aloud. Avoid bullet points; speak in natural sentences. Do not repeat "I'm sorry" more than once.

If the context doesn't address their specific issue, acknowledge that honestly rather than guessing.

CONTEXT:
{context}

USER'S LATEST MESSAGE:
{question}
"""

NO_CONTEXT_EMPATHETIC_RESPONSE = (
    "I hear you, and I'm sorry you're running into this. I don't have specific "
    "troubleshooting steps for that exact issue right now. Would you like me to "
    "send this over to our support team so they can take a closer look?"
)


@traceable(name="empathetic_response_node")
def empathetic_response_node(state: dict) -> dict:
    state["escalate"] = False

    messages = state["messages"]
    user_message = messages[-1]["content"]
    namespace = state.get("target_namespace")
    sentiment = state.get("sentiment", "frustrated")
    history = get_recent_history(messages, limit=10)
    history_text = format_history_for_prompt(history)

    # If the user has signaled that troubleshooting has already failed,
    # don't retrieve again or push more steps at them — ask directly
    # whether they'd like this escalated, and let the user decide.
    if state.get("exhausted_troubleshooting"):
        state["response"] = (
            "I'm sorry the troubleshooting steps haven't resolved this. Would you like me to "
            "send this over to our support team so they can take a closer look?"
        )
        state["offered_manual_escalation"] = True
        print("[empathetic_response_node] User signaled exhausted troubleshooting, offering escalation")
        return state

    if not namespace:
        state["response"] = NO_CONTEXT_EMPATHETIC_RESPONSE
        state["retrieval_score"] = None
        state["offered_manual_escalation"] = True
        print(f"[empathetic_response_node] No namespace, using generic empathetic fallback, offering escalation")
        return state

    context, top_score = _retrieve_context(user_message, namespace)

    print(f"[empathetic_response_node] Retrieval score for '{user_message}': {top_score:.4f}")

    # Troubleshooting needs a HIGHER confidence bar than general Q&A —
    # a weak, loosely-related match here produces a hedging, half-made-up
    # response instead of real steps, which is worse than honestly saying
    # "I don't have this" and offering to escalate.
    if context is None or top_score < EMPATHETIC_SCORE_THRESHOLD:
        state["response"] = NO_CONTEXT_EMPATHETIC_RESPONSE
        state["retrieval_score"] = top_score
        state["offered_manual_escalation"] = True
        print(f"[empathetic_response_node] Score {top_score:.4f} below empathetic threshold {EMPATHETIC_SCORE_THRESHOLD}, using generic fallback, offering escalation")
        return state

    prompt = EMPATHETIC_ANSWER_PROMPT.format(
        context=context, question=user_message, sentiment=sentiment, history=history_text
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    state["response"] = answer
    state["retrieval_score"] = top_score

    print(f"[empathetic_response_node] Response: {answer}")

    return state


# ---------------------------------------------------------------------
# Escalation: ticket prep, email consent flow, finalization
# ---------------------------------------------------------------------

SUPPORT_TEAM_EMAIL = "support@shopnest.com"
SUPPORT_RESPONSE_TIME = "within 24 hours"

HANDOFF_MESSAGE = (
    f"I've sent your issue over to our support team, and they'll have the full "
    f"details of our conversation so you won't need to repeat yourself. "
    f"You can expect to hear back {SUPPORT_RESPONSE_TIME}."
)

TICKET_SUMMARY_PROMPT = """Summarize this customer support conversation in 2-3 factual sentences for an internal support ticket. Mention the core issue and what troubleshooting (if any) was already attempted.

CONVERSATION:
{conversation}
"""


def _generate_summary(messages: list[dict]) -> str:
    convo_text = format_history_for_prompt(get_recent_history(messages, limit=10))
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": TICKET_SUMMARY_PROMPT.format(conversation=convo_text)}],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def _build_ticket_email_html(ticket: dict, customer_email: str, summary: str) -> str:
    convo_lines = "".join(
        f"<p><strong>{m['role'].title()}:</strong> {m['content']}</p>"
        for m in ticket.get("conversation", [])[-10:]
    )
    return f"""
    <h2>New Support Ticket — {ticket['ticket_id']}</h2>
    <p><strong>Customer email:</strong> {customer_email}</p>
    <p><strong>Submitted:</strong> {ticket['timestamp']}</p>
    <h3>Summary</h3>
    <p>{summary}</p>
    <h3>Recent Conversation</h3>
    {convo_lines}
    """


@traceable(name="escalate_handoff_node")
def escalate_handoff_node(state: dict) -> dict:
    """
    Entry point when the same issue has gone unresolved for 3 consecutive
    negative turns. Prepares the ticket (not yet saved) and asks whether
    the customer wants their details emailed to the support team.
    """
    messages = state.get("messages", [])
    sentiment_history = state.get("sentiment_history", [])

    timestamp = datetime.now(timezone.utc)
    ticket_id = f"ticket_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    pending_ticket = {
        "ticket_id": ticket_id,
        "timestamp": timestamp.isoformat(),
        "turn_count": state.get("turn_count", 0),
        "sentiment_history": sentiment_history,
        "conversation": messages,
        "last_intent": state.get("intent"),
        "last_namespace": state.get("target_namespace"),
    }

    state["_pending_ticket"] = pending_ticket
    state["escalation_email"] = {"status": "collecting_email", "ticket_id": ticket_id, "email": None}
    state["is_handoff_turn"] = True
    state["response"] = (
        "I'd like to get this over to our support team so they can take a closer look. "
        "To make sure they can follow up, could you share your email address?"
    )

    print(f"[escalate_handoff_node] Prepared ticket {ticket_id}, awaiting email consent")
    return state


@traceable(name="manual_escalation_followup_node")
def manual_escalation_followup_node(state: dict) -> dict:
    """
    Handles the reply after being asked whether to escalate. A clear
    "yes" triggers real escalation. A clear "no" cancels the offer. Any
    off-topic message (a genuine new question) gets a real answer via
    classify_and_retrieve, and the escalation offer STAYS PENDING so
    the user can still say yes/no afterward instead of losing it.
    """
    user_message = state["messages"][-1]["content"]
    wants_escalation = _interpret_yes_no(user_message)

    if wants_escalation is True:
        state["offered_manual_escalation"] = False
        return escalate_handoff_node(state)

    if wants_escalation is False:
        state["offered_manual_escalation"] = False
        state["response"] = "No problem — let me know if there's anything else I can help with."
        return state

    # Ambiguous/off-topic reply — answer it for real, keep the offer alive.
    state_after_classify = classify_and_retrieve(state)
    intent = state_after_classify.get("intent")

    if intent == "inquiry":
        answer_state = inquiry_node(state_after_classify)
    elif intent == "stock_check":
        answer_state = check_stock_node(state_after_classify)
    else:
        answer_state = other_node(state_after_classify)

    state.update(answer_state)
    state["offered_manual_escalation"] = True
    state["response"] = state["response"] + " Would you still like me to send your earlier issue over to our support team?"
    print("[manual_escalation_followup_node] Answered off-topic question, kept escalation offer pending")
    return state


def _finalize_ticket(state: dict, customer_email: Optional[str]) -> None:
    pending_ticket = state.pop("_pending_ticket", None) or {
        "ticket_id": f"ticket_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation": state.get("messages", []),
        "sentiment_history": state.get("sentiment_history", []),
    }

    pending_ticket["customer_email"] = customer_email
    pending_ticket["assigned_to"] = SUPPORT_TEAM_EMAIL
    pending_ticket["expected_response_time"] = SUPPORT_RESPONSE_TIME
    pending_ticket["email_sent"] = False

    summary = _generate_summary(pending_ticket["conversation"])
    pending_ticket["summary"] = summary

    if customer_email:
        html = _build_ticket_email_html(pending_ticket, customer_email, summary)
        try:
            send_support_email(subject=f"[VoiceCart Support] Ticket {pending_ticket['ticket_id']}", html_body=html)
            pending_ticket["email_sent"] = True
        except Exception as e:
            print(f"[_finalize_ticket] Email send failed: {e}")

    notify_support_ticket(pending_ticket["ticket_id"], summary, customer_email)

    try:
        tickets_collection.insert_one(dict(pending_ticket))
    except Exception as e:
        print(f"[_finalize_ticket] MongoDB insert failed: {e}")

    state["ticket_id"] = pending_ticket["ticket_id"]
    state["escalated"] = True
    state["is_handoff_turn"] = True

    if customer_email:
        state["response"] = (
            f"Done — I've sent your details to our support team, with a copy noted for {customer_email}. "
            f"You can expect to hear back {SUPPORT_RESPONSE_TIME}."
        )
    else:
        state["response"] = HANDOFF_MESSAGE


@traceable(name="escalation_followup_node")
def escalation_followup_node(state: dict) -> dict:
    """
    Collects the customer's email for an escalated ticket. Email is
    OPTIONAL here (unlike warranty claims) — if the user provides one,
    we finalize with it. If the user explicitly declines, we back out
    of escalation gracefully and immediately, rather than insisting.
    """
    escalation_email = state.get("escalation_email") or {}
    user_message = state["messages"][-1]["content"]

    email_found = _extract_email(user_message)

    if email_found:
        escalation_email["email"] = email_found
        escalation_email["status"] = "sent"
        state["escalation_email"] = escalation_email
        _finalize_ticket(state, customer_email=email_found)
        return state

    declined = _interpret_yes_no(user_message)
    if declined is False:
        escalation_email["status"] = "declined"
        state["escalation_email"] = escalation_email
        _finalize_ticket(state, customer_email=None)
        state["response"] = (
            "No worries — I've sent your issue over to our support team without an email on file. "
            f"You can expect to hear back {SUPPORT_RESPONSE_TIME}."
        )
        return state

    state["response"] = "I couldn't catch a valid email address — could you type it again, or let me know if you'd rather skip it?"
    return state


POST_ESCALATION_RESPONSE = (
    "Your conversation has already been sent to our support team, and they'll be in touch soon. "
    "Is there anything else I can help you with in the meantime?"
)


def post_escalation_node(state: dict) -> dict:
    state["response"] = POST_ESCALATION_RESPONSE
    state["is_handoff_turn"] = False
    state["escalation_acknowledged"] = True
    print("[post_escalation_node] Acknowledged escalation once, conversation will resume normally after this")
    return state


# ---------------------------------------------------------------------
# other_node
# ---------------------------------------------------------------------

OTHER_NODE_SYSTEM_PROMPT = """You are a customer support voice agent for ShopNest Pulse wireless earbuds, named the ShopNest support assistant.

The user's latest message does not require a product lookup, complaint handling, or stock check — it's a greeting, casual remark, acknowledgment ("ok", "good", "thanks"), or the user stating their own name.

{name_context}

{escalation_status}

Recent conversation history:
{history}

Instructions:
- If the user is greeting you or making small talk, respond warmly and briefly, and naturally invite them to ask about ShopNest Pulse earbuds (setup, specs, policies, or issues) — but don't repeat this invitation if you've already said it earlier in the history.
- If the user just told you their name, acknowledge it naturally and use it if it fits.
- If the user's message is genuinely OUT OF SCOPE — explicitly say you don't have information outside of ShopNest Pulse support. Do NOT attempt to answer the out-of-scope question.
- CRITICAL: NEVER claim that an issue has been escalated, that an email has been sent, or that a support ticket has been created unless the ESCALATION STATUS above explicitly confirms this already happened. If the user asks whether something was escalated/emailed and it has NOT, say honestly that it hasn't happened yet — do not invent or assume this occurred just because the conversation history sounds frustrated.

USER'S LATEST MESSAGE:
{message}
"""


@traceable(name="other_node")
def other_node(state: dict) -> dict:
    messages = state["messages"]
    user_message = messages[-1]["content"]
    history = get_recent_history(messages, limit=10)
    history_text = format_history_for_prompt(history)
    user_name = state.get("user_name")

    name_context = (
        f"The user's name is {user_name}. You may address them by name where natural."
        if user_name
        else "You don't know the user's name yet."
    )

    if state.get("escalated"):
        escalation_status = f"ESCALATION STATUS: A support ticket (ID: {state.get('ticket_id')}) HAS already been created and sent to the support team."
    else:
        escalation_status = "ESCALATION STATUS: No support ticket has been created and nothing has been escalated or emailed in this conversation yet."

    prompt = OTHER_NODE_SYSTEM_PROMPT.format(name_context=name_context, escalation_status=escalation_status, history=history_text, message=user_message)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.4,
    )

    state["response"] = response.choices[0].message.content
    print(f"[other_node] response={state['response']}")
    return state


# ---------------------------------------------------------------------
# Tool-calling: stock check (now backed by MongoDB)
# ---------------------------------------------------------------------

def check_stock(color: str) -> dict:
    doc = inventory_collection.find_one({"product": PRODUCT_NAME, "color": color})
    if not doc:
        return {"color": color, "found": False, "in_stock": None, "quantity": None}
    quantity = doc.get("quantity", 0)
    return {"color": color, "found": True, "in_stock": quantity > 0, "quantity": quantity}


STOCK_RESPONSE_PROMPT = """You are a customer support voice agent for ShopNest Pulse wireless earbuds.

You just checked stock availability for a specific color. Phrase a brief, natural, conversational response based on the result below. If in stock, you may mention roughly how many are available. If out of stock, you may suggest alternatives — but ONLY from the full inventory list below.

REQUESTED COLOR RESULT:
{result}

FULL INVENTORY (the only colors that exist — do not invent others):
{full_inventory}

Keep your response short since it will be spoken aloud.
"""

def _no_color_detected_response() -> str:
    colors = _get_valid_colors()
    colors_str = ", ".join(colors[:-1]) + f", or {colors[-1]}" if len(colors) > 1 else colors[0]
    return f"I can check stock for our {colors_str} colors — which one are you interested in?"


@traceable(name="check_stock_node")
def check_stock_node(state: dict) -> dict:
    color = state.get("color")

    if not color:
        # Check whether this is a "show me all quantities" request vs
        # a color-specific question with no color given yet.
        all_docs = list(inventory_collection.find({"product": PRODUCT_NAME}))
        user_message = state["messages"][-1]["content"].lower()
        quantity_keywords = ["quantity", "quantities", "how many", "how much", "each", "all", "available", "stock of each", "inventory"]

        if any(kw in user_message for kw in quantity_keywords):
            lines = []
            for d in all_docs:
                qty = d.get("quantity", 0)
                if qty > 0:
                    lines.append(f"{d['color']}: {qty} available")
                else:
                    lines.append(f"{d['color']}: out of stock")
            inventory_summary = ", ".join(lines)
            state["response"] = f"Here's our current stock across all colors — {inventory_summary}."
            print("[check_stock_node] Full inventory summary requested, returning all quantities")
            return state

        state["response"] = _no_color_detected_response()
        print("[check_stock_node] No color detected, asking for clarification")
        return state

    result = check_stock(color)
    print(f"[check_stock_node] Tool call check_stock('{color}') -> {result}")

    all_docs = list(inventory_collection.find({"product": PRODUCT_NAME}))

    if not result["found"]:
        available_colors = ", ".join(d["color"] for d in all_docs)
        state["response"] = f"I don't recognize that color option. We currently offer: {available_colors}."
        return state

    full_inventory_str = "\n".join(
        f"- {d['color']}: {'in stock (' + str(d['quantity']) + ' available)' if d['quantity'] > 0 else 'out of stock'}"
        for d in all_docs
    )

    prompt = STOCK_RESPONSE_PROMPT.format(result=result, full_inventory=full_inventory_str)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.3,
    )

    state["response"] = response.choices[0].message.content
    print(f"[check_stock_node] Response: {state['response']}")
    return state


# ---------------------------------------------------------------------
# Tool-calling: order booking
# ---------------------------------------------------------------------

REQUIRED_ORDER_FIELDS = ["color", "quantity", "shipping_address", "email"]

ORDER_EXTRACTION_PROMPT = """You are extracting order booking details from a customer's message.

Extract any of the following present in the message below. Use null for anything not mentioned.

- color: one of the product colors mentioned, or null
- quantity: a whole number of units, or null
- shipping_address: a REAL, concrete shipping address (street, city, etc.) if one is actually given. If the user instead refers to a previous address vaguely (e.g. "same address", "same as before", "same as last time", "ship it there again"), set shipping_address to null and instead set uses_previous_address to true.
- email: an email address if mentioned. If the user instead refers to a previous email vaguely (e.g. "same email", "same as above", "use the one I gave before"), set email to null and instead set uses_previous_email to true.

CUSTOMER MESSAGE:
{message}

Respond ONLY with valid JSON in this exact format:
{{"color": "..." or null, "quantity": 0 or null, "shipping_address": "..." or null, "uses_previous_address": true or false, "email": "..." or null, "uses_previous_email": true or false}}
"""


def _missing_order_fields(order: dict) -> list[str]:
    return [f for f in REQUIRED_ORDER_FIELDS if not order.get(f)]


def _describe_order_field(field: str) -> str:
    descriptions = {
        "color": "which color you'd like",
        "quantity": "how many you'd like to order",
        "shipping_address": "your shipping address",
        "email": "your email address",
    }
    return descriptions.get(field, field)


def _build_order_confirmation_email_html(order: dict) -> str:
    return f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto; color: #2b2622;">
      <h2 style="color: #a0522d;">Order Confirmed — ShopNest Pulse</h2>
      <p>Hi{f" {order['customer_name']}" if order.get('customer_name') else ''},</p>
      <p>Thanks for your order! Here are your order details:</p>

      <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
        <tr><td style="padding: 6px 0; color: #6e6359;">Order ID</td><td style="padding: 6px 0; font-weight: bold;">{order['order_id']}</td></tr>
        <tr><td style="padding: 6px 0; color: #6e6359;">Product</td><td style="padding: 6px 0;">{order['product']} — {order['color']}</td></tr>
        <tr><td style="padding: 6px 0; color: #6e6359;">Quantity</td><td style="padding: 6px 0;">{order['quantity']}</td></tr>
        <tr><td style="padding: 6px 0; color: #6e6359;">Total</td><td style="padding: 6px 0; font-weight: bold;">${order['total_price']:.2f}</td></tr>
        <tr><td style="padding: 6px 0; color: #6e6359;">Payment</td><td style="padding: 6px 0;">Cash on Delivery (COD)</td></tr>
        <tr><td style="padding: 6px 0; color: #6e6359;">Shipping Address</td><td style="padding: 6px 0;">{order['shipping_address']}</td></tr>
      </table>

      <p style="color: #6e6359; font-size: 13px;">Payment will be collected upon delivery.</p>
      <p style="margin-top: 24px;">Thanks for shopping with ShopNest!</p>
    </div>
    """


def place_order(color: str, quantity: int, shipping_address: str, customer_email: str, customer_name: str | None) -> dict:
    """Writes an order to MongoDB and decrements inventory. Assumes stock was already confirmed available."""
    doc = inventory_collection.find_one({"product": PRODUCT_NAME, "color": color})
    unit_price = doc.get("price", 0) if doc else 0
    total_price = round(unit_price * quantity, 2)

    timestamp = datetime.now(timezone.utc)
    order_id = f"ORD-{timestamp.strftime('%Y%m%d-%H%M%S')}"

    order = {
        "order_id": order_id,
        "timestamp": timestamp.isoformat(),
        "product": PRODUCT_NAME,
        "color": color,
        "quantity": quantity,
        "unit_price": unit_price,
        "total_price": total_price,
        "shipping_address": shipping_address,
        "customer_email": customer_email,
        "customer_name": customer_name,
        "status": "processing",
    }

    try:
        orders_collection.insert_one(dict(order))
        inventory_collection.update_one(
            {"product": PRODUCT_NAME, "color": color},
            {"$inc": {"quantity": -quantity}},
        )
        print(f"[place_order] Decremented {quantity} units of {color} from inventory")
    except Exception as e:
        print(f"[place_order] MongoDB operation failed: {e}")

    return order


@traceable(name="order_booking_node")
def order_booking_node(state: dict) -> dict:
    """
    Collects color, quantity, shipping address, and email in one adaptive
    message. If the user references a previous address, confirms it
    explicitly before using it, using the MOST RECENTLY mentioned real
    address in this session (not just the last successfully booked one).
    """
    existing_order_state = state.get("order_booking") or {}
    full_reset_statuses = {"booked", "abandoned"}

    if existing_order_state.get("status") in full_reset_statuses:
        order_state = {
            "status": "not_started", "color": None, "quantity": None,
            "shipping_address": None, "email": None,
        }
        print("[order_booking_node] Previous order was terminal, starting a fresh order")
    elif existing_order_state.get("status") == "out_of_stock":
        # Only the color was invalid (insufficient stock) — keep
        # everything else the user already gave (address, email) and
        # just ask them to pick a different color or quantity.
        order_state = dict(existing_order_state)
        order_state["status"] = "collecting"
        order_state["color"] = None
        print("[order_booking_node] Previous attempt was out of stock, keeping address/email, re-collecting color")
    else:
        order_state = existing_order_state or {
            "status": "not_started", "color": None, "quantity": None,
            "shipping_address": None, "email": None,
        }

    user_message = state["messages"][-1]["content"]
    is_first_turn = order_state["status"] == "not_started"

    if is_first_turn:
        if state.get("color"):
            order_state["color"] = state["color"]
        if state.get("quantity"):
            order_state["quantity"] = state["quantity"]
        order_state["status"] = "collecting"

    # --- Handle pending address confirmation BEFORE anything else ---
    if order_state.get("confirming_address"):
        confirmed = _interpret_yes_no(user_message)
        if confirmed is True:
            order_state["shipping_address"] = order_state.pop("candidate_address")
            order_state["confirming_address"] = False
        elif confirmed is False:
            order_state.pop("candidate_address", None)
            order_state["confirming_address"] = False
            state["order_booking"] = order_state
            state["response"] = "No problem — what shipping address would you like to use instead?"
            print("[order_booking_node] User rejected remembered address")
            return state
        else:
            state["order_booking"] = order_state
            state["response"] = f"Just to confirm — should I ship to {order_state.get('candidate_address')}?"
            return state

    # --- Handle final order confirmation BEFORE anything else ---
    if order_state.get("status") == "confirming":
        confirmed = _interpret_yes_no(user_message)

        if confirmed is True:
            customer_name = state.get("user_name")
            order = place_order(
                order_state["color"], int(order_state["quantity"]), order_state["shipping_address"],
                order_state["email"], customer_name,
            )

            order_state["status"] = "booked"
            order_state["order_id"] = order["order_id"]
            state["order_booking"] = order_state
            state["last_shipping_address"] = order_state["shipping_address"]
            state["last_email"] = order_state["email"]

            notify_new_order(order["order_id"], order["color"], order["quantity"], order["total_price"], order["customer_email"])

            confirmation_html = _build_order_confirmation_email_html(order)
            email_sent = send_customer_email(
                to_email=order["customer_email"],
                subject=f"Order Confirmed — {order['order_id']}",
                html_body=confirmation_html,
            )
            print(f"[order_booking_node] Confirmation email sent: {email_sent}")

            state["response"] = (
                f"Your order is booked — reference {order['order_id']}, {order['quantity']}x {order['color']} "
                f"for a total of ${order['total_price']:.2f}. Payment will be Cash on Delivery (COD) when your "
                f"order arrives. It'll ship to {order['shipping_address']}, and a confirmation email has "
                f"been sent to {order['customer_email']}."
            )
            print(f"[order_booking_node] Order booked: {order['order_id']}")
            return state

        elif confirmed is False:
            order_state["status"] = "abandoned"
            state["order_booking"] = order_state
            state["response"] = "No problem — your order has not been booked. Let me know if you'd like to order something else."
            print("[order_booking_node] User declined final confirmation, order not booked")
            return state

        else:
            state["order_booking"] = order_state
            state["response"] = "Just to confirm — should I go ahead and place this order?"
            return state

    # Check if the user wants to abandon the order entirely mid-collection
    # (not the first turn, since a decline there would just mean "no
    # thanks" to the greeting, which classify_node already routes elsewhere).
    if not is_first_turn:
        # Only match explicit cancellation phrases here — there's no
        # actual yes/no question being asked during collection, so
        # running _interpret_yes_no on arbitrary info-providing messages
        # (like "3 red") risks misreading them as a decline.
        quit_keywords = ["cancel", "never mind", "nevermind", "forget it", "leave it", "don't want", "do not want", "stop this", "stop the order"]
        stripped_message = user_message.strip().lower().rstrip(".!")
        exact_decline_replies = {"no", "nope", "nah", "no thanks", "not now"}

        looks_like_quit = (
            any(kw in user_message.lower() for kw in quit_keywords)
            or stripped_message in exact_decline_replies
        )

        if looks_like_quit:
            order_state["status"] = "abandoned"
            state["order_booking"] = order_state
            state["response"] = "No problem — I've stopped the order. Let me know if you'd like to order something else."
            print("[order_booking_node] User abandoned order mid-collection")
            return state

        # If the user is asking a stock/color question instead of giving
        # order details, answer it directly, then re-ask for what's still
        # missing — this doesn't count as a failed attempt since the user
        # is engaging productively, just not with a direct answer yet.
        stock_question_keywords = ["what colors", "which colors", "colors are available", "colors do you have",
                                    "check stock", "in stock", "available colors", "what's available", "whats available",
                                    "what options", "which options", "options are available", "options do you have",
                                    "what do you have", "show me colors", "list colors", "what's in stock"]
        if any(kw in user_message.lower() for kw in stock_question_keywords):
            all_docs = list(inventory_collection.find({"product": PRODUCT_NAME}))
            available = ", ".join(f"{d['color']} ({d['quantity']} available)" for d in all_docs if d.get("quantity", 0) > 0)
            state["order_booking"] = order_state
            missing = _missing_order_fields(order_state)
            missing_text = ", ".join(_describe_order_field(f) for f in missing)
            state["response"] = f"Here's what we currently have in stock: {available}. Once you've decided, could you tell me {missing_text}?"
            print("[order_booking_node] Answered stock question mid-collection, not counted as an attempt")
            return state

    extraction_response = client.chat.completions.create(
        messages=[{"role": "user", "content": ORDER_EXTRACTION_PROMPT.format(message=user_message)}],
        model="llama-3.3-70b-versatile",
        temperature=0,
        response_format={"type": "json_object"},
    )
    extracted = json.loads(extraction_response.choices[0].message.content)

    # Normalize color casing against the real inventory list — the
    # extraction LLM has no controlled vocabulary, so "silver" typed by
    # the user must be matched case-insensitively to the real "Silver".
    if extracted.get("color"):
        valid_colors = _get_valid_colors()
        matched = next((c for c in valid_colors if c.lower() == extracted["color"].strip().lower()), None)
        extracted["color"] = matched

    # Track the MOST RECENT real address/email mentioned in this session,
    # updated every time one is given — regardless of whether that order succeeds.
    if extracted.get("shipping_address"):
        state["last_shipping_address"] = extracted["shipping_address"]
    if extracted.get("email"):
        state["last_email"] = extracted["email"]

    for field in REQUIRED_ORDER_FIELDS:
        if extracted.get(field) and not order_state.get(field):
            order_state[field] = extracted[field]

    if not order_state.get("email") and extracted.get("uses_previous_email"):
        last_email = state.get("last_email")
        if last_email:
            order_state["email"] = last_email
            print(f"[order_booking_node] Resolved 'same email' to remembered value: {last_email}")

    if not order_state.get("shipping_address") and extracted.get("uses_previous_address"):
        last_address = state.get("last_shipping_address")
        if last_address:
            order_state["confirming_address"] = True
            order_state["candidate_address"] = last_address
            state["order_booking"] = order_state
            state["response"] = f"Just to confirm — should I ship to {last_address}?"
            print(f"[order_booking_node] Asking to confirm remembered address: {last_address}")
            return state
        else:
            print("[order_booking_node] User referenced 'same address' but none is remembered yet")

    missing = _missing_order_fields(order_state)

    if missing:
        if is_first_turn:
            state["order_booking"] = order_state
            missing_text = ", ".join(_describe_order_field(f) for f in missing)
            state["response"] = f"I'd be happy to help you order that. Could you tell me {missing_text}?"
            print(f"[order_booking_node] Started order collection, missing: {missing}")
            return state

        # Only count this as a stalled attempt if the user's message
        # didn't actually provide any new field this turn — real
        # progress (even partial) should never count against the limit.
        made_progress = any(extracted.get(f) for f in REQUIRED_ORDER_FIELDS)

        if made_progress:
            order_state["attempts"] = 0
        else:
            order_state["attempts"] = order_state.get("attempts", 0) + 1

        state["order_booking"] = order_state

        if order_state["attempts"] >= 3:
            order_state["status"] = "abandoned"
            state["order_booking"] = order_state
            state["response"] = (
                "I'm having trouble getting all the order details. Let's pause here — "
                "feel free to try again whenever you're ready."
            )
            print(f"[order_booking_node] Gave up after {order_state['attempts']} attempts")
            return state

        missing_text = ", ".join(_describe_order_field(f) for f in missing)
        state["response"] = f"Thanks — I still need {missing_text} to complete your order."
        print(f"[order_booking_node] Still missing: {missing} (attempt {order_state['attempts']})")
        return state

    stock_doc = inventory_collection.find_one({"product": PRODUCT_NAME, "color": order_state["color"]})

    if not stock_doc:
        available = ", ".join(d["color"] for d in inventory_collection.find({"product": PRODUCT_NAME}))
        order_state["status"] = "collecting"
        order_state["color"] = None
        state["order_booking"] = order_state
        state["response"] = f"I don't recognize that color. We currently offer: {available}. Which would you like?"
        return state

    try:
        requested_qty = int(order_state["quantity"])
    except (ValueError, TypeError):
        requested_qty = 0

    if stock_doc.get("quantity", 0) < requested_qty or stock_doc.get("quantity", 0) == 0:
        order_state["status"] = "out_of_stock"
        state["order_booking"] = order_state
        state["response"] = (
            f"Unfortunately we only have {stock_doc.get('quantity', 0)} of the {order_state['color']} "
            f"in stock right now, which isn't enough for {requested_qty}. Would you like a different "
            f"color or a smaller quantity?"
        )
        return state

    # All fields present and stock confirmed — show a summary and ask for
    # final confirmation before actually committing the order (writing to
    # MongoDB, decrementing inventory, sending email, notifying Slack).
    unit_price = stock_doc.get("price", 0)
    total_price = round(unit_price * requested_qty, 2)

    order_state["status"] = "confirming"
    order_state["pending_total"] = total_price
    state["order_booking"] = order_state

    state["response"] = (
        f"Just to confirm — {requested_qty}x {order_state['color']} for a total of ${total_price:.2f}, "
        f"paid via Cash on Delivery, shipping to {order_state['shipping_address']}, with confirmation "
        f"sent to {order_state['email']}. Should I go ahead and place this order?"
    )
    print(f"[order_booking_node] Awaiting final confirmation before booking")
    return state


# ---------------------------------------------------------------------
# Tool-calling: order tracking
# ---------------------------------------------------------------------

ORDER_ID_REGEX = re.compile(r"ORD-\d{8}-\d{6}", re.IGNORECASE)

ATTEMPTED_ID_REGEX = re.compile(r"\b(?=[a-zA-Z0-9-]*\d)(?=[a-zA-Z0-9-]*[a-zA-Z])[a-zA-Z0-9-]{6,}\b")


def _extract_order_id_or_detect_attempt(message: str) -> tuple[Optional[str], bool]:
    """
    Returns (order_id, looked_like_an_attempt).
    - If a real order ID is found: (order_id, True)
    - If no real ID, but the message contains something that looks like
      a malformed attempt at one: (None, True)
    - If nothing order-ID-like is present at all: (None, False)
    """
    match = ORDER_ID_REGEX.search(message)
    if match:
        return match.group(0).upper(), True

    attempt_match = ATTEMPTED_ID_REGEX.search(message)
    if attempt_match:
        return None, True

    return None, False


def track_order(order_id: str) -> dict | None:
    """Deterministic lookup — returns the real order record or None if not found."""
    return orders_collection.find_one({"order_id": order_id.upper()})


@traceable(name="order_tracking_node")
def order_tracking_node(state: dict) -> dict:
    """
    Looks up an order by ID. Never guesses — if the order_id isn't found
    in MongoDB, says so plainly rather than fabricating a status. Only
    trusts an order_id if it literally appears in THIS message, since
    classify_node's history context can otherwise leak an order ID
    mentioned earlier in the conversation. Distinguishes "no ID given
    at all" from "gave something that looks like a malformed ID".
    """
    user_message = state["messages"][-1]["content"]

    order_id, looked_like_attempt = _extract_order_id_or_detect_attempt(user_message)

    if not order_id:
        if looked_like_attempt:
            state["response"] = (
                "That doesn't look like a valid order ID. It should be in the format "
                "ORD-YYYYMMDD-HHMMSS, for example ORD-20260711-064722. Could you share it again?"
            )
            print("[order_tracking_node] Malformed order_id attempt detected, asking for correct format")
        else:
            state["response"] = "Sure — what's your order ID? It looks like ORD- followed by numbers."
            print("[order_tracking_node] No order_id found, asking for it")
        return state

    order = track_order(order_id)

    if not order:
        state["response"] = (
            f"I couldn't find an order with the ID {order_id}. Could you double-check the "
            f"order number? It should look like ORD-YYYYMMDD-HHMMSS."
        )
        print(f"[order_tracking_node] Order '{order_id}' not found")
        return state

    if order["status"] == "cancelled":
        state["response"] = (
            f"Your order {order['order_id']} — {order['quantity']}x {order['color']} — "
            f"was cancelled and is no longer being processed."
        )
    else:
        state["response"] = (
            f"Your order {order['order_id']} — {order['quantity']}x {order['color']} — "
            f"is currently marked as \"{order['status']}\". It was placed on "
            f"{order['timestamp'][:10]} and is being shipped to {order['shipping_address']}."
        )
    print(f"[order_tracking_node] Order '{order_id}' found, status={order['status']}")
    return state


# ---------------------------------------------------------------------
# Tool-calling: order cancellation
# ---------------------------------------------------------------------

def cancel_order(order_id: str) -> dict:
    """
    Deterministic cancellation logic. Only cancels orders still in
    'processing' status, and restores the inventory quantity that was
    decremented at booking time.
    """
    order = orders_collection.find_one({"order_id": order_id.upper()})

    if not order:
        return {"found": False, "cancelled": False, "reason": "not_found"}

    if order.get("status") != "processing":
        return {"found": True, "cancelled": False, "reason": "wrong_status", "order": order}

    orders_collection.update_one(
        {"order_id": order_id.upper()},
        {"$set": {"status": "cancelled"}},
    )
    inventory_collection.update_one(
        {"product": order["product"], "color": order["color"]},
        {"$inc": {"quantity": order["quantity"]}},
    )
    print(f"[cancel_order] Cancelled {order_id}, restored {order['quantity']} units of {order['color']}")

    order["status"] = "cancelled"
    return {"found": True, "cancelled": True, "reason": None, "order": order}


@traceable(name="order_cancel_node")
def order_cancel_node(state: dict) -> dict:
    """
    Cancels an order by ID if it's still processing. Never guesses —
    if the order isn't found or isn't cancellable, says so plainly.
    Only trusts an order_id if it literally appears in THIS message,
    since classify_node's history context can otherwise leak an order
    ID mentioned earlier in the conversation. Distinguishes "no ID
    given at all" from "gave something that looks like a malformed ID".
    """
    user_message = state["messages"][-1]["content"]

    order_id, looked_like_attempt = _extract_order_id_or_detect_attempt(user_message)

    if not order_id:
        if looked_like_attempt:
            state["response"] = (
                "That doesn't look like a valid order ID. It should be in the format "
                "ORD-YYYYMMDD-HHMMSS, for example ORD-20260711-064722. Could you share it again?"
            )
            print("[order_cancel_node] Malformed order_id attempt detected, asking for correct format")
        else:
            state["response"] = "Sure — what's the order ID you'd like to cancel? It looks like ORD- followed by numbers."
            print("[order_cancel_node] No order_id found, asking for it")
        return state

    result = cancel_order(order_id)

    if not result["found"]:
        state["response"] = (
            f"I couldn't find an order with the ID {order_id}. Could you double-check the order number?"
        )
        print(f"[order_cancel_node] Order '{order_id}' not found")
        return state

    if not result["cancelled"]:
        current_status = result["order"]["status"]
        state["response"] = (
            f"I'm sorry, but order {order_id} is already marked as \"{current_status}\" and can no "
            f"longer be cancelled. Orders can only be cancelled while they're still processing."
        )
        print(f"[order_cancel_node] Order '{order_id}' not cancellable, status={current_status}")
        return state

    order = result["order"]

    notify_order_cancelled(order_id, order["color"], order["quantity"])

    state["response"] = (
        f"Done — order {order_id} has been cancelled, and your {order['quantity']}x {order['color']} "
        f"has been returned to stock. You won't be charged for this order."
    )
    print(f"[order_cancel_node] Order '{order_id}' successfully cancelled")
    return state


# ---------------------------------------------------------------------
# Tool-calling: warranty claim filing (MongoDB + email consent flow)
# ---------------------------------------------------------------------

REQUIRED_CLAIM_FIELDS = ["earbud_affected", "issue_description", "ownership_duration"]

WARRANTY_EXTRACTION_PROMPT = """You are extracting warranty claim details for a customer's warranty claim.

Extract any of the following that are present EITHER in the customer's latest message OR earlier in the conversation history below (e.g. if they described their problem a few turns ago before explicitly asking for a warranty claim). Use null for anything not mentioned anywhere — do not guess or invent values.

- earbud_affected: "left", "right", "both", or null if not specified anywhere
- issue_description: a short description of the actual problem, or null if not specified anywhere
- ownership_duration: how long they've had the product (e.g. "2 weeks"), or null if not mentioned anywhere

CONVERSATION HISTORY:
{history}

CUSTOMER'S LATEST MESSAGE:
{message}

Respond ONLY with valid JSON in this exact format:
{{"earbud_affected": "..." or null, "issue_description": "..." or null, "ownership_duration": "..." or null}}
"""


def file_warranty_claim(earbud_affected: str, issue_description: str, ownership_duration: str) -> dict:
    """Writes a warranty claim to MongoDB and returns the claim record (without _id)."""
    timestamp = datetime.now(timezone.utc)
    claim_id = f"WC-{timestamp.strftime('%Y%m%d-%H%M%S')}"

    claim = {
        "claim_id": claim_id,
        "timestamp": timestamp.isoformat(),
        "product": PRODUCT_NAME,
        "earbud_affected": earbud_affected,
        "issue_description": issue_description,
        "ownership_duration": ownership_duration,
        "status": "pending_review",
        "customer_email": None,
        "email_sent": False,
    }

    try:
        warranty_claims_collection.insert_one(dict(claim))
    except Exception as e:
        print(f"[file_warranty_claim] MongoDB insert failed: {e}")

    return claim


def _missing_claim_fields(claim: dict) -> list[str]:
    return [f for f in REQUIRED_CLAIM_FIELDS if not claim.get(f)]


def _describe_claim_field(field: str) -> str:
    descriptions = {
        "earbud_affected": "which earbud is affected — left, right, or both",
        "issue_description": "what exactly is happening with it",
        "ownership_duration": "how long you've had them",
    }
    return descriptions.get(field, field)


def _build_claim_email_html(claim: dict, customer_email: str, summary: str) -> str:
    return f"""
    <h2>New Warranty Claim — {claim['claim_id']}</h2>
    <p><strong>Customer email:</strong> {customer_email}</p>
    <p><strong>Submitted:</strong> {claim['timestamp']}</p>
    <p><strong>Earbud affected:</strong> {claim['earbud_affected']}</p>
    <p><strong>Ownership duration:</strong> {claim['ownership_duration']}</p>
    <h3>Issue</h3>
    <p>{claim['issue_description']}</p>
    <h3>Summary</h3>
    <p>{summary}</p>
    """


def _check_within_warranty_period(ownership_duration: str) -> Optional[bool]:
    """
    Deterministic check: does the stated ownership duration fall within
    the 12-month warranty window? Returns True/False, or None if the
    duration text couldn't be confidently parsed (in which case we let
    the claim proceed rather than blocking on an ambiguous answer).
    """
    text = ownership_duration.lower().strip()

    match = re.search(r"(\d+)\s*(day|week|month|year)s?", text)
    if not match:
        return None

    number = int(match.group(1))
    unit = match.group(2)

    total_months = {
        "day": number / 30,
        "week": number / 4.345,
        "month": number,
        "year": number * 12,
    }[unit]

    return total_months <= 12


@traceable(name="warranty_claim_node")
def warranty_claim_node(state: dict) -> dict:
    """
    Collects earbud_affected, issue_description, and ownership_duration
    in one adaptive message (asking only for what's missing). Runs
    extraction on every turn, including the first, so a fully detailed
    first message never gets its info thrown away. If a previous claim
    in this session already reached a terminal outcome (filed, voided,
    abandoned, declined_out_of_warranty), starts a completely fresh
    claim instead of reusing stale details from that earlier attempt.
    """
    existing_claim_state = state.get("warranty_claim") or {}
    terminal_statuses = {"filed", "voided", "abandoned", "declined_out_of_warranty"}

    if existing_claim_state.get("status") in terminal_statuses:
        claim_state = {
            "status": "not_started",
            "earbud_affected": None,
            "issue_description": None,
            "ownership_duration": None,
        }
        print("[warranty_claim_node] Previous claim was terminal, starting a fresh claim")
    else:
        claim_state = existing_claim_state or {
            "status": "not_started",
            "earbud_affected": None,
            "issue_description": None,
            "ownership_duration": None,
        }

    user_message = state["messages"][-1]["content"]
    is_first_turn = claim_state["status"] == "not_started"

    if is_first_turn:
        claim_state["status"] = "collecting"
        # Extraction can see everything since the LAST claim ended (not
        # just this exact message) — so a genuine lead-up complaint like
        # "my earbuds died" followed later by "I want to claim warranty
        # for this" still works, while an OLDER, already-resolved claim's
        # details never bleed into a new one.
        claim_state["start_index"] = state.get("warranty_claim_boundary_index", 0)

    claim_start_index = claim_state.get("start_index", 0)
    relevant_messages = state["messages"][claim_start_index:]
    history_text_for_extraction = format_history_for_prompt(get_recent_history(relevant_messages, limit=10))
    extraction_response = client.chat.completions.create(
        messages=[{"role": "user", "content": WARRANTY_EXTRACTION_PROMPT.format(message=user_message, history=history_text_for_extraction)}],
        model="llama-3.3-70b-versatile",
        temperature=0,
        response_format={"type": "json_object"},
    )
    extracted = json.loads(extraction_response.choices[0].message.content)

    for field in REQUIRED_CLAIM_FIELDS:
        if extracted.get(field) and not claim_state.get(field):
            claim_state[field] = extracted[field]

    missing = _missing_claim_fields(claim_state)

    if missing:
        if is_first_turn:
            state["warranty_claim"] = claim_state
            missing_text = " and ".join(_describe_claim_field(f) for f in missing)
            state["response"] = f"This does sound like something that could be covered under our 12-month warranty. Could you tell me {missing_text}?"
            print(f"[warranty_claim_node] Started claim collection, missing: {missing}")
            return state

        claim_state["attempts"] = claim_state.get("attempts", 0) + 1
        state["warranty_claim"] = claim_state

        if claim_state["attempts"] >= 3:
            claim_state["status"] = "abandoned"
            state["warranty_claim"] = claim_state
            state["warranty_claim_boundary_index"] = len(state["messages"])
            state["response"] = (
                "I'm having trouble getting the specific details needed for a warranty claim. "
                "Let me connect you with our support team directly instead — they'll be able to help."
            )
            state["escalate"] = True
            print(f"[warranty_claim_node] Gave up after {claim_state['attempts']} attempts, escalating instead")
            return state

        missing_text = " and ".join(_describe_claim_field(f) for f in missing)
        state["response"] = f"Thanks — I still need to know {missing_text} to file the claim."
        print(f"[warranty_claim_node] Still missing: {missing} (attempt {claim_state['attempts']})")
        return state

    # All fields present — check ownership duration is within the 12-month
    # warranty window BEFORE filing. Never file a claim for a product
    # clearly owned longer than a year.
    within_warranty = _check_within_warranty_period(claim_state["ownership_duration"])

    if within_warranty is False:
        claim_state["status"] = "declined_out_of_warranty"
        state["warranty_claim"] = claim_state
        state["warranty_claim_boundary_index"] = len(state["messages"])
        state["response"] = (
            f"I'm sorry, but our warranty only covers the first 12 months from purchase, and you've "
            f"mentioned owning them for {claim_state['ownership_duration']}, which is outside that window. "
            f"I'm not able to file a warranty claim for this, but you're welcome to reach out to our "
            f"support team directly if you'd like to discuss other options."
        )
        print(f"[warranty_claim_node] Declined — ownership duration '{claim_state['ownership_duration']}' exceeds 12-month warranty")
        return state

    claim = file_warranty_claim(
        claim_state["earbud_affected"], claim_state["issue_description"], claim_state["ownership_duration"]
    )

    claim_state["status"] = "filed"
    state["warranty_claim"] = claim_state
    state["warranty_email"] = {"status": "collecting_email", "claim_id": claim["claim_id"], "email": None}
    state["warranty_claim_boundary_index"] = len(state["messages"])
    state["response"] = (
        f"Thanks for those details. I've filed a warranty claim for you — reference {claim['claim_id']}. "
        f"To make sure our team can follow up, could you share your email address?"
    )

    print(f"[warranty_claim_node] Claim filed: {claim['claim_id']}, awaiting email consent")
    return state


@traceable(name="warranty_email_node")
def warranty_email_node(state: dict) -> dict:
    """
    Collects the customer's email for a filed warranty claim. Email is
    mandatory — if the user explicitly declines or asks to leave/stop,
    the claim is VOIDED entirely (not finalized without contact info),
    since the team has no way to follow up without an email.
    """
    warranty_email = state.get("warranty_email") or {}
    claim_id = warranty_email.get("claim_id")
    user_message = state["messages"][-1]["content"]

    email_found = _extract_email(user_message)

    if email_found:
        claim_doc = warranty_claims_collection.find_one({"claim_id": claim_id}) or {}
        summary = f"Warranty claim for {claim_doc.get('earbud_affected', 'unspecified')} earbud: {claim_doc.get('issue_description', '')}"
        html = _build_claim_email_html(claim_doc, email_found, summary)
        try:
            send_support_email(subject=f"[VoiceCart Warranty] Claim {claim_id}", html_body=html)
            warranty_claims_collection.update_one(
                {"claim_id": claim_id},
                {"$set": {"customer_email": email_found, "email_sent": True}},
            )
        except Exception as e:
            print(f"[warranty_email_node] Email/Mongo update failed: {e}")

        notify_warranty_claim(
            claim_id, claim_doc.get("earbud_affected", "unspecified"),
            claim_doc.get("issue_description", ""), email_found,
        )

        warranty_email["email"] = email_found
        warranty_email["status"] = "sent"
        state["warranty_email"] = warranty_email
        state["response"] = f"Done — I've emailed our support team about claim {claim_id}, and they'll follow up at {email_found}."
        return state

    decline_keywords = ["no", "cancel", "leave it", "never mind", "nevermind", "forget it", "don't want", "do not want", "skip", "stop"]
    declined = any(kw in user_message.lower() for kw in decline_keywords)

    if declined:
        try:
            warranty_claims_collection.update_one(
                {"claim_id": claim_id},
                {"$set": {"status": "voided_no_email"}},
            )
        except Exception as e:
            print(f"[warranty_email_node] Failed to void claim: {e}")

        warranty_email["status"] = "declined"
        state["warranty_email"] = warranty_email

        claim_state = state.get("warranty_claim") or {}
        claim_state["status"] = "voided"
        state["warranty_claim"] = claim_state

        state["response"] = (
            "Since an email is required for our team to follow up, I'm not able to file this claim "
            "without one. Your claim has not been submitted — feel free to start again anytime with "
            "an email address on hand."
        )
        print(f"[warranty_email_node] Claim {claim_id} voided — no email provided")
        return state

    state["response"] = "I couldn't catch a valid email address — could you type it again, or let me know if you'd rather skip it?"
    return state


# ---------------------------------------------------------------------
# Streaming support (text input path only)
# ---------------------------------------------------------------------

def classify_and_retrieve(state: dict) -> dict:
    state = classify_node(state)

    intent = state.get("intent")
    namespace = state.get("target_namespace")
    warranty_claim = state.get("warranty_claim") or {}
    needs_warranty_flow = (
        intent == "complaint" and state.get("warranty_eligible") and warranty_claim.get("status") != "filed"
    )

    if intent in ("other", "stock_check", "order_booking", "order_tracking", "order_cancel") or needs_warranty_flow or not namespace:
        state["context"] = None
        return state

    messages = state["messages"]
    user_message = messages[-1]["content"]
    context, top_score = _retrieve_context(user_message, namespace)

    if context is None:
        state["context"] = None
        state["retrieval_score"] = top_score
        return state

    secondary_namespace = state.get("secondary_namespace")
    if secondary_namespace and secondary_namespace != namespace:
        secondary_context, _ = _retrieve_context(user_message, secondary_namespace)
        if secondary_context:
            context = context + "\n\n" + secondary_context

    state["context"] = context
    state["retrieval_score"] = top_score
    return state


def stream_answer(state: dict):
    intent = state.get("intent")
    sentiment = state.get("sentiment", "neutral")
    context = state.get("context")
    messages = state["messages"]
    user_message = messages[-1]["content"]
    history = get_recent_history(messages, limit=10)
    history_text = format_history_for_prompt(history)

    if context is None:
        if intent == "complaint":
            yield NO_CONTEXT_EMPATHETIC_RESPONSE
        else:
            yield NO_ANSWER_RESPONSE
        return

    if intent == "complaint":
        prompt = EMPATHETIC_ANSWER_PROMPT.format(context=context, question=user_message, sentiment=sentiment, history=history_text)
    else:
        prompt = GROUNDED_ANSWER_PROMPT.format(context=context, question=user_message, history=history_text)

    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta