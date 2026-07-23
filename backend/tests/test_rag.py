"""
test_rag.py — Standalone test suite for VoiceCart's RAG pipeline:
hybrid retrieval (Pinecone dense + BM25 sparse) and grounded answer
generation (inquiry_node / empathetic_response_node).

Run with:
    python test_cases/test_rag.py

============================================================
WHY alpha = 0.75 (dense-weighted hybrid search)
============================================================
Hybrid search combines two retrieval signals:
  - DENSE (Cohere embeddings via Pinecone): captures semantic MEANING.
    Good for paraphrased questions like "are these good for running?"
    matching content that never uses the word "running" directly.
  - SPARSE (BM25 via pinecone-text): captures exact KEYWORD overlap.
    Good for precise terms like "IPX5", "Bluetooth 5.3", product names,
    or numbers that dense embeddings can blur together.

alpha controls the blend: alpha=1.0 is pure dense, alpha=0.0 is pure
sparse. We use alpha=0.75 (dense-leaning) because:
  1. Most real user questions are natural language, not keyword search
     ("what's the battery life?" not "battery life spec"), so semantic
     matching should dominate.
  2. BM25 still meaningfully boosts exact-term queries (return window,
     IPX rating, specific color names) without needing alpha=0.5, since
     even a 25% sparse contribution is enough to break ties toward the
     chunk that contains the literal keyword.
  3. This was tuned empirically during Phase 2 retrieval testing — pure
     dense (alpha=1.0) occasionally missed exact-spec questions, and a
     lower alpha (e.g. 0.5) over-prioritized keyword overlap and hurt
     paraphrased/guidance-style questions. 0.75 was the best balance
     found by testing real queries against the actual knowledge base.

============================================================
WHY SCORE_THRESHOLD = 0.22 (anti-hallucination gate)
============================================================
Every retrieval returns a top score. If that score is too low, the top
result is probably NOT actually relevant to the question — using it as
context would risk the LLM confidently answering from a weakly-related
chunk, which is a subtle form of hallucination (grounded in the wrong
thing, not ungrounded).

0.22 was chosen empirically, not arbitrarily:
  - We ran a battery of known-answerable questions across all 5
    namespaces and recorded real top-1 scores (see Phase 2 test logs).
    Correct, relevant answers scored consistently >= 0.22 even for
    "weak" but legitimate matches (e.g. vague questions like "what's
    included in the box?" scored ~0.27).
  - We ran adversarial, deliberately irrelevant questions (e.g. "what's
    the capital of France?" forced into the product-info namespace) and
    those scored ~0.04, an order of magnitude below any real match.
  - 0.22 sits comfortably in the gap between "genuinely relevant, even
    if a weak match" and "not relevant at all" — low enough to avoid
    false rejections of real answers, high enough to catch true
    non-matches before they reach the LLM.

If retrieval score < 0.22, the system returns an honest "I don't have
that information" instead of generating an answer — this is enforced
in inquiry_node, empathetic_response_node, and classify_and_retrieve.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app", "retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app", "graph"))

from retriever import VoiceCartRetriever
from nodes import inquiry_node, empathetic_response_node, SCORE_THRESHOLD

ALPHA = 0.75

retriever = VoiceCartRetriever(alpha=ALPHA)


# ---------------------------------------------------------------------
# Section 1: Raw retrieval tests (scores + correct chunk returned)
# ---------------------------------------------------------------------

RETRIEVAL_TEST_CASES = [
    # (query, namespace, expected_title_substring)
    ("What is the IPX rating?", "product-info", "Water Resistance"),
    ("How many hours of battery life?", "product-info", "Battery Life"),
    ("What's included in the box?", "product-info", "What's in the Box"),
    ("My earbuds keep falling out", "usage-guidance", "Fit and Comfort"),
    ("Are these good for sleeping?", "usage-guidance", "Sleeping"),
    ("Can I use these for gaming?", "usage-guidance", "Gaming"),
    ("One earbud isn't working", "troubleshooting", "One Earbud Not Producing Sound"),
    ("The earbuds won't charge", "troubleshooting", "Won't Charge"),
    ("How long is the return window?", "policies", "Return Window"),
    ("Can I pay with PayPal?", "policies", "Payment Methods"),
    ("Does ANC block all noise?", "limitations", "ANC Is Not Complete Silence"),
    ("Can I swim with these?", "limitations", "Not Swimproof"),
    ("Will the battery get worse over time?", "limitations", "Battery Capacity Decreases"),
]

# Adversarial: genuinely irrelevant question forced into a namespace —
# must score well below SCORE_THRESHOLD.
ADVERSARIAL_CASES = [
    ("What's the capital of France?", "product-info"),
    ("What's 247 times 13?", "policies"),
    ("Tell me a joke", "usage-guidance"),
]


def test_retrieval_returns_correct_chunk():
    failures = []
    for query, namespace, expected_substring in RETRIEVAL_TEST_CASES:
        results = retriever.retrieve(query, namespace, top_k=5)
        if not results:
            failures.append(f"'{query}' -> no results at all")
            continue
        top_title = results[0]["title"]
        if expected_substring.lower() not in top_title.lower():
            failures.append(f"'{query}' -> expected '{expected_substring}', got '{top_title}'")

    if failures:
        print("FAIL: some queries returned the wrong top chunk:")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"PASS: all {len(RETRIEVAL_TEST_CASES)} queries returned the correct top chunk")


def test_known_answers_score_above_threshold():
    below_threshold = []
    for query, namespace, _ in RETRIEVAL_TEST_CASES:
        results = retriever.retrieve(query, namespace, top_k=5)
        top_score = results[0]["score"] if results else 0.0
        if top_score < SCORE_THRESHOLD:
            below_threshold.append(f"'{query}' scored {top_score:.4f} (below {SCORE_THRESHOLD})")

    if below_threshold:
        print(f"FAIL: known-answerable queries scored below threshold {SCORE_THRESHOLD}:")
        for f in below_threshold:
            print(f"  - {f}")
    else:
        print(f"PASS: all known-answerable queries score >= {SCORE_THRESHOLD}")


def test_adversarial_queries_score_below_threshold():
    failures = []
    for query, namespace in ADVERSARIAL_CASES:
        results = retriever.retrieve(query, namespace, top_k=3)
        top_score = results[0]["score"] if results else 0.0
        if top_score >= SCORE_THRESHOLD:
            failures.append(f"'{query}' scored {top_score:.4f} (should be below {SCORE_THRESHOLD})")
        else:
            print(f"  '{query}' -> {top_score:.4f} (correctly below threshold)")

    if failures:
        print("FAIL: adversarial queries scored high enough to risk hallucination:")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"PASS: all adversarial queries correctly scored below {SCORE_THRESHOLD}")


# ---------------------------------------------------------------------
# Section 2: End-to-end grounded answer tests (inquiry_node)
# ---------------------------------------------------------------------

GROUNDED_ANSWER_CASES = [
    ("What is the IPX rating on these earbuds?", "product-info", ["IPX5"]),
    ("How long is the return window?", "policies", ["30 days", "30-day"]),
    ("Can I swim with these?", "limitations", ["not", "swimproof"]),
    ("What's the battery life?", "product-info", ["6 hours", "4.5 hours"]),
]


def test_inquiry_node_grounded_answers():
    failures = []
    for question, namespace, expected_keywords in GROUNDED_ANSWER_CASES:
        state = {"messages": [{"role": "user", "content": question}], "target_namespace": namespace}
        result = inquiry_node(state)
        answer = result.get("response", "")

        if not any(kw.lower() in answer.lower() for kw in expected_keywords):
            failures.append(f"'{question}' -> answer missing expected keywords {expected_keywords}: {answer}")

    if failures:
        print("FAIL: some grounded answers were missing expected facts:")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"PASS: all {len(GROUNDED_ANSWER_CASES)} grounded answers contained expected facts")


def test_inquiry_node_adversarial_no_hallucination():
    state = {"messages": [{"role": "user", "content": "What's the capital of France?"}], "target_namespace": "product-info"}
    result = inquiry_node(state)
    answer = result.get("response", "").lower()
    score = result.get("retrieval_score")

    assert score is not None and score < SCORE_THRESHOLD, f"Expected score below threshold, got {score}"
    assert "paris" not in answer, f"Hallucinated an answer about France: {answer}"
    assert "don't have" in answer or "recommend reaching out" in answer, f"Expected an honest fallback, got: {answer}"
    print(f"PASS: adversarial query correctly triggered honest fallback (score={score:.4f})")


# ---------------------------------------------------------------------
# Section 3: Score distribution report (informational, not pass/fail)
# ---------------------------------------------------------------------

def print_score_distribution():
    print("\nScore distribution across all test queries (for manual inspection):")
    all_cases = [(q, n) for q, n, _ in RETRIEVAL_TEST_CASES] + ADVERSARIAL_CASES
    for query, namespace in all_cases:
        results = retriever.retrieve(query, namespace, top_k=1)
        score = results[0]["score"] if results else 0.0
        flag = "OK" if score >= SCORE_THRESHOLD else "BELOW THRESHOLD"
        print(f"  [{flag:16}] {score:.4f}  |  '{query}'  ({namespace})")


if __name__ == "__main__":
    print("=" * 70)
    print(f"TESTING RAG PIPELINE  (alpha={ALPHA}, threshold={SCORE_THRESHOLD})")
    print("=" * 70)

    print("\n--- Section 1: Raw retrieval ---")
    test_retrieval_returns_correct_chunk()
    test_known_answers_score_above_threshold()
    test_adversarial_queries_score_below_threshold()

    print("\n--- Section 2: Grounded answer generation ---")
    test_inquiry_node_grounded_answers()
    test_inquiry_node_adversarial_no_hallucination()

    print_score_distribution()

    print("\nAll RAG tests completed.")