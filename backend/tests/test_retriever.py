"""
Tests app/retrieval/retriever.py — validates hybrid search retrieval
correctness across all 5 namespaces, plus the anti-hallucination
threshold behavior.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app", "retrieval"))
from retriever import VoiceCartRetriever

SCORE_THRESHOLD = 0.22

# (query, namespace, expected_title_substring)
EXPECTED_TOP_RESULTS = [
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
]

# Adversarial: a question with zero relevance to the namespace it's forced into
ADVERSARIAL_QUERY = "What's the capital of France?"
ADVERSARIAL_NAMESPACE = "product-info"


def test_top_result_relevance(retriever):
    failures = []
    for query, namespace, expected_substring in EXPECTED_TOP_RESULTS:
        results = retriever.retrieve(query, namespace, top_k=3)
        assert results, f"No results returned for '{query}' in namespace '{namespace}'"
        top_title = results[0]["title"]
        if expected_substring.lower() not in top_title.lower():
            failures.append(
                f"Query '{query}' -> expected title containing '{expected_substring}', got '{top_title}'"
            )

    assert not failures, "Relevance failures:\n" + "\n".join(failures)
    print(f"PASS: top-1 result correct for all {len(EXPECTED_TOP_RESULTS)} test queries")


def test_scores_above_threshold_for_known_answers(retriever):
    below_threshold = []
    for query, namespace, _ in EXPECTED_TOP_RESULTS:
        results = retriever.retrieve(query, namespace, top_k=3)
        top_score = results[0]["score"]
        if top_score < SCORE_THRESHOLD:
            below_threshold.append(f"'{query}' scored {top_score:.4f} (below {SCORE_THRESHOLD})")

    assert not below_threshold, (
        "Known-good queries scored below threshold (would incorrectly trigger fallback):\n"
        + "\n".join(below_threshold)
    )
    print(f"PASS: all known-answerable queries score >= {SCORE_THRESHOLD}")


def test_adversarial_query_scores_low(retriever):
    """
    Core anti-hallucination check: an irrelevant question forced into a
    namespace should score well below threshold, so inquiry_node's
    fallback logic correctly avoids fabricating an answer.
    """
    results = retriever.retrieve(ADVERSARIAL_QUERY, ADVERSARIAL_NAMESPACE, top_k=3)
    top_score = results[0]["score"] if results else 0.0

    assert top_score < SCORE_THRESHOLD, (
        f"Adversarial query scored {top_score:.4f}, which is ABOVE threshold "
        f"{SCORE_THRESHOLD} — this would risk hallucination, since inquiry_node "
        f"would proceed to generate an answer from irrelevant context."
    )
    print(f"PASS: adversarial query scored {top_score:.4f} (correctly below threshold)")


def test_invalid_namespace_raises(retriever):
    try:
        retriever.retrieve("test query", "not-a-real-namespace", top_k=3)
        assert False, "Expected ValueError for invalid namespace, but none was raised"
    except ValueError:
        print("PASS: invalid namespace correctly raises ValueError")


def test_results_are_sorted_descending(retriever):
    results = retriever.retrieve("How long is the return window?", "policies", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"Results not sorted by descending score: {scores}"
    )
    print("PASS: results returned in descending score order")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING retriever.py")
    print("=" * 60)

    retriever = VoiceCartRetriever(alpha=0.75)

    test_top_result_relevance(retriever)
    test_scores_above_threshold_for_known_answers(retriever)
    test_adversarial_query_scores_low(retriever)
    test_invalid_namespace_raises(retriever)
    test_results_are_sorted_descending(retriever)

    print("\nAll retriever.py tests passed.")