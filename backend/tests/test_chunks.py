"""
Tests app/ingest/chunks.py — validates PDF parsing, section splitting,
and namespace assignment.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app", "ingest"))
from chunks import build_chunks, FILE_NAMESPACE_MAP

EXPECTED_COUNTS = {
    "product-info": 10,
    "usage-guidance": 19,
    "troubleshooting": 5,
    "policies": 8,
    "limitations": 5,
}

VALID_NAMESPACES = set(FILE_NAMESPACE_MAP.values())


def test_total_chunk_count():
    chunks = build_chunks()
    expected_total = sum(EXPECTED_COUNTS.values())
    assert len(chunks) == expected_total, (
        f"Expected {expected_total} total chunks, got {len(chunks)}"
    )
    print(f"PASS: total chunk count = {len(chunks)}")


def test_per_namespace_counts():
    chunks = build_chunks()
    counts = {}
    for c in chunks:
        counts[c["namespace"]] = counts.get(c["namespace"], 0) + 1

    for namespace, expected in EXPECTED_COUNTS.items():
        actual = counts.get(namespace, 0)
        assert actual == expected, (
            f"Namespace '{namespace}': expected {expected} chunks, got {actual}"
        )
    print(f"PASS: per-namespace counts = {counts}")


def test_all_namespaces_valid():
    chunks = build_chunks()
    for c in chunks:
        assert c["namespace"] in VALID_NAMESPACES, (
            f"Chunk '{c['id']}' has invalid namespace '{c['namespace']}'"
        )
    print("PASS: all chunks have valid namespaces")


def test_no_empty_content():
    chunks = build_chunks()
    for c in chunks:
        assert c["content"].strip(), f"Chunk '{c['id']}' has empty content"
        assert c["title"].strip(), f"Chunk '{c['id']}' has empty title"
    print("PASS: no chunk has empty title or content")


def test_no_leaked_title_fragments():
    """
    Regression test for the bug where truncated title-matching left
    fragments like ', Refunds, and Warranty' or '(for honest...)' at
    the start of content.
    """
    chunks = build_chunks()
    suspicious_starts = (",", ")", "(for", "(Common")
    for c in chunks:
        stripped = c["content"].strip()
        assert not stripped.startswith(suspicious_starts), (
            f"Chunk '{c['id']}' content starts suspiciously: '{stripped[:50]}'"
        )
    print("PASS: no leaked title fragments at start of content")


def test_unique_ids():
    chunks = build_chunks()
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"
    print("PASS: all chunk IDs are unique")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING chunks.py")
    print("=" * 60)

    test_total_chunk_count()
    test_per_namespace_counts()
    test_all_namespaces_valid()
    test_no_empty_content()
    test_no_leaked_title_fragments()
    test_unique_ids()

    print("\nAll chunks.py tests passed.")