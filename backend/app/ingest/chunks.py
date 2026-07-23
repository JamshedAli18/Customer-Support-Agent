"""
Loads 5 separate KB PDFs (one per namespace), splits each into numbered
sections using the literal '## N. Title' header marker, and tags every
chunk with its namespace based on filename — same pattern as PolicyMind's
filename-based classifier, applied across multiple files instead of one.
"""

import os
import re
import pdfplumber

INGEST_DIR = os.path.dirname(__file__)

# Filename -> namespace mapping, PolicyMind-style
FILE_NAMESPACE_MAP = {
    "product-info.pdf": "product-info",
    "usage-guidance.pdf": "usage-guidance",
    "troubleshooting.pdf": "troubleshooting",
    "policies.pdf": "policies",
    "limitations.pdf": "limitations",
}

# Matches literal '## N. Title' at the start of a line
SECTION_HEADER_PATTERN = re.compile(
    r"^##\s*(\d{1,2})\.\s+(.+?)\s*$", re.MULTILINE
)


def extract_pdf_text(pdf_path: str) -> str:
    """Extract raw text from all pages of a PDF, preserving order."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)


def split_into_sections(full_text: str, filename: str) -> list[dict]:
    """Split raw text into sections based on '## N. Title' headers."""
    matches = list(SECTION_HEADER_PATTERN.finditer(full_text))

    if not matches:
        raise ValueError(
            f"No '## N. Title' headers found in {filename}. "
            f"Check the file actually contains literal '##' markers."
        )

    sections = []
    for i, match in enumerate(matches):
        section_num = int(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()
        sections.append({
            "section_num": section_num,
            "title": title,
            "content": content,
        })
    return sections


def build_chunks() -> list[dict]:
    """Build the final chunk list across all 5 KB files: {id, namespace, title, content}."""
    all_chunks = []

    for filename, namespace in FILE_NAMESPACE_MAP.items():
        pdf_path = os.path.join(INGEST_DIR, filename)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Expected file not found: {pdf_path}")

        full_text = extract_pdf_text(pdf_path)
        sections = split_into_sections(full_text, filename)

        for section in sections:
            all_chunks.append({
                "id": f"{namespace}-{section['section_num']:03d}",
                "namespace": namespace,
                "title": section["title"],
                "content": section["content"],
            })

        print(f"Parsed {len(sections)} sections from {filename} -> namespace '{namespace}'")

    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"\nTotal chunks parsed: {len(chunks)}\n")
    for c in chunks:
        preview = c["content"][:80].replace("\n", " ")
        print(f"[{c['namespace']}] {c['id']} — {c['title']}")
        print(f"   {preview}...\n")