"""
Embeds each KB chunk both ways (Cohere dense + BM25 sparse) and upserts
them into the correct Pinecone namespace. BM25 is fit on our own 15-chunk
corpus (not generic MS MARCO defaults) so sparse retrieval reflects term
importance specific to ShopNest Pulse content.
"""

import os
import json
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from chunks import build_chunks

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

BM25_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "bm25_params.json")


def get_dense_embedding(co: cohere.Client, text: str) -> list[float]:
    """Embed a single text as a document (not a query) using Cohere."""
    response = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_document",
    )
    return response.embeddings[0]


def fit_bm25_on_corpus(chunks: list[dict]) -> BM25Encoder:
    """Fit BM25 on our own KB content and save params for reuse at query time."""
    corpus = [chunk["content"] for chunk in chunks]
    bm25 = BM25Encoder()
    bm25.fit(corpus)
    bm25.dump(BM25_PARAMS_PATH)
    print(f"BM25 fitted on {len(corpus)} documents. Params saved to {BM25_PARAMS_PATH}")
    return bm25


def upload_chunks():
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY not found in .env")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in .env")
    if not PINECONE_INDEX_NAME:
        raise ValueError("PINECONE_INDEX_NAME not found in .env")

    chunks = build_chunks()
    print(f"Loaded {len(chunks)} chunks from product.pdf\n")

    co = cohere.Client(COHERE_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    bm25 = fit_bm25_on_corpus(chunks)

    # Group chunks by namespace so we can upsert per-namespace
    by_namespace: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_namespace.setdefault(chunk["namespace"], []).append(chunk)

    for namespace, namespace_chunks in by_namespace.items():
        vectors = []
        for chunk in namespace_chunks:
            dense_vec = get_dense_embedding(co, chunk["content"])
            sparse_vec = bm25.encode_documents(chunk["content"])

            vectors.append({
                "id": chunk["id"],
                "values": dense_vec,
                "sparse_values": {
                    "indices": sparse_vec["indices"],
                    "values": sparse_vec["values"],
                },
                "metadata": {
                    "title": chunk["title"],
                    "text": chunk["content"],
                    "namespace": chunk["namespace"],
                },
            })

        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Upserted {len(vectors)} vectors into namespace '{namespace}'")

    print("\nUpload complete.")
    stats = index.describe_index_stats()
    print(f"\nIndex stats:\n{stats}")


if __name__ == "__main__":
    upload_chunks()