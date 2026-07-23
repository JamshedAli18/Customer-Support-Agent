"""
Namespace-aware hybrid retriever for VoiceCart's knowledge base.
Combines Cohere dense embeddings with BM25 sparse vectors (fitted on our
own corpus during ingestion) using Pinecone's hybrid query pattern.
"""

import os
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

BM25_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "ingest", "bm25_params.json"
)

VALID_NAMESPACES = {
    "product-info",
    "usage-guidance",
    "troubleshooting",
    "policies",
    "limitations",
}


def hybrid_scale(dense: list[float], sparse: dict, alpha: float):
    """
    Scale dense and sparse vectors by alpha so they're comparable before
    combining. alpha=1 is pure dense (semantic), alpha=0 is pure sparse
    (keyword/BM25). alpha=0.75 is a dense-leaning default for natural-
    language queries on document-style content.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    scaled_sparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    scaled_dense = [v * alpha for v in dense]
    return scaled_dense, scaled_sparse


class VoiceCartRetriever:
    def __init__(self, alpha: float = 0.75):
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not found in .env")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in .env")
        if not PINECONE_INDEX_NAME:
            raise ValueError("PINECONE_INDEX_NAME not found in .env")
        if not os.path.exists(BM25_PARAMS_PATH):
            raise FileNotFoundError(
                f"bm25_params.json not found at {BM25_PARAMS_PATH}. "
                f"Run app/ingest/upload.py first to generate it."
            )

        self.co = cohere.Client(COHERE_API_KEY)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX_NAME)

        self.bm25 = BM25Encoder()
        self.bm25.load(BM25_PARAMS_PATH)

        self.alpha = alpha

    def retrieve(self, query: str, namespace: str, top_k: int = 3) -> list[dict]:
        """Query a specific namespace using hybrid (dense + sparse) search."""
        if namespace not in VALID_NAMESPACES:
            raise ValueError(
                f"Invalid namespace '{namespace}'. Must be one of {VALID_NAMESPACES}."
            )

        try:
            dense_vec = self.co.embed(
                texts=[query],
                model="embed-english-v3.0",
                input_type="search_query",
            ).embeddings[0]
        except Exception as e:
            print(f"[retriever] Cohere embedding error: {e}")
            raise

        sparse_vec = self.bm25.encode_queries(query)

        scaled_dense, scaled_sparse = hybrid_scale(dense_vec, sparse_vec, self.alpha)

        try:
            results = self.index.query(
                vector=scaled_dense,
                sparse_vector=scaled_sparse,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
            )
        except Exception as e:
            print(f"[retriever] Pinecone query error for namespace '{namespace}': {e}")
            raise

        return [
            {
                "id": match.id,
                "score": match.score,
                "title": match.metadata.get("title", ""),
                "text": match.metadata.get("text", ""),
            }
            for match in results.matches
        ]


if __name__ == "__main__":
    retriever = VoiceCartRetriever(alpha=0.75)

    test_queries = [
        # product-info: new sections 11-15
        ("What colors are available?", "product-info"),
        ("How much do the earbuds cost?", "product-info"),
        ("How much does the whole package weigh?", "product-info"),
        ("Can I update the firmware?", "product-info"),
        ("Can I trigger Siri with these?", "product-info"),

        # troubleshooting: new sections 6-15
        ("My earbuds keep falling out when I run", "troubleshooting"),
        ("There's an echo when I'm on a call", "troubleshooting"),
        ("One earbud is louder than the other", "troubleshooting"),
        ("The case won't charge my earbuds", "troubleshooting"),
        ("The LED on the case isn't lighting up", "troubleshooting"),
        ("My earbud feels hot", "troubleshooting"),
        ("Triple tap isn't triggering my voice assistant", "troubleshooting"),
        ("There's a delay between video and audio", "troubleshooting"),
        ("My earbuds are stuck in pairing mode", "troubleshooting"),
        ("The case lid won't close", "troubleshooting"),

        # policies: new sections 9-15
        ("How do I pay for my order?", "policies"),
        ("Can I cancel my order?", "policies"),
        ("How do I check my order status?", "policies"),
        ("How do I file a warranty claim?", "policies"),
        ("Do you ship internationally?", "policies"),
        ("Can I get a bulk discount?", "policies"),
        ("Can I order this as a gift?", "policies"),

        # limitations: new sections 6-15
        ("Is drop damage covered?", "limitations"),
        ("What if my earbuds get stolen?", "limitations"),
        ("Can I use these as hearing protection?", "limitations"),
        ("Are these OK to use at the beach with lots of sand?", "limitations"),
        ("Can I connect to 3 devices at once?", "limitations"),
        ("Can I charge the case wirelessly?", "limitations"),
        ("Can I use these with a wired cable if the battery dies?", "limitations"),
        ("Does the voice assistant work without phone signal?", "limitations"),
        ("Can I leave these in a hot car?", "limitations"),
        ("Does the case battery age separately from the earbuds?", "limitations"),
    ]

    for query, namespace in test_queries:
        print(f"\nQuery: '{query}'  |  Namespace: '{namespace}'")
        results = retriever.retrieve(query, namespace, top_k=3)
        for r in results:
            print(f"  [{r['score']:.4f}] {r['title']}")
            print(f"    {r['text'][:80]}...")