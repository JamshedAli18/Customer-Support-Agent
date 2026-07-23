from pinecone_text.sparse import BM25Encoder

# Small sample corpus, similar in spirit to your KB content
corpus = [
    "The ShopNest Pulse has an IPX5 water resistance rating.",
    "Battery life is up to 6 hours with ANC off.",
    "The return window is 30 days from delivery date.",
]

bm25 = BM25Encoder()
bm25.fit(corpus)

doc_sparse = bm25.encode_documents("Battery life is up to 6 hours with ANC off.")
query_sparse = bm25.encode_queries("What is the IPX rating?")

print("Document sparse vector:")
print(doc_sparse)
print("\nQuery sparse vector:")
print(query_sparse)
print("\nSuccess — BM25Encoder is working")