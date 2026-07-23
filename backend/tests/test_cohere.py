import os
from dotenv import load_dotenv
import cohere

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found in .env")

co = cohere.Client(api_key)

response = co.embed(
    texts=["This is a test sentence for ShopNest Pulse earbuds."],
    model="embed-english-v3.0",
    input_type="search_document",
)

embedding = response.embeddings[0]
print(f"Success — embedding generated with {len(embedding)} dimensions")
print(f"First 5 values: {embedding[:5]}")