import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

if not api_key:
    raise ValueError("PINECONE_API_KEY not found in .env")
if not index_name:
    raise ValueError("PINECONE_INDEX_NAME not found in .env")

pc = Pinecone(api_key=api_key)

if not pc.has_index(index_name):
    print(f"Index '{index_name}' does not exist. Creating it now...")
    pc.create_index(
        name=index_name,
        dimension=1024,  # matches Cohere embed-english-v3.0
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists.")

index_info = pc.describe_index(index_name)
print(f"Index details: {index_info}")