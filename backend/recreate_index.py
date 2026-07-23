import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=api_key)

if pc.has_index(index_name):
    print(f"Deleting existing index '{index_name}'...")
    pc.delete_index(index_name)
    print("Deleted.")

print(f"Creating index '{index_name}' with dotproduct metric (required for hybrid search)...")
pc.create_index(
    name=index_name,
    dimension=1024,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
print("Index recreated successfully.")

index_info = pc.describe_index(index_name)
print(f"Index details: {index_info}")