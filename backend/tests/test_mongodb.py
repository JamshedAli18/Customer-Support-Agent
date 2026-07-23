"""
Standalone test for MongoDB Atlas connection — verifying it works
before wiring it into the real application.
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "voicecart_db")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI not found in .env")

client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("PASS: Successfully connected to MongoDB Atlas")
except Exception as e:
    print(f"FAIL: Connection error: {e}")
    raise

db = client[MONGODB_DB_NAME]
test_collection = db["connection_test"]

result = test_collection.insert_one({"test": "VoiceCart MongoDB connection test", "status": "success"})
print(f"PASS: Inserted test document with id: {result.inserted_id}")

found = test_collection.find_one({"_id": result.inserted_id})
print(f"PASS: Retrieved test document: {found}")

test_collection.delete_one({"_id": result.inserted_id})
print("PASS: Cleaned up test document")

client.close()
print("\nAll MongoDB connection tests passed.")