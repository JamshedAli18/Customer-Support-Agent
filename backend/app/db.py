"""
MongoDB Atlas connection helper. Provides shared collection handles
for the whole application.
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

_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
db = _client[MONGODB_DB_NAME]

inventory_collection = db["inventory"]
tickets_collection = db["tickets"]
warranty_claims_collection = db["warranty_claims"]
sessions_collection = db["sessions"]
orders_collection = db["orders"]