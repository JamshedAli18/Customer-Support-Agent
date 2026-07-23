"""
One-time script to seed the MongoDB 'inventory' collection with
ShopNest Pulse product/color data. Safe to re-run — it clears and
re-inserts, so it won't create duplicates.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from app.db import inventory_collection

PRODUCT_NAME = "ShopNest Pulse"

INVENTORY_DATA = [
    {"product": PRODUCT_NAME, "color": "Matte Black", "quantity": 42, "price": 79.99},
    {"product": PRODUCT_NAME, "color": "Pearl White", "quantity": 0, "price": 79.99},
    {"product": PRODUCT_NAME, "color": "Slate Blue", "quantity": 17, "price": 79.99},
    {"product": PRODUCT_NAME, "color": "Red", "quantity": 25, "price": 79.99},
    {"product": PRODUCT_NAME, "color": "Green", "quantity": 8, "price": 79.99},
    {"product": PRODUCT_NAME, "color": "Purple", "quantity": 0, "price": 79.99},
    {"product": PRODUCT_NAME, "color": "Silver", "quantity": 30, "price": 79.99},
    {"product": PRODUCT_NAME, "color": "Gold", "quantity": 5, "price": 79.99},
    {"product": PRODUCT_NAME, "color": "Navy", "quantity": 14, "price": 79.99},
    {"product": PRODUCT_NAME, "color": "Pink", "quantity": 0, "price": 79.99},
]


def seed_inventory():
    inventory_collection.delete_many({"product": PRODUCT_NAME})
    result = inventory_collection.insert_many(INVENTORY_DATA)
    print(f"Seeded {len(result.inserted_ids)} inventory documents")

    for doc in inventory_collection.find({"product": PRODUCT_NAME}):
        print(f"  {doc['color']}: quantity={doc['quantity']}, price=${doc['price']}")


if __name__ == "__main__":
    seed_inventory()