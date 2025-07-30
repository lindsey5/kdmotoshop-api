import os
from pymongo import MongoClient

db = None

def load_db():
    global db
    try:
        client = MongoClient(os.environ.get("DB_URI"))
        client.admin.command("ping")  # Check connection

        db = client["kd-motoshop"]

        print("✅ Connected to MongoDB successfully.")

    except Exception as e:
        print("❌ An unexpected error occurred:", e)

    return None 

def get_products_collection():
    global db
    if db is None:
        load_db()
    try:
        collection = db["products"]

        print("✅ Connected to MongoDB successfully.")
        return collection

    except Exception as e:
        print("❌ An unexpected error occurred:", e)

    return None  # Explicitly return None if connection fails
