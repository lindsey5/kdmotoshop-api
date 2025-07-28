import os
from pymongo import MongoClient, errors

def get_products_collection():
    try:
        client = MongoClient(os.environ.get("DB_URI"))
        client.admin.command("ping")  # Check connection

        db = client["kd-motoshop"]
        collection = db["products"]

        print("✅ Connected to MongoDB successfully.")
        return collection

    except errors.ServerSelectionTimeoutError as err:
        print("❌ Failed to connect to MongoDB:", err)

    except Exception as e:
        print("❌ An unexpected error occurred:", e)

    return None  # Explicitly return None if connection fails
