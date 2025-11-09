from flask import Blueprint, jsonify
from pymongo import MongoClient
import os

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["agri_llava"]
history_collection = db["history"]

history_bp = Blueprint('history', __name__)

@history_bp.route("/clear", methods=["POST"])
def clear_history():
    """Clear all history entries."""
    try:
        # Delete all history entries
        result = history_collection.delete_many({})
        return jsonify({
            "message": "History cleared successfully",
            "entries_deleted": result.deleted_count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500