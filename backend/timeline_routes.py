from datetime import datetime
from flask import Blueprint, jsonify, request
from pymongo import MongoClient
from bson.objectid import ObjectId
import os

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["agri_llava"]
timelines_collection = db["timelines"]

timeline_bp = Blueprint('timeline', __name__)

# crops persistence collection (simple per-farmer crop list)
crops_collection = db.get_collection("crops")


@timeline_bp.route("/crops", methods=["GET"])
def get_crops():
    farmer_id = request.args.get("farmer_id", "farmer1")
    try:
        docs = list(crops_collection.find({"farmer_id": farmer_id}, {"_id": 0, "name": 1}))
        crops = [d.get("name") for d in docs if d.get("name")]
        return jsonify({"farmer_id": farmer_id, "crops": crops})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@timeline_bp.route("/crops", methods=["POST"])
def add_crop():
    try:
        data = request.get_json(silent=True) or {}
        farmer_id = data.get("farmer_id", "farmer1")
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        doc = {"farmer_id": farmer_id, "name": name, "created_at": datetime.utcnow()}
        crops_collection.update_one({"farmer_id": farmer_id, "name": name}, {"$setOnInsert": doc}, upsert=True)
        return jsonify({"ok": True, "name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@timeline_bp.route("/entry/<entry_id>/disease", methods=["PUT"])
def update_entry_disease(entry_id):
    try:
        data = request.get_json(silent=True) or {}
        disease = data.get("disease")
        if disease is None:
            return jsonify({"error": "disease value required"}), 400
        res = timelines_collection.update_one({"_id": ObjectId(entry_id)}, {"$set": {"disease": disease}})
        if res.matched_count == 0:
            return jsonify({"error": "entry not found"}), 404
        return jsonify({"ok": True, "entry_id": entry_id, "disease": disease})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@timeline_bp.route("/clear", methods=["POST"])
def clear_timeline():
    """Clear all entries from a timeline."""
    try:
        data = request.get_json(silent=True) or {}
        farmer_id = data.get("farmer_id", "farmer1")
        plant_id = data.get("plant_id", None)
        crop = data.get("crop", None)

        # Delete entries for this farmer (and optionally specific plant)
        query = {"farmer_id": farmer_id}
        if plant_id:
            query["plant_id"] = plant_id
        if crop:
            query["crop"] = crop
        result = timelines_collection.delete_many(query)
        return jsonify({
            "message": "Timeline cleared successfully",
            "entries_deleted": result.deleted_count,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@timeline_bp.route("/end", methods=["POST"])
def end_timeline():
    """Mark a timeline as completed and archive it."""
    try:
        data = request.get_json(silent=True) or {}
        farmer_id = data.get("farmer_id", "farmer1")
        plant_id = data.get("plant_id", None)
        crop = data.get("crop", None)

        # Update the timeline status to completed
        query = {"farmer_id": farmer_id}
        if plant_id:
            query["plant_id"] = plant_id
        if crop:
            query["crop"] = crop
        result = timelines_collection.update_many(
            query,
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                    "archived": True,
                }
            },
        )

        if result.modified_count > 0:
            return jsonify({
                "message": "Timeline ended successfully",
                "entries_updated": result.modified_count,
            })
        return jsonify({"message": "No active timeline found to end"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500