"""
Lightweight Mongo-backed auth server for local development.
Provides /auth/signup and /auth/login and stores users in MongoDB.

Usage:
  - Ensure MongoDB is running (default: mongodb://localhost:27017)
  - Install dependencies: pip install flask pymongo bcrypt PyJWT flask-cors
  - Run: python mongo_auth.py

The server listens on port 5000 to match the frontend's expectations.
"""
import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
import jwt

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "agri_llava")
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", "604800"))

app = Flask(__name__)
CORS(app)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users = db["users"]

def issue_jwt(user_id: str, email: str) -> str:
    payload = {"sub": str(user_id), "email": email, "exp": int(time.time()) + JWT_EXPIRE_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

@app.route('/auth/signup', methods=['POST'])
def signup():
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not name or not email or not password:
        return jsonify({"error": "name, email, password are required"}), 400
    if users.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 409
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    doc = {"name": name, "email": email, "password": hashed, "created_at": int(time.time())}
    res = users.insert_one(doc)
    token = issue_jwt(str(res.inserted_id), email)
    return jsonify({"token": token, "user": {"id": str(res.inserted_id), "name": name, "email": email}})

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    user = users.find_one({"email": email})
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401
    stored = user.get('password')
    # pymongo returns bytes for Binary; ensure bytes
    try:
        ok = bcrypt.checkpw(password.encode('utf-8'), stored)
    except Exception:
        ok = False
    if not ok:
        return jsonify({"error": "Invalid credentials"}), 401
    token = issue_jwt(str(user['_id']), email)
    return jsonify({"token": token, "user": {"id": str(user['_id']), "name": user.get('name',''), "email": email}})

@app.route('/health', methods=['GET'])
def health():
    try:
        # quick ping
        client.admin.command('ping')
        return jsonify({"ok": True, "mongo": True})
    except Exception as e:
        return jsonify({"ok": False, "mongo": False, "error": str(e)}), 500

if __name__ == '__main__':
    print(f"Starting mongo auth server on http://127.0.0.1:5000 (MONGO_URI={MONGO_URI})")
    app.run(host='127.0.0.1', port=5000)
