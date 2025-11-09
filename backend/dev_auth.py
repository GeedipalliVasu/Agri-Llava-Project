"""
Lightweight auth server for local frontend development.
Stores users in a local SQLite DB (dev_auth.db) and issues JWTs.
This avoids starting the full `app.py` (which requires heavy ML libs) while you test login/signup flows.

Run:
  python dev_auth.py

It listens on port 5000 by default to match the frontend's calls.
"""
import os
import sqlite3
import time
from flask import Flask, request, jsonify
import jwt
import bcrypt

DB_PATH = os.path.join(os.path.dirname(__file__), "dev_auth.db")
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", "604800"))

app = Flask(__name__)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password BLOB NOT NULL,
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

def hash_password(plain: str) -> bytes:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt())

def verify_password(plain: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed)
    except Exception:
        return False

def issue_jwt(user_id: int, email: str) -> str:
    payload = {"sub": str(user_id), "email": email, "exp": int(time.time()) + JWT_EXPIRE_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

@app.route("/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not name or not email or not password:
        return jsonify({"error": "name, email, password are required"}), 400

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            return jsonify({"error": "Email already registered"}), 409
        hashed = hash_password(password)
        cur.execute(
            "INSERT INTO users (name, email, password, created_at) VALUES (?, ?, ?, ?)",
            (name, email, hashed, int(time.time())),
        )
        conn.commit()
        user_id = cur.lastrowid
        token = issue_jwt(user_id, email)
        return jsonify({"token": token, "user": {"id": str(user_id), "name": name, "email": email}})
    finally:
        conn.close()

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, name, password FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "Invalid credentials"}), 401
        stored_hashed = row[2]
        # stored_hashed is bytes stored by sqlite3; ensure bytes type
        if isinstance(stored_hashed, str):
            stored_hashed = stored_hashed.encode('latin1')
        if not verify_password(password, stored_hashed):
            return jsonify({"error": "Invalid credentials"}), 401
        token = issue_jwt(row[0], email)
        return jsonify({"token": token, "user": {"id": str(row[0]), "name": row[1], "email": email}})
    finally:
        conn.close()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "db": os.path.exists(DB_PATH)})

if __name__ == "__main__":
    init_db()
    print("Starting dev auth server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)
