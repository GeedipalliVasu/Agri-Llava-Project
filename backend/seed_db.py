"""
Seed script to create a test user and sample history/timeline entries in MongoDB.

Usage:
  Set MONGO_URI environment variable or pass as first arg.
  python seed_db.py [mongodb_uri]

This helps when you connect MongoDB from VSCode — run this to populate a test user
and a couple of history/timeline records linked to that user.
"""
import os
import sys
import time
from pymongo import MongoClient
import bcrypt
from bson.objectid import ObjectId

DEFAULT_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

def get_client(uri: str):
    return MongoClient(uri)

def ensure_user(db, name, email, password):
    users = db['users']
    existing = users.find_one({'email': email})
    if existing:
        print(f"User with email {email} already exists: {existing['_id']}")
        return existing['_id']
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    res = users.insert_one({'name': name, 'email': email, 'password': hashed, 'created_at': int(time.time())})
    print(f"Inserted user {email} -> id={res.inserted_id}")
    return res.inserted_id

def insert_sample_history(db, user_id):
    history = db['history']
    sample = {
        'filename': 'sample_leaf.jpg',
        'prediction': 'Potato___Late_blight',
        'confidence': 0.87,
        'confidence_percent': '87.00%',
        'probabilities': {
            'Potato___Early_blight': 0.05,
            'Potato___Late_blight': 0.87,
            'Potato___healthy': 0.08
        },
        'heatmap': None,
        'segmentation': None,
        'legend': [{'color': '#FF0000', 'label': 'Diseased region'}],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'user_id': ObjectId(user_id) if not isinstance(user_id, ObjectId) else user_id
    }
    res = history.insert_one(sample)
    print(f"Inserted history entry id={res.inserted_id}")

def insert_sample_timeline(db, user_id):
    timelines = db['timelines']
    sample = {
        'farmer_id': 'farmer1',
        'filename': 'timeline_leaf.jpg',
        'prediction': 'Potato___Early_blight',
        'confidence': 0.65,
        'severity_percent': 12.34,
        'heatmap': None,
        'segmentation': None,
        'week_index': 1,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'user_id': ObjectId(user_id) if not isinstance(user_id, ObjectId) else user_id
    }
    res = timelines.insert_one(sample)
    print(f"Inserted timeline entry id={res.inserted_id}")

def main():
    uri = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URI
    print(f"Connecting to MongoDB at {uri}")
    client = get_client(uri)
    db = client.get_database(os.getenv('MONGO_DB', 'agri_llava'))

    # Replace these test credentials as you like
    test_name = 'Dev User'
    test_email = 'devuser@example.com'
    test_password = 'Password123'

    user_id = ensure_user(db, test_name, test_email, test_password)
    insert_sample_history(db, user_id)
    insert_sample_timeline(db, user_id)

    print('Seeding complete. You can verify in MongoDB Compass.')

if __name__ == '__main__':
    main()
