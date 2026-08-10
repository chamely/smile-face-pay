import pickle, os
from config import DB_PATH

def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)

def save_db(db):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

def register_user(user_id, embedding, smile_baseline):
    db = load_db()
    db[user_id] = {"embedding": embedding, "smile_baseline": smile_baseline}
    save_db(db)