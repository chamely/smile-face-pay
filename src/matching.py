import numpy as np
from config import RECOGNITION_THRESHOLD

def match(emb, db):
    """1:N 코사인 매칭. emb는 정규화된 벡터 가정."""
    best_id, best_sim = None, -1.0
    for uid, data in db.items():
        sim = float(np.dot(emb, data["embedding"]))
        if sim > best_sim:
            best_id, best_sim = uid, sim
    return (best_id if best_sim >= RECOGNITION_THRESHOLD else None), best_sim

def smile_score(raw_happy, baseline):
    """raw happy 확률(0~1)을 baseline 기준 상대 정규화 → 0~100."""
    adjusted = max(0.0, raw_happy - baseline) / max(1e-6, (1.0 - baseline))
    return round(float(np.clip(adjusted, 0.0, 1.0)) * 100, 1)