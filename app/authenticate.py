import cv2
from src.face_engine import FaceEngine
from src.expression import SmileScorer
from src.utils import align_face
from src.reward import calc_reward
from src.liveness import SmileLiveness
from src.database import load_db
from app.register import capture_frame

def authenticate(amount=15000):
    engine, scorer = FaceEngine(), SmileScorer()
    liveness = SmileLiveness(delta_threshold=25.0)
    db = load_db()

    # 1컷: 무표정 (신원 확인 + smile before)
    neutral = capture_frame("무표정을 지어주세요")
    face_n = engine.get_face(neutral)
    if face_n is None:
        return {"status": "no_face"}

    emb = engine.embedding(face_n)
    user_id, sim = engine.match(emb, db)
    if user_id is None:
        return {"status": "unknown", "similarity": round(sim, 3)}

    baseline = db[user_id]["smile_baseline"]
    score_before = scorer.score(align_face(neutral, face_n.kps), baseline)

    # 2컷: 웃는 얼굴 (smile after)
    smiling = capture_frame("활짝 웃어주세요")
    face_s = engine.get_face(smiling)
    if face_s is None:
        return {"status": "no_face"}
    score_after = scorer.score(align_face(smiling, face_s.kps), baseline)

    # liveness: 표정 변화 delta 검증 (정지 사진 방어)
    if not liveness.verify(score_before, score_after):
        return {"status": "liveness_failed",
                "before": score_before, "after": score_after}

    reward = calc_reward(score_after, amount)  # amount: 결제 금액
    return {
        "status": "ok",
        "user_id": user_id,
        "similarity": round(sim, 3),
        "smile_score": score_after,
        "amount": amount,
        "reward_rate": reward["reward_rate"],
        "reward_points": reward["reward_points"],
    }

if __name__ == "__main__":
    print(authenticate(amount=15000))