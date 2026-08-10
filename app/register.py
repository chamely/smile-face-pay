import cv2
from src.face_engine import FaceEngine
from src.expression import SmileScorer
from src.utils import align_face
from src.database import load_db, register_user

def capture_frame(prompt):
    cap = cv2.VideoCapture(0)
    print(prompt, "→ 스페이스바로 캡처, ESC 취소")
    frame = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cv2.imshow("capture", f)
        k = cv2.waitKey(1)
        if k == 32:      # space
            frame = f.copy(); break
        elif k == 27:    # esc
            break
    cap.release(); cv2.destroyAllWindows()
    return frame

def register(user_id):
    engine, scorer = FaceEngine(), SmileScorer()
    db = load_db()

    neutral = capture_frame("무표정을 지어주세요")
    face = engine.get_face(neutral)
    if face is None:
        print("얼굴 미검출"); return
    emb = engine.embedding(face)

    # ① 얼굴 중복 검사: 이미 등록된 얼굴인지 확인
    matched_id, sim = engine.match(emb, db)
    if matched_id is not None:
        print(f"[중복] 이 얼굴은 이미 '{matched_id}'로 등록되어 있습니다. (similarity={sim:.3f})")
        if matched_id == user_id:
            ans = input(f"→ '{user_id}' 정보를 새로 갱신할까요? (y/n): ")
        else:
            ans = input(f"→ 기존 '{matched_id}'를 삭제하고 '{user_id}'로 새로 등록할까요? (y/n): ")
        if ans.lower() != "y":
            print("등록 취소."); return
        # 다른 ID로 재등록하는 경우 기존 항목 삭제
        if matched_id != user_id:
            del db[matched_id]

    # ② user_id 중복 검사 (얼굴은 다른데 같은 ID를 쓰려는 경우)
    elif user_id in db:
        ans = input(f"[중복] ID '{user_id}'가 이미 존재합니다. 덮어쓸까요? (y/n): ")
        if ans.lower() != "y":
            print("등록 취소."); return

    # 등록 진행
    aligned = align_face(neutral, face.kps)
    baseline = scorer.raw_happy_prob(aligned)
    # del 반영을 위해 db를 직접 갱신 후 저장
    db[user_id] = {"embedding": emb, "smile_baseline": baseline}
    from src.database import save_db
    save_db(db)
    print(f"[등록 완료] {user_id} | smile_baseline={baseline:.3f}")

if __name__ == "__main__":
    import sys
    register(sys.argv[1] if len(sys.argv) > 1 else "user1")