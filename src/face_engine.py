# src/face_engine.py  (detection + recognition 통합 로더)
from insightface.app import FaceAnalysis
import numpy as np
from config import RECOGNITION_THRESHOLD

class FaceEngine:
    def __init__(self, det_size=(320, 320), providers=None):
        # providers 예:
        #   CPU        → ["CPUExecutionProvider"]
        #   CUDA(GPU)  → ["CUDAExecutionProvider", "CPUExecutionProvider"]
        #   TensorRT   → ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        # ctx_id는 CPU면 -1, GPU EP면 0
        ctx = -1 if providers[0] == "CPUExecutionProvider" else 0
        self.app.prepare(ctx_id=ctx, det_size=det_size)

    def get_face(self, image_bgr):
        """가장 큰 얼굴 1개. face 객체에 kps, normed_embedding 포함."""
        faces = self.app.get(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    def embedding(self, face):
        return face.normed_embedding  # 512-d, 정규화됨

    def match(self, emb, db):
        best_id, best_sim = None, -1.0
        for uid, data in db.items():
            sim = float(np.dot(emb, data["embedding"]))
            if sim > best_sim:
                best_id, best_sim = uid, sim
        return (best_id if best_sim >= RECOGNITION_THRESHOLD else None), best_sim

    def assess_quality(self, face, frame_shape, strict=True):
        """품질 게이트. strict=True 등록용(엄격), False 결제용(완화)."""
        H, W = frame_shape[:2]
        x1, y1, x2, y2 = face.bbox

        # 1) 검출 신뢰도
        min_det = 0.60 if strict else 0.50
        if float(face.det_score) < min_det:
            return False, "얼굴이 뚜렷하지 않습니다"

        # 2) bbox가 프레임 경계에 붙음 → 잘림 (등록·결제 공통)
        m = 5
        if x1 <= m or y1 <= m or x2 >= W - m or y2 >= H - m:
            return False, "얼굴이 화면 안에 다 들어오도록 해주세요"

        # 3) 랜드마크(눈·코·입) 경계 체크 → 얼굴이 화면 밖으로 잘림 직접 차단
        #    bbox가 안쪽에 좁게 잡혀도, kps가 가장자리로 가면 잡힘
        edge = 0.06
        for (px, py) in face.kps:
            if px < W * edge or px > W * (1 - edge) or py < H * edge or py > H * (1 - edge):
                return False, "얼굴 전체가 화면 안에 들어오게 해주세요"

        # 4) 얼굴 크기 (등록·결제 공통, 임계값만 차등)
        min_area = 0.08 if strict else 0.05
        if (x2 - x1) * (y2 - y1) < min_area * W * H:
            return False, "얼굴을 화면에 더 크게 맞춰주세요"

        # 5) 정면/좌우대칭 (등록·결제 공통, 임계값만 차등)
        kps = face.kps
        nose_x = kps[2][0]
        d_left = abs(nose_x - kps[0][0])
        d_right = abs(kps[1][0] - nose_x)
        ratio = min(d_left, d_right) / max(d_left, d_right, 1e-6)
        sym_min = 0.55 if strict else 0.42
        if ratio < sym_min:
            return False, "얼굴이 정면으로 보이지 않습니다"

        return True, ""