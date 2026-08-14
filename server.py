"""
Smile Face Pay — GPU 추론 서버 (데스크탑에서 실행)
프레임을 받아 검출+임베딩+표정+품질을 반환. Gradio 클라이언트가 호출.

실행 예:
  CPU:      EP=cpu  uvicorn server:app --host 0.0.0.0 --port 8000
  GPU:      EP=cuda uvicorn server:app --host 0.0.0.0 --port 8000
  TensorRT: EP=trt  uvicorn server:app --host 0.0.0.0 --port 8000
"""
import os, io, base64, time
import numpy as np
import cv2
from fastapi import FastAPI
from pydantic import BaseModel

from src.face_engine import FaceEngine
from src.expression import SmileScorer
from src.utils import align_face

# ---- EP 선택 (환경변수 EP로 스위치) ----
_EP = os.environ.get("EP", "cpu").lower()
_PROVIDERS = {
    "cpu":  ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "trt":  ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}[_EP]

print(f"[server] loading models with EP={_EP} → {_PROVIDERS[0]}")
_engine = FaceEngine(providers=_PROVIDERS)
_scorer = SmileScorer()

app = FastAPI(title="Smile Face Pay Inference Server")


class FrameRequest(BaseModel):
    image_b64: str          # BGR 이미지를 PNG로 인코딩 후 base64
    strict: bool = True     # 품질 게이트 강도(등록=True, 결제=False)


def _decode(image_b64: str):
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)   # BGR


@app.get("/health")
def health():
    return {"status": "ok", "ep": _EP}


@app.post("/analyze")
def analyze(req: FrameRequest):
    t0 = time.perf_counter()
    frame_bgr = _decode(req.image_b64)
    if frame_bgr is None:
        return {"face": False, "reason": "decode_failed"}

    face = _engine.get_face(frame_bgr)
    if face is None:
        return {"face": False, "reason": "no_face",
                "latency_ms": (time.perf_counter() - t0) * 1000}

    ok, reason = _engine.assess_quality(face, frame_bgr.shape, strict=req.strict)
    if not ok:
        return {"face": True, "quality_ok": False, "reason": reason,
                "latency_ms": (time.perf_counter() - t0) * 1000}

    emb = _engine.embedding(face)
    aligned = align_face(frame_bgr, face.kps)
    raw_happy = float(_scorer.raw_happy_prob(aligned))

    return {"face": True, "quality_ok": True,
            "emb": emb.tolist(), "raw_happy": raw_happy,
            "latency_ms": (time.perf_counter() - t0) * 1000}