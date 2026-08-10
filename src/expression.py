import urllib.request
import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

class SmileScorer:
    def __init__(self, model_name="enet_b0_8_best_afew"):
        # CPU에서 가볍게 도는 경량 ONNX 모델
        self.fer = HSEmotionRecognizer(model_name=model_name)
        # HSEmotion 8-class 순서에서 'Happiness' 인덱스
        self.happy_idx = 4

    def raw_happy_prob(self, aligned_face_bgr):
        """
        aligned_face_bgr: 정렬된 얼굴 crop (BGR, 112x112 등)
        return: happy 확률 0~1
        """
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        _, scores = self.fer.predict_emotions(rgb, logits=False)  # scores: softmax 확률
        return float(scores[self.happy_idx])

    def score(self, aligned_face_bgr, baseline=0.0):
        """
        baseline: 등록 시 저장한 무표정 happy 확률
        상대값 정규화로 사용자별 편차 보정 → 0~100
        """
        raw = self.raw_happy_prob(aligned_face_bgr)
        adjusted = max(0.0, raw - baseline) / max(1e-6, (1.0 - baseline))
        return round(float(np.clip(adjusted, 0.0, 1.0)) * 100, 1)