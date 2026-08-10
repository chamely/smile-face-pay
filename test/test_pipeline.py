from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from src.reward import calc_reward


def test_emotion_labels():
    """설치된 버전의 클래스 순서에서 Happiness 인덱스 확인용"""
    fer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
    print("labels:", fer.idx_to_class)  # 여기서 'Happiness' 인덱스 확인 후 expression.py 반영

def test_reward_cap():
    assert calc_reward(100, 10000)["reward_rate"] == 10.0
    assert calc_reward(100, 10000)["reward_points"] == 1000
    assert calc_reward(50, 10000)["reward_points"] == 500
    assert calc_reward(0, 10000)["reward_points"] == 0