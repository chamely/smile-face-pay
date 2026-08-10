class SmileLiveness:
    """
    간이 액티브 라이브니스:
    무표정 프레임 → 웃는 프레임으로의 smile score '상승 변화'를 요구.
    정지된 사진은 표정 변화를 만들 수 없으므로 기본적 위조를 걸러냄.
    """
    def __init__(self, delta_threshold=25.0):
        self.delta_threshold = delta_threshold

    def verify(self, score_before: float, score_after: float) -> bool:
        return (score_after - score_before) >= self.delta_threshold