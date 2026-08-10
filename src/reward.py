from config import SMILE_REWARD_RATE, SMILE_MAX_REWARD

def calc_reward(smile_score: float, amount: int) -> dict:
    """
    smile_score(0~100) → 적립률(%) → 실제 적립 포인트.
    적립률 = smile_score * SMILE_REWARD_RATE, 최대 SMILE_MAX_REWARD(%) 캡.
    amount: 결제 금액(원)
    """
    rate = min(smile_score * SMILE_REWARD_RATE, SMILE_MAX_REWARD)  # %
    points = int(amount * rate / 100)                              # 원 단위 절사
    return {"reward_rate": round(rate, 1), "reward_points": points}