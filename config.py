RECOGNITION_THRESHOLD = 0.35      # 코사인 유사도 임계값 (ArcFace 기준 튜닝 필요)
SMILE_REWARD_RATE = 0.1     # smile_score * rate = 적립률(%)
SMILE_MAX_REWARD = 10.0     # 최대 적립률(%)
DEVICE = "cuda"                   # or "cpu"
DB_PATH = "data/registered/users.pkl"