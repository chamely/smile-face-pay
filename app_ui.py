"""
Smile Face Pay — Gradio 단말기형 데모 UI (v5, 서버 추론)
- 좌: 단말기(브랜드 + 웹캠 + 상태바 + 등록/결제 버튼)
- 우: 입력 패널(이름, 결제 금액) + 적립 결과 카드
- 추론은 원격 GPU 서버(FastAPI)가 담당. 이 앱은 프레임을 서버로 보내고
  결과(임베딩·표정·품질)를 받아 신원매칭/적립/DB를 처리하는 클라이언트.
- 깜빡임 방지: 웹캠 stream은 세션(State)만 갱신하고,
  화면 HTML은 Timer가 '값이 바뀔 때만' 다시 그림.
"""
import time
import base64
import numpy as np
import cv2
import requests
import gradio as gr

from config import RECOGNITION_THRESHOLD
from src.matching import match, smile_score
from src.reward import calc_reward
from src.database import load_db, save_db

# ------------------------------------------------------------------
# 추론 서버 주소
#   - 데스크탑에서 로컬 검증: "http://localhost:8000"
#   - 노트북에서 데스크탑 접속: "http://172.30.1.3:8000"
# ------------------------------------------------------------------
SERVER_URL = "http://172.30.1.3:8000"
BACKEND = f"server({SERVER_URL})"


def analyze(frame_bgr, strict=True):
    """프레임을 서버로 보내 추론 결과를 받는다."""
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        return {"error": "encode_failed"}
    b64 = base64.b64encode(buf).decode("ascii")
    try:
        r = requests.post(f"{SERVER_URL}/analyze",
                          json={"image_b64": b64, "strict": strict},
                          timeout=10)
        r.raise_for_status()
        return r.json()   # {face, quality_ok?, emb?, raw_happy?, latency_ms?, reason?}
    except Exception as e:
        return {"error": str(e)}

def crop_to_display(frame_rgb):
    """화면 표시(4:5)와 동일하게 원본 프레임을 center-crop."""
    h, w = frame_rgb.shape[:2]
    target_ratio = 4 / 5          # CSS aspect-ratio와 일치
    cur_ratio = w / h
    if cur_ratio > target_ratio:  # 너무 넓음 → 좌우 자름
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        return frame_rgb[:, x0:x0 + new_w]
    else:                          # 너무 높음 → 위아래 자름
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        return frame_rgb[y0:y0 + new_h, :]

SMILE_FRAMES = 6          # 웃음 측정에 모을 프레임 수(처리 프레임 기준)

# ------------------------------------------------------------------
# 세션 상태
# ------------------------------------------------------------------
def new_session(mode, user_id="", amount=15000):
    return {"mode": mode, "stage": "await_neutral", "user_id": user_id,
            "amount": amount, "neutral_raw": None, "emb_neutral": None,
            "baseline": None, "conflict_id": None,
            "best_smile": 0, "smile_frames": 0,
            "target_pct": 0, "display_pct": 0,
            "msg": "무표정을 지어주세요", "result": None,
            "last_render": None, "started": time.time()}


def step(frame, sess):
    """stream 프레임 처리 → 세션만 갱신(화면 출력은 Timer가 담당)."""
    if sess is None:
        return sess
    stage = sess.get("stage")

    if stage == "await_overwrite":
        sess["msg"] = "이미 등록된 얼굴입니다. 새로 등록할지 선택해주세요."
        return sess
    if stage in ("done", None):
        return sess
    if frame is None:
        sess["msg"] = "카메라 신호 없음"
        return sess

    # 단계별 품질 게이트 강도: 등록=엄격, 결제 신원확인=완화
    strict = (sess["mode"] == "register")
    frame_cropped = crop_to_display(frame)      # 화면과 동일 영역만
    info = analyze(frame_cropped[:, :, ::-1].copy(), strict=strict)

    if info.get("error"):
        sess["msg"] = f"서버 연결 오류: {info['error'][:40]}"
        return sess
    if not info.get("face"):
        sess["msg"] = "얼굴을 화면 중앙에 맞춰주세요."
        return sess
    if info.get("quality_ok") is False:
        sess["msg"] = info.get("reason", "얼굴 품질을 확인해주세요")
        return sess

    # 서버가 준 임베딩(리스트)을 numpy로 복원
    emb = np.array(info["emb"], dtype=np.float32)
    raw_happy = info["raw_happy"]

    db = load_db()

    # 1) 무표정 단계
    if stage == "await_neutral":
        sess["neutral_raw"] = raw_happy
        sess["emb_neutral"] = emb

        if sess["mode"] == "pay":
            uid, sim = match(emb, db)
            if uid is None:
                sess["stage"] = "done"
                sess["result"] = {"status": "unknown", "similarity": round(sim, 3)}
                sess["msg"] = "미등록 얼굴입니다."
                return sess
            sess["user_id"] = uid
            sess["baseline"] = db[uid]["smile_baseline"]
            sess["stage"] = "await_smile"
            sess["msg"] = f"{uid}님 확인됨 — 이제 활짝 웃어주세요 😊"
            return sess
        else:  # register
            uid, sim = match(emb, db)
            if uid is not None:
                sess["stage"] = "await_overwrite"
                sess["conflict_id"] = uid
                sess["baseline"] = raw_happy
                sess["result"] = {"status": "confirm_overwrite",
                                  "matched": uid, "similarity": round(sim, 3)}
                sess["msg"] = f"이미 '{uid}'로 등록된 얼굴입니다."
                return sess
            sess["baseline"] = raw_happy
            sess["stage"] = "confirm_register"
            sess["msg"] = "무표정 확인 — 등록을 마무리합니다."
            return sess

    # 2) 웃음 단계(결제): 처리 프레임 N장을 모아 최고 미소로 확정
    if stage == "await_smile":
        s_now = smile_score(raw_happy, sess["baseline"])
        sess["best_smile"] = max(sess["best_smile"], s_now)
        sess["smile_frames"] = sess.get("smile_frames", 0) + 1

        # 진행률 = 처리된 프레임 / 목표 프레임 수 (미소 점수와 분리)
        sess["target_pct"] = min(100, sess["smile_frames"] * 100 / SMILE_FRAMES)

        if sess["smile_frames"] < SMILE_FRAMES:
            sess["msg"] = "collecting"
            return sess

        # N프레임 수집 완료 → 최고점으로 확정
        s_after = sess["best_smile"]
        reward = calc_reward(s_after, sess["amount"])
        sess["stage"] = "done"
        sess["result"] = {"status": "ok", "user_id": sess["user_id"],
                          "smile_score": s_after, "amount": sess["amount"],
                          "reward_rate": reward["reward_rate"],
                          "reward_points": reward["reward_points"]}
        sess["msg"] = "결제 완료!"
        return sess

    # 3) 등록 확정(신규)
    if stage == "confirm_register":
        db[sess["user_id"]] = {"embedding": sess["emb_neutral"],
                               "smile_baseline": sess["baseline"]}
        save_db(db)
        sess["stage"] = "done"
        sess["result"] = {"status": "registered", "user_id": sess["user_id"],
                          "smile_baseline": round(sess["baseline"], 3)}
        sess["msg"] = "등록 완료!"
        return sess

    return sess


# ------------------------------------------------------------------
# 렌더링
# ------------------------------------------------------------------
def render_status(sess):
    if sess and sess.get("msg") == "collecting":
        pct = max(0, min(100, sess.get("display_pct", 0)))
        name = sess.get("user_id") or "고객"
        return f"""<div id='status-bar'>
          <div class='collect-label'>😊 {name}님의 미소를 모으고 있어요!</div>
          <div class='bar-track'><div class='bar-fill' style='width:{pct:.0f}%'></div></div>
          <div class='bar-pct'>{pct:.0f}%</div>
        </div>"""
    msg = sess.get("msg") if sess else "얼굴 등록 또는 결제를 시작하세요"
    return f"<div id='status-bar'>{msg}</div>"


def render_result(sess):
    if not sess or not sess.get("result"):
        return "<div class='result-empty'>결과가 여기에 표시됩니다</div>"
    r = sess["result"]; st = r["status"]
    if st == "ok":
        return f"""<div class="result-card ok">
          <div class="rc-title">결제 완료</div>
          <div class="rc-user">{r['user_id']}님</div>
          <div class="rc-smile">😊 미소 {r['smile_score']:.0f}점</div>
          <div class="rc-reward">{r['reward_points']:,}P 적립</div>
          <div class="rc-sub">{r['amount']:,}원 · 적립률 {r['reward_rate']}%</div>
        </div>"""
    if st == "registered":
        return f"""<div class="result-card ok">
          <div class="rc-title">등록 완료</div>
          <div class="rc-user">{r['user_id']}님</div>
          <div class="rc-sub">기준 미소값 {r['smile_baseline']}</div></div>"""
    if st == "confirm_overwrite":
        return f"""<div class="result-card warn">
          <div class="rc-title">이미 등록된 얼굴</div>
          <div class="rc-user">{r['matched']}님</div>
          <div class="rc-sub">유사도 {r['similarity']} · 새로 등록할까요?</div></div>"""
    if st == "unknown":
        return """<div class="result-card warn">
          <div class="rc-title">미등록 얼굴</div>
          <div class="rc-sub">먼저 얼굴을 등록해주세요</div></div>"""
    return f"<div class='result-card'>{r}</div>"


# ------------------------------------------------------------------
# CSS
# ------------------------------------------------------------------
CSS = """
.gradio-container { background:#f2f4f6 !important;
  font-family:-apple-system,'Segoe UI',sans-serif; }
#stage { max-width:820px; margin:0 auto; }

.device { background:#fff; border-radius:24px; padding:12px;
  box-shadow:0 12px 40px rgba(0,0,0,0.10); border:1px solid #eef0f3; }
#brand { text-align:center; font-weight:700; font-size:16px; color:#0064ff;
  padding:4px 0 8px; letter-spacing:-0.3px; }
#status-bar { text-align:center; font-size:14px; color:#333d4b; font-weight:600;
  background:#f2f4f6; border-radius:14px; padding:9px; margin-top:8px; min-height:18px; }
.btn-row { display:flex; gap:8px; margin-top:8px; }

.device .image-container, .device [data-testid="image"] {
  aspect-ratio:4/5; overflow:hidden; border-radius:16px; }
.device .image-container img, .device .image-container video,
.device [data-testid="image"] img, .device [data-testid="image"] video {
  width:100%; height:100%; object-fit:cover; }

/* A안: 웹캠 시작 버튼을 토스풍으로 다듬기(숨기지 않음) */
.device .icon-with-text {
  background:#0064ff !important; color:#fff !important;
  border-radius:12px !important; padding:8px 16px !important;
  font-weight:600 !important; font-size:14px !important;
  box-shadow:0 4px 12px rgba(0,100,255,0.25) !important; }
.device .icon-with-text .icon { color:#fff !important; }

.side-panel { background:#fff; border-radius:24px; padding:20px;
  box-shadow:0 12px 40px rgba(0,0,0,0.06); border:1px solid #eef0f3; }
.side-title { font-size:15px; font-weight:700; color:#191f28; margin-bottom:8px; }

.result-empty { color:#b0b8c1; font-size:14px; text-align:center;
  padding:26px 0; border:1px dashed #e5e8eb; border-radius:18px; }
.result-card { border-radius:18px; padding:20px; text-align:center; }
.result-card.ok   { background:#eef4ff; }
.result-card.warn { background:#fff3f0; }
.rc-title { font-size:14px; color:#8b95a1; font-weight:600; }
.rc-user  { font-size:22px; font-weight:800; color:#191f28; margin:4px 0; }
.rc-smile { font-size:16px; color:#333d4b; margin:6px 0; }
.rc-reward{ font-size:26px; font-weight:800; color:#0064ff; margin:8px 0 2px; }
.rc-sub   { font-size:13px; color:#8b95a1; }

/* 미소 수집 프로그레스 바 */
.collect-label { font-size:14px; font-weight:700; color:#0064ff; margin-bottom:6px; }
.bar-track { width:100%; height:12px; background:#e5e8eb;
  border-radius:8px; overflow:hidden; }
.bar-fill { height:100%; background:linear-gradient(90deg,#4d94ff,#0064ff);
  border-radius:8px; transition:width 0.06s linear; }
.bar-pct { font-size:13px; font-weight:700; color:#0064ff; margin-top:4px; }
"""

with gr.Blocks(title="Smile Face Pay") as demo:
    sess_state = gr.State(None)
    with gr.Row(elem_id="stage", equal_height=False):
        with gr.Column(scale=1, elem_classes="device"):
            gr.HTML("<div id='brand'>smile face pay</div>")
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                           show_label=False,
                           webcam_options=gr.WebcamOptions(mirror=True))
            status = gr.HTML("<div id='status-bar'>얼굴 등록 또는 결제를 시작하세요</div>")
            with gr.Row(elem_classes="btn-row"):
                btn_reg = gr.Button("얼굴 등록", variant="secondary")
                btn_pay = gr.Button("결제하기", variant="primary")

        with gr.Column(scale=1, elem_classes="side-panel"):
            gr.HTML("<div class='side-title'>설정</div>")
            uid_box = gr.Textbox(label="이름", value="지훈")
            amt_box = gr.Number(label="결제 금액(원)", value=15000)
            gr.HTML("<div class='side-title' style='margin-top:14px'>결과</div>")
            result = gr.HTML(render_result(None))
            with gr.Row(visible=False) as overwrite_row:
                btn_yes = gr.Button("예, 새로 등록", variant="primary")
                btn_no  = gr.Button("아니오", variant="secondary")

    # 화면 갱신 Timer: 0.05초(빠름). 바는 목표값까지 부드럽게 이징,
    # 그 외 상태는 값이 바뀔 때만 갱신(깜빡임 방지).
    ui_timer = gr.Timer(0.05)

    # ---------------- 콜백 ----------------
    def start_register(uid):
        s = new_session("register", user_id=uid)
        s["msg"] = "무표정을 지어주세요"
        return s

    def start_pay(amount):
        s = new_session("pay", amount=int(amount))
        s["msg"] = "무표정으로 화면을 바라봐주세요"
        return s

    # 버튼 클릭 시 즉시 화면 초기화(깜빡임과 무관하게 한 번만 갱신)
    def on_register(uid):
        s = start_register(uid)
        return s, render_status(s), render_result(s), gr.update(visible=False)

    def on_pay(amount):
        s = start_pay(amount)
        return s, render_status(s), render_result(s), gr.update(visible=False)

    btn_reg.click(on_register, [uid_box],
                  [sess_state, status, result, overwrite_row])
    btn_pay.click(on_pay, [amt_box],
                  [sess_state, status, result, overwrite_row])

    # stream: 세션만 갱신 (HTML 출력 없음 → 깜빡임 없음)
    def on_stream(frame, sess):
        return step(frame, sess)

    cam.stream(on_stream, [cam, sess_state], [sess_state],
               stream_every=0.25)

    # Timer: 바 이징 + 값 변경 시에만 status/result/버튼 갱신
    def on_tick(sess):
        if sess is None:
            return gr.skip(), gr.skip(), gr.skip(), sess

        collecting = sess.get("msg") == "collecting"

        # 바 이징: 표시값을 목표값 쪽으로 남은 거리의 30%씩(목표 초과 금지)
        if collecting:
            target = sess.get("target_pct", 0)
            disp = sess.get("display_pct", 0)
            if disp < target:
                disp = min(target, disp + max(1.0, (target - disp) * 0.3))
                sess["display_pct"] = disp

        show = bool(sess.get("stage") == "await_overwrite")
        # 측정 중엔 display_pct(정수)까지 비교에 포함 → 바가 오를 때마다 갱신
        disp_key = round(sess.get("display_pct", 0)) if collecting else None
        cur = (sess.get("msg"), disp_key, sess.get("result"), show)

        if sess.get("last_render") == cur:
            return gr.skip(), gr.skip(), gr.skip(), sess
        sess["last_render"] = cur
        return (render_status(sess), render_result(sess),
                gr.update(visible=show), sess)

    ui_timer.tick(on_tick, [sess_state],
                  [status, result, overwrite_row, sess_state])

    # 덮어쓰기 버튼
    def do_overwrite(sess):
        db = load_db()
        old = sess.get("conflict_id")
        if old in db:
            del db[old]
        db[sess["user_id"]] = {"embedding": sess["emb_neutral"],
                               "smile_baseline": sess["baseline"]}
        save_db(db)
        sess["stage"] = "done"
        sess["result"] = {"status": "registered", "user_id": sess["user_id"],
                          "smile_baseline": round(sess["baseline"], 3)}
        sess["msg"] = "등록 완료!"
        sess["last_render"] = None
        return sess, render_status(sess), render_result(sess), gr.update(visible=False)

    def cancel_overwrite(sess):
        sess["stage"] = "done"
        sess["result"] = None
        sess["msg"] = "등록을 취소했습니다."
        sess["last_render"] = None
        return sess, render_status(sess), render_result(sess), gr.update(visible=False)

    btn_yes.click(do_overwrite, [sess_state],
                  [sess_state, status, result, overwrite_row])
    btn_no.click(cancel_overwrite, [sess_state],
                 [sess_state, status, result, overwrite_row])

if __name__ == "__main__":
    print(f"[backend] {BACKEND}")
    demo.launch(css=CSS)
