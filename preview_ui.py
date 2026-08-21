"""
Smile Face Pay — 상태별 화면 미리보기 (서버·웹캠 불필요)
- app_ui.py의 CSS / render_status / render_result 를 그대로 재사용
- 좌측 캠 자리에는 character2.png(정적 이미지)를 표시
- 드롭다운으로 상태를 바꿔가며 스크린샷을 찍는 용도
실행: python preview_ui.py  →  브라우저에서 상태 선택 후 캡처
"""
import gradio as gr

# ------------------------------------------------------------------
# app_ui.py 에서 렌더 함수 / CSS 를 그대로 가져온다
#   (import 시 config·src 의존성이 걸리면 아래 '자립 버전' 참고)
# ------------------------------------------------------------------
try:
    from app_ui import CSS, render_status, render_result
except Exception:
    raise SystemExit(
        "app_ui.py를 import 하지 못했습니다. preview_ui.py를 app_ui.py와 "
        "같은 폴더에 두고, config.py·src 패키지가 옆에 있는지 확인하세요."
    )

CAM_IMG = "images/character2.png"   # 캠 자리에 넣을 정적 이미지 경로

# ------------------------------------------------------------------
# 상태별 세션 딕셔너리 (render_status / render_result 가 읽는 키만 채움)
# ------------------------------------------------------------------
STATES = {
    "① 기본 화면": {
        "msg": "얼굴 등록 또는 결제를 시작하세요",
        "result": None,
    },
    "② 무표정 대기(등록)": {
        "msg": "무표정을 지어주세요",
        "result": None,
    },
    "③ 등록 완료": {
        "msg": "등록 완료!",
        "result": {"status": "registered", "user_id": "지훈",
                   "smile_baseline": 0.063},
    },
    "④ 신원 확인(결제)": {
        "msg": "무표정으로 화면을 바라봐주세요",
        "result": None,
    },
    "⑤ 신원 확인 완료": {
        "msg": "지훈님 확인됨 — 이제 활짝 웃어주세요 😊",
        "result": None,
    },
    "⑤ 미소 수집 중": {
        "msg": "collecting",
        "user_id": "지훈",
        "display_pct": 60,
    },
    "⑥ 결제 완료": {
        "msg": "결제 완료!",
        "result": {"status": "ok", "user_id": "지훈",
                   "smile_score": 82, "amount": 3000000,
                   "reward_rate": 8.2, "reward_points": 246000},
    },
    "⑦ 미등록 얼굴": {
        "msg": "미등록 얼굴입니다.",
        "result": {"status": "unknown", "similarity": 0.41},
    },
    "⑧ 중복 등록 확인": {
        "msg": "이미 '지훈'로 등록된 얼굴입니다.",
        "result": {"status": "confirm_overwrite", "matched": "지훈",
                   "similarity": 0.88},
    },
}

# 원본 device 컬럼 구조를 그대로 흉내 낸 좌측 단말기 HTML
def device_html(sess):
    return f"""
    <div class="device">
      <div id="brand">smile face pay</div>
      <div class="cam-box">
        <img src="/gradio_api/file={CAM_IMG}" />
        <div class="cam-stop-wrap"><button class="cam-stop">■ 중지</button></div>
      </div>
      {render_status(sess)}
      <div class="btn-row">
        <button class="tbtn sec">얼굴 등록</button>
        <button class="tbtn pri">결제하기</button>
      </div>
    </div>"""


def side_html(sess):
    overwrite = (sess.get("result") or {}).get("status") == "confirm_overwrite"
    ow_btns = ("""
      <div class='btn-row' style='margin-top:10px'>
        <button class='ow yes'>예, 새로 등록</button>
        <button class='ow no'>아니오</button>
      </div>""" if overwrite else "")
    return f"""
    <div class="side-panel">
      <div class="side-title">설정</div>
      <div class="field"><label>이름</label><div class="box">지훈</div></div>
      <div class="field"><label>결제 금액(원)</label><div class="box">15000</div></div>
      <div class="side-title" style="margin-top:14px">결과</div>
      {render_result(sess)}
      {ow_btns}
    </div>"""


def render(state_name):
    sess = STATES[state_name]
    return device_html(sess), side_html(sess)


# 미리보기 전용 보강 CSS (원본 CSS 위에 얹음)
EXTRA_CSS = """
#stage { max-width:860px; margin:0 auto; display:flex; gap:16px; }
.cam-box { aspect-ratio:4/5; overflow:hidden; border-radius:16px; margin-top:4px; }
.cam-box img { width:100%; height:100%; object-fit:cover; display:block; }
.btn-row .tbtn { flex:1; border:0; border-radius:12px; padding:12px 0;
  font-weight:700; font-size:14px; cursor:pointer; }
.btn-row .tbtn.sec { background:#e5e8eb; color:#333d4b; }
.btn-row .tbtn.pri { background:#ff7a00; color:#fff; }
.btn-row .ow.yes { flex:1; border:0; border-radius:12px; padding:10px 0;
  background:#ff7a00; color:#fff; font-weight:700; }
.btn-row .ow.no  { flex:1; border:0; border-radius:12px; padding:10px 0;
  background:#e5e8eb; color:#333d4b; font-weight:700; }
.field { margin:6px 0; }
.field label { font-size:13px; color:#8b95a1; display:block; margin-bottom:4px; }
.field .box { border:1px solid #e5e8eb; border-radius:12px; padding:11px 14px;
  font-size:15px; color:#191f28; }
.cam-box { position:relative; }
.cam-stop-wrap {
  position:absolute; left:50%; bottom:6px; transform:translateX(-50%);
  background:#fff; border-radius:10px; padding:6px 15px;
  box-shadow:0 4px 14px rgba(0,0,0,0.15); }
.cam-stop {
  background:#0064ff; color:#fff; border:0; border-radius:9px;
  padding:5px 13px; font-weight:700; font-size:14px; cursor:pointer;
  display:flex; align-items:center; gap:6px; }
"""

with gr.Blocks(title="Smile Face Pay — 미리보기", css=CSS + EXTRA_CSS) as demo:
    picker = gr.Dropdown(list(STATES.keys()), value="① 기본 화면",
                         label="상태 선택")
    with gr.Row(elem_id="stage"):
        left = gr.HTML()
        right = gr.HTML()

    picker.change(render, [picker], [left, right])
    demo.load(lambda: render("① 기본 화면"), None, [left, right])

if __name__ == "__main__":
    demo.launch(allowed_paths=[CAM_IMG])
