# =========================================================================
# crack_3.py : 도로 균열 감지 웹 애플리케이션 (수정 버전 3)
# -------------------------------------------------------------------------
# [crack.py → crack_1.py 주요 변경 사항]  (★ 표시가 붙은 주석을 찾으면 됨)
#  ★1. ROI(관심 영역) 다각형 마스크 추가
#       - 보고서 초록과 Ⅲ-2-B에 적힌 기능이 기존 코드에는 없어서 추가함
#       - 사다리꼴 모양으로 도로 노면 부분만 남기고 나머지는 검사하지 않음
#  ★2. 형상 필터링(길쭉한 모양만 균열로 인정) 추가
#       - 기존 코드는 '면적'만 검사했음 → '모양' 조건을 추가함
#  ★3. 두 기능 모두 사이드바에서 켜고 끌 수 있음
#       - 둘 다 끄면 기존 crack.py와 똑같이 동작 → 개선 전/후 비교 실험 가능
#  ★4. 분석 시간 측정 표시 (보고서 Ⅳ-2-A '1장당 약 ○초' 근거)
#  ★5. 처리 단계별 이미지 보기 (원본→흑백→블러→이진화, 포스터용 캡처)
#  ★6. 보관함 기록에 '분석 당시 설정값'도 함께 저장
#       - 기존에는 기록을 다시 볼 때 현재 슬라이더 값이 표시되는 문제가 있었음
#  ★7. 슬라이더에 key를 연결해 초기화 버튼이 확실하게 동작하도록 수정
#  ★8. 이미지 읽기 안정성 개선 (read → getvalue, 잘못된 파일 오류 안내)
#  ★9. [crack_2 추가] ROI 아랫변 위치 슬라이더
#       - 차량에서 찍은 사진은 맨 아래에 자동차 보닛(앞부분)이 찍혀 균열로
#         잘못 잡히는 문제가 있어, ROI 아랫변을 위로 올려 보닛을 제외할 수 있게 함
#       - 기본값 100%는 crack_1과 똑같이 사진 맨 아래까지 검사함
#  ★10. [crack_3 추가] 도로 표시(차선·횡단보도) 제외
#       - 정상 도로에서도 차선 가장자리가 길쭉한 선으로 잡히는 문제 해결용
#       - 흰색·노란색 페인트 영역을 색(HSV)으로 찾아, 그 주변은 검사하지 않음
#  ★11. [crack_3 추가] 명암 차이 기준(C) 슬라이더
#       - 기존에는 2로 고정 → 아스팔트의 자잘한 질감까지 균열 후보로 잡혔음
#       - 값을 올리면 '주변보다 확실히 어두운 곳'만 남음 (너무 올리면 흐린 균열을 놓침)
#  ※ ★9~★11을 기본값(아랫변 100%, 도로 표시 제외 끔, C=2)으로 두면 crack_1과 똑같이 동작함
#  프로그램의 큰 틀(입력 → 처리 → 출력/보관함)과 화면 구성은 그대로 유지함
# =========================================================================

import time  # ★4 분석 시간 측정용

import cv2
import numpy as np
import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="도로 균열 감지 웹 애플리케이션", page_icon="🛣️", layout="wide"
)

# -------------------------------------------------------------------------
# 기본 설정값 (초기화 버튼을 누르면 이 값으로 돌아감)
# -------------------------------------------------------------------------
DEFAULTS = {
    "min_area": 100,         # 최소 균열 크기(픽셀 면적)
    "block_size": 11,        # 명암 비교 범위(적응형 이진화 구역 크기, 홀수)
    "c_value": 2,            # ★11 명암 차이 기준(기존 코드의 고정값 2)
    "use_roi": True,         # ★1 ROI 마스크 사용 여부
    "roi_top": 40,           # ★1 ROI 윗변 위치(사진 위에서부터 %)
    "roi_top_width": 60,     # ★1 ROI 윗변 폭(사진 가로 길이 대비 %)
    "roi_bottom": 100,       # ★9 ROI 아랫변 위치(사진 위에서부터 %)
    "use_shape": True,       # ★2 형상 필터 사용 여부
    "min_ratio": 3.0,        # ★2 길쭉함 기준(긴 변 ÷ 짧은 변)
    "use_paint_mask": False, # ★10 도로 표시(차선·횡단보도) 제외 여부
}

# ★2 '구불구불하거나 가지가 갈라진 균열'을 위한 보조 기준
#    외곽선이 감싼 면적 ÷ 외곽선을 감싸는 최소 사각형 면적이 이 값 이하이면
#    (=속이 텅 빈 가는 선 형태) 비율이 낮아도 균열로 인정함
MAX_FILL_RATIO = 0.25

# 1. 세션 스테이트 초기화 (기록 보관함 및 모드 관리용)
if "history" not in st.session_state:
  st.session_state.history = []
if "mode" not in st.session_state:
  st.session_state.mode = "realtime"  # 'realtime' 또는 'history'
if "active_history_title" not in st.session_state:
  st.session_state.active_history_title = None
for k, v in DEFAULTS.items():  # 설정값 초기화
  if k not in st.session_state:
    st.session_state[k] = v


# -------------------------------------------------------------------------
# [영상 처리 함수] 화면 코드와 분리해 두면 테스트·수정이 쉬워짐
# -------------------------------------------------------------------------
def make_roi_mask(h, w, top_ratio, top_width_ratio, bottom_ratio=1.0):
  """★1 도로 노면 모양(사다리꼴)의 ROI 다각형 마스크를 만든다.

  - 아랫변: 사진 위에서 bottom_ratio 위치, 가로 전체 (★9 기본은 맨 아래)
  - 윗변: 사진 위에서 top_ratio 위치, 가운데를 기준으로 top_width_ratio 폭
  차량이나 사람이 서서 찍은 도로 사진은 아래쪽이 노면, 위쪽이 하늘·건물·풀숲인
  경우가 많으므로 위쪽 영역을 검사 대상에서 빼는 원리이다.
  """
  top_y = int(h * top_ratio)
  bottom_y = min(int(h * bottom_ratio), h) - 1  # ★9 아랫변 위치
  half_top = int(w * top_width_ratio / 2)
  cx = w // 2
  pts = np.array(
      [
          [0, bottom_y],              # 왼쪽 아래
          [w - 1, bottom_y],          # 오른쪽 아래
          [cx + half_top, top_y],     # 오른쪽 위
          [cx - half_top, top_y],     # 왼쪽 위
      ],
      dtype=np.int32,
  )
  mask = np.zeros((h, w), dtype=np.uint8)
  cv2.fillPoly(mask, [pts], 255)  # 다각형 안쪽만 흰색(255) = 검사할 영역
  return mask, pts


def make_paint_mask(org_img, grow):
  """★10 흰색·노란색 도로 표시(차선, 횡단보도, 정지선 등) 영역을 찾는다.

  - HSV 색 공간: 색(H)·선명도(S)·밝기(V)로 나누어 색을 고르기 쉬움
  - 흰색 = 아주 밝고(V 높음) 색이 거의 없는(S 낮음) 곳
  - 노란색 = 색(H)이 노랑 범위이고 어느 정도 선명하고 밝은 곳
  - 적응형 이진화는 밝은 차선 바로 옆 아스팔트를 '상대적으로 어둡다'고 판단해
    차선 테두리를 따라 가짜 선을 만든다. 그래서 찾은 영역을 grow 픽셀만큼
    부풀려(dilate) 테두리까지 함께 제외한다.
  """
  hsv = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)
  white = cv2.inRange(hsv, (0, 0, 170), (180, 50, 255))
  yellow = cv2.inRange(hsv, (15, 70, 100), (40, 255, 255))
  paint = cv2.bitwise_or(white, yellow)
  return cv2.dilate(paint, np.ones((grow, grow), np.uint8), iterations=1)


def is_crack_shape(cnt, min_ratio):
  """★2 외곽선(컨투어)이 '균열다운 모양'인지 판단한다.

  조건 ① 길쭉함: 회전까지 고려한 최소 사각형의 (긴 변 ÷ 짧은 변) ≥ min_ratio
         → 대각선 방향 균열도 길쭉하게 인식됨
  조건 ② 가는 선: (외곽선 면적 ÷ 최소 사각형 면적) ≤ MAX_FILL_RATIO
         → 가지가 갈라지거나 구불구불한 균열처럼 비율이 낮아도 속이 빈 선 형태
  둘 중 하나라도 맞으면 균열로 인정한다.
  둥글고 꽉 찬 덩어리(얼룩, 자갈, 사람 몸통 등)는 두 조건 모두 탈락한다.
  """
  (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
  long_side = max(rw, rh)
  short_side = max(min(rw, rh), 1.0)  # 0으로 나누기 방지
  ratio = long_side / short_side
  fill_ratio = cv2.contourArea(cnt) / max(rw * rh, 1.0)
  return ratio >= min_ratio or fill_ratio <= MAX_FILL_RATIO


def detect_cracks(org_img, p):
  """원본 이미지(BGR)와 설정값 p를 받아 균열 감지 결과를 돌려준다."""
  # (1) 흑백 변환 → (2) 가우시안 블러 → (3) 적응형 이진화  [기존과 동일]
  gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  bs = p["block_size"] if p["block_size"] % 2 == 1 else p["block_size"] + 1
  thresh = cv2.adaptiveThreshold(
      blurred,
      255,
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv2.THRESH_BINARY_INV,
      bs,
      p["c_value"],  # ★11 기존에는 2로 고정되어 있던 값
  )

  # (4) ★1 ROI 마스크 적용: 다각형 바깥의 흰 점은 모두 지움
  roi_pts = None
  if p["use_roi"]:
    h, w = thresh.shape
    mask, roi_pts = make_roi_mask(
        h, w, p["roi_top"] / 100, p["roi_top_width"] / 100, p["roi_bottom"] / 100
    )
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

  # (4-2) ★10 도로 표시 제외: 차선·횡단보도와 그 테두리를 검사 대상에서 뺌
  if p["use_paint_mask"]:
    thresh[make_paint_mask(org_img, bs) > 0] = 0

  # (5) 컨투어(외곽선) 찾기  [기존과 동일]
  contours, _ = cv2.findContours(
      thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  # (6) 면적 필터: 너무 작은 점 노이즈 제거  [기존과 동일]
  area_ok = [c for c in contours if cv2.contourArea(c) >= p["min_area"]]

  # (7) ★2 형상 필터: 길쭉하거나 가는 선 형태만 균열로 인정
  if p["use_shape"]:
    crack_contours = [c for c in area_ok if is_crack_shape(c, p["min_ratio"])]
  else:
    crack_contours = area_ok

  # (8) 결과 그리기: 균열은 초록색, ROI 경계는 노란색 얇은 선
  result_img = org_img.copy()
  if roi_pts is not None:
    cv2.polylines(result_img, [roi_pts], True, (0, 255, 255), 1)
  cv2.drawContours(result_img, crack_contours, -1, (0, 255, 0), 2)

  return {
      "gray": gray,
      "blurred": blurred,
      "thresh": thresh,
      "result": result_img,
      "count": len(crack_contours),
      "removed_by_shape": len(area_ok) - len(crack_contours),
  }


def params_text(p):
  """현재 설정값을 한 줄 문자열로 만든다 (화면 표시·기록 저장용)."""
  roi = (
      f"ROI 윗변 {p['roi_top']}%·폭 {p['roi_top_width']}%·아랫변 {p['roi_bottom']}%"
      if p["use_roi"] else "ROI 끔"
  )
  shape = f"길쭉함 ≥ {p['min_ratio']}" if p["use_shape"] else "형상 필터 끔"
  paint = "도로 표시 제외" if p["use_paint_mask"] else "도로 표시 포함"
  return (
      f"최소 면적 {p['min_area']}px | 명암 비교 범위 {p['block_size']} | "
      f"명암 차이 기준 {p['c_value']} | {roi} | {shape} | {paint}"
  )


# 타이틀 및 소개
st.title("🛣️ 도로 균열 감지 웹 애플리케이션")
st.write(
    "도로 이미지(PNG, JPG)를 업로드하거나 카메라로 촬영하여 관심 영역(ROI)"
    " 기반 OpenCV 처리로 균열을 감지합니다."
)
st.divider()

# -------------------------------------------------------------------------
# [사이드바] 탐지 상세 설정 및 이전 기록 보관함
# -------------------------------------------------------------------------
st.sidebar.header("⚙️ 탐지 상세 설정")


# 초기화 버튼
def reset_params():
  for k, v in DEFAULTS.items():
    st.session_state[k] = v


st.sidebar.button("🔄 초기화", on_click=reset_params)

# ★7 key를 연결하면 슬라이더 값이 st.session_state에 자동 저장되고,
#    초기화 버튼(reset_params)이 바꾼 값도 슬라이더에 바로 반영됨
st.sidebar.slider("최소 균열 크기", min_value=10, max_value=500, step=10, key="min_area")
st.sidebar.slider("명암 비교 범위", min_value=3, max_value=51, step=2, key="block_size")
# ★11 값이 클수록 '주변보다 확실히 어두운 곳'만 균열 후보로 남김
st.sidebar.slider("명암 차이 기준 (C)", min_value=0, max_value=20, step=1, key="c_value")

# ★1 ROI 설정
st.sidebar.markdown("**🧭 관심 영역(ROI)**")
st.sidebar.checkbox("ROI 마스크 사용 (도로 노면만 검사)", key="use_roi")
st.sidebar.slider(
    "ROI 윗변 위치 (사진 위에서부터 %)",
    min_value=0, max_value=90, step=5, key="roi_top",
    disabled=not st.session_state.use_roi,
)
st.sidebar.slider(
    "ROI 윗변 폭 (%)",
    min_value=10, max_value=100, step=5, key="roi_top_width",
    disabled=not st.session_state.use_roi,
)
# ★9 ROI 아랫변 위치 (차량 보닛 등 사진 아래쪽 제외용)
st.sidebar.slider(
    "ROI 아랫변 위치 (사진 위에서부터 %)",
    min_value=50, max_value=100, step=2, key="roi_bottom",
    disabled=not st.session_state.use_roi,
)
if st.session_state.use_roi and st.session_state.roi_bottom <= st.session_state.roi_top + 5:
  # 아랫변이 윗변보다 위로 올라가면 검사 영역이 사라지므로 안내함
  st.sidebar.warning("⚠️ ROI 아랫변이 윗변보다 아래에 있어야 합니다. 값을 다시 맞춰 주세요.")

# ★2 형상 필터 설정
st.sidebar.markdown("**📏 형상 필터**")
st.sidebar.checkbox("길쭉한 선 형태만 균열로 인정", key="use_shape")
st.sidebar.slider(
    "길쭉함 기준 (긴 변 ÷ 짧은 변)",
    min_value=1.0, max_value=10.0, step=0.5, key="min_ratio",
    disabled=not st.session_state.use_shape,
)

# ★10 도로 표시 제외 설정
st.sidebar.markdown("**🎨 도로 표시 제외**")
st.sidebar.checkbox("차선·횡단보도(흰색·노란색) 영역 제외", key="use_paint_mask")

# 현재 설정값 묶음
params = {k: st.session_state[k] for k in DEFAULTS}

# [사이드바] 이전 기록 보관함 (텍스트 버튼 및 삭제 기능)
st.sidebar.markdown("---")
st.sidebar.subheader("📂 이전 분석 기록 보관함")

if len(st.session_state.history) > 0:
  st.sidebar.caption("👉 기록 이름을 누르면 해당 분석 결과로 이동합니다.")

  for item in list(st.session_state.history):
    col_item1, col_item2 = st.sidebar.columns([3, 1])

    with col_item1:
      is_current = (
          st.session_state.mode == "history"
          and st.session_state.active_history_title == item["title"]
      )
      btn_label = f"📌 {item['title']}" if is_current else item["title"]

      if st.button(btn_label, key=f"view_{item['title']}", use_container_width=True):
        st.session_state.mode = "history"
        st.session_state.active_history_title = item["title"]
        st.rerun()

    with col_item2:
      if st.button("삭제", key=f"del_{item['title']}", use_container_width=True):
        st.session_state.history = [
            h for h in st.session_state.history if h["title"] != item["title"]
        ]
        if st.session_state.active_history_title == item["title"]:
          st.session_state.mode = "realtime"
          st.session_state.active_history_title = None
        st.rerun()
else:
  st.sidebar.info("저장된 기록이 없습니다.")

# -------------------------------------------------------------------------
# 메인 영역: 입력 방식 선택
# -------------------------------------------------------------------------
st.subheader("📷 이미지 입력 방식 선택")
input_mode = st.radio(
    "입력 방식을 선택하세요",
    ["📁 파일 업로드", "📷 카메라 촬영"],
    horizontal=True,
)


def on_upload_file():
  st.session_state.mode = "realtime"


uploaded_file = None

if input_mode == "📁 파일 업로드":
  uploaded_file = st.file_uploader(
      "검사할 도로 이미지를 업로드하세요",
      type=["png", "jpg", "jpeg"],
      on_change=on_upload_file,
  )
else:
  uploaded_file = st.camera_input(
      "도로 현장 사진을 직접 촬영하세요", on_change=on_upload_file
  )

# -------------------------------------------------------------------------
# 이미지 표시 변수 설정 (보관함 모드 vs 실시간 분석 모드)
# -------------------------------------------------------------------------
display_org_rgb = None
display_res_rgb = None
is_from_history = False
crack_contours_count = 0
current_result_bgr = None   # 다운로드용 BGR 이미지 저장 변수
shown_params_text = params_text(params)  # ★6 화면에 보여 줄 설정값
step_images = None          # ★5 처리 단계별 이미지
elapsed = None              # ★4 분석 시간(초)
removed_by_shape = 0

# 1) 보관함 조회 모드인 경우
if (
    st.session_state.mode == "history"
    and st.session_state.active_history_title
    and len(st.session_state.history) > 0
):
  selected_item = next(
      (
          item
          for item in st.session_state.history
          if item["title"] == st.session_state.active_history_title
      ),
      None,
  )
  if selected_item:
    display_org_rgb = selected_item["original"]
    display_res_rgb = selected_item["result"]
    current_result_bgr = selected_item["result_bgr"]
    # ★6 기록 당시의 설정값을 보여 줌 (예전 기록에는 없을 수 있어 get 사용)
    shown_params_text = selected_item.get("params", "기록된 설정값 없음")
    is_from_history = True

# 2) 실시간 분석 모드이면서 새 파일이 업로드된 경우
elif uploaded_file is not None:
  # ★8 getvalue(): 화면이 다시 그려져도 파일 내용을 안정적으로 다시 읽음
  file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
  org_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  if org_img is None:
    st.error("❌ 이미지를 읽을 수 없습니다. PNG 또는 JPG 파일인지 확인해 주세요.")
  else:
    t0 = time.perf_counter()  # ★4 분석 시작 시각
    out = detect_cracks(org_img, params)
    elapsed = time.perf_counter() - t0  # ★4 걸린 시간(초)

    display_org_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    display_res_rgb = cv2.cvtColor(out["result"], cv2.COLOR_BGR2RGB)
    current_result_bgr = out["result"]
    crack_contours_count = out["count"]
    removed_by_shape = out["removed_by_shape"]
    step_images = out  # ★5

# -------------------------------------------------------------------------
# 결과 및 다운로드 버튼 출력
# -------------------------------------------------------------------------
if display_org_rgb is not None and display_res_rgb is not None:
  st.divider()

  if is_from_history:
    st.info(
        f"📌 [보관함 기록 조회 중] 선택된 항목: `{st.session_state.active_history_title}`"
    )

  col1, col2 = st.columns(2)
  with col1:
    st.subheader("원본 이미지")
    st.image(display_org_rgb, use_container_width=True)

  with col2:
    st.subheader("균열 분석 결과")
    st.image(display_res_rgb, use_container_width=True)

  # ★5 처리 단계별 이미지 (보고서·포스터의 '원본→흑백→블러→이진화' 그림용)
  if step_images is not None:
    with st.expander("🔍 처리 단계별 이미지 보기 (흑백 → 블러 → 이진화)"):
      s1, s2, s3 = st.columns(3)
      with s1:
        st.image(step_images["gray"], caption="① 흑백 변환", use_container_width=True)
      with s2:
        st.image(step_images["blurred"], caption="② 가우시안 블러", use_container_width=True)
      with s3:
        st.image(
            step_images["thresh"],
            caption="③ 적응형 이진화 (+ROI 적용)" if params["use_roi"] else "③ 적응형 이진화",
            use_container_width=True,
        )

  st.divider()

  # 📥 이미지 다운로드 버튼 (실시간 분석 및 보관함 조회 모두에서 상시 제공)
  d_col1, d_col2 = st.columns([1, 1])
  with d_col1:
    if current_result_bgr is not None:
      is_success, buffer = cv2.imencode(".png", current_result_bgr)
      if is_success:
        st.download_button(
            label="📥 균열 감지 결과 이미지 다운로드",
            data=buffer.tobytes(),
            file_name="road_crack_result.png",
            mime="image/png",
            use_container_width=True,
        )

  with d_col2:
    st.caption(f"ℹ️ {shown_params_text}")  # ★6
    if elapsed is not None:
      st.caption(f"⏱️ 분석 시간: {elapsed:.2f}초")  # ★4

  # 실시간 분석된 이미지일 때만 결과 요약 및 보관함 저장 폼 활성화
  if not is_from_history:
    if crack_contours_count == 0:
      st.success("✅ [검사 결과] 균열이 발견되지 않았습니다. (정상 도로)")
    else:
      st.error(
          f"⚠️ [검사 결과] 균열이 감지되었습니다! (총 {crack_contours_count}개 영역)"
      )
    if params["use_shape"] and removed_by_shape > 0:
      # ★2 형상 필터가 걸러낸 개수를 보여 주면 필터 효과를 바로 확인 가능
      st.caption(f"📏 형상 필터가 균열이 아닌 모양 {removed_by_shape}개를 제외했습니다.")

    # 보관함 저장 폼 (중복 검사 포함)
    st.markdown("---")
    st.subheader("💾 분석 기록 보관함에 저장하기")
    record_title = st.text_input(
        "📝 이 분석 기록의 제목을 입력하세요",
        value=f"도로 현장 점검 #{len(st.session_state.history) + 1}",
    )

    if st.button("보관함에 저장하기"):
      existing_titles = [item["title"] for item in st.session_state.history]

      if record_title in existing_titles:
        st.error(
            f"❌ 이미 '{record_title}'(으)로 저장된 기록이 존재합니다. 다른"
            " 제목을 입력해주세요."
        )
      else:
        new_record = {
            "title": record_title,
            "original": display_org_rgb,
            "result": display_res_rgb,
            "result_bgr": current_result_bgr,
            "params": shown_params_text,  # ★6 분석 당시 설정값 함께 저장
        }
        st.session_state.history.append(new_record)
        st.success(f"'{record_title}' 기록이 사이드바 보관함에 저장되었습니다!")
        st.rerun()
