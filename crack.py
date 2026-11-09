import time

import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="도로 균열 감지 웹 애플리케이션", page_icon="🛣️", layout="wide"
)

DEFAULTS = {
    "min_area": 100,
    "block_size": 11,
    "c_value": 2,
    "use_roi": True,
    "roi_top": 40,
    "roi_top_width": 60,
    "roi_bottom": 100,
    "use_shape": True,
    "min_ratio": 3.0,
    "use_paint_mask": False,
}

MAX_FILL_RATIO = 0.25

if "history" not in st.session_state:
  st.session_state.history = []
if "mode" not in st.session_state:
  st.session_state.mode = "realtime"
if "active_history_title" not in st.session_state:
  st.session_state.active_history_title = None
for k, v in DEFAULTS.items():
  if k not in st.session_state:
    st.session_state[k] = v


def make_roi_mask(h, w, top_ratio, top_width_ratio, bottom_ratio=1.0):
  top_y = int(h * top_ratio)
  bottom_y = min(int(h * bottom_ratio), h) - 1
  half_top = int(w * top_width_ratio / 2)
  cx = w // 2
  pts = np.array(
      [
          [0, bottom_y],
          [w - 1, bottom_y],
          [cx + half_top, top_y],
          [cx - half_top, top_y],
      ],
      dtype=np.int32,
  )
  mask = np.zeros((h, w), dtype=np.uint8)
  cv2.fillPoly(mask, [pts], 255)
  return mask, pts


def make_paint_mask(org_img, grow):
  hsv = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)
  white = cv2.inRange(hsv, (0, 0, 170), (180, 50, 255))
  yellow = cv2.inRange(hsv, (15, 70, 100), (40, 255, 255))
  paint = cv2.bitwise_or(white, yellow)
  return cv2.dilate(paint, np.ones((grow, grow), np.uint8), iterations=1)


def is_crack_shape(cnt, min_ratio):
  (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
  long_side = max(rw, rh)
  short_side = max(min(rw, rh), 1.0)
  ratio = long_side / short_side
  fill_ratio = cv2.contourArea(cnt) / max(rw * rh, 1.0)
  return ratio >= min_ratio or fill_ratio <= MAX_FILL_RATIO


def detect_cracks(org_img, p):
  gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  bs = p["block_size"] if p["block_size"] % 2 == 1 else p["block_size"] + 1
  thresh = cv2.adaptiveThreshold(
      blurred,
      255,
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv2.THRESH_BINARY_INV,
      bs,
      p["c_value"],
  )

  roi_pts = None
  if p["use_roi"]:
    h, w = thresh.shape
    mask, roi_pts = make_roi_mask(
        h, w, p["roi_top"] / 100, p["roi_top_width"] / 100, p["roi_bottom"] / 100
    )
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

  if p["use_paint_mask"]:
    thresh[make_paint_mask(org_img, bs) > 0] = 0

  contours, _ = cv2.findContours(
      thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  area_ok = [c for c in contours if cv2.contourArea(c) >= p["min_area"]]

  if p["use_shape"]:
    crack_contours = [c for c in area_ok if is_crack_shape(c, p["min_ratio"])]
  else:
    crack_contours = area_ok

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


st.title("🛣️ 도로 균열 감지 웹 애플리케이션")
st.write(
    "도로 이미지(PNG, JPG)를 업로드하거나 카메라로 촬영하여 관심 영역(ROI)"
    " 기반 OpenCV 처리로 균열을 감지합니다."
)
st.divider()

st.sidebar.header("⚙️ 탐지 상세 설정")


def reset_params():
  for k, v in DEFAULTS.items():
    st.session_state[k] = v


st.sidebar.button("🔄 초기화", on_click=reset_params)

st.sidebar.slider("최소 균열 크기", min_value=10, max_value=500, step=10, key="min_area")
st.sidebar.slider("명암 비교 범위", min_value=3, max_value=51, step=2, key="block_size")
st.sidebar.slider("명암 차이 기준 (C)", min_value=0, max_value=20, step=1, key="c_value")

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
st.sidebar.slider(
    "ROI 아랫변 위치 (사진 위에서부터 %)",
    min_value=50, max_value=100, step=2, key="roi_bottom",
    disabled=not st.session_state.use_roi,
)
if st.session_state.use_roi and st.session_state.roi_bottom <= st.session_state.roi_top + 5:
  st.sidebar.warning("⚠️ ROI 아랫변이 윗변보다 아래에 있어야 합니다. 값을 다시 맞춰 주세요.")

st.sidebar.markdown("**📏 형상 필터**")
st.sidebar.checkbox("길쭉한 선 형태만 균열로 인정", key="use_shape")
st.sidebar.slider(
    "길쭉함 기준 (긴 변 ÷ 짧은 변)",
    min_value=1.0, max_value=10.0, step=0.5, key="min_ratio",
    disabled=not st.session_state.use_shape,
)

st.sidebar.markdown("**🎨 도로 표시 제외**")
st.sidebar.checkbox("차선·횡단보도(흰색·노란색) 영역 제외", key="use_paint_mask")

params = {k: st.session_state[k] for k in DEFAULTS}

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

display_org_rgb = None
display_res_rgb = None
is_from_history = False
crack_contours_count = 0
current_result_bgr = None
shown_params_text = params_text(params)
step_images = None
elapsed = None
removed_by_shape = 0

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
    shown_params_text = selected_item.get("params", "기록된 설정값 없음")
    is_from_history = True

elif uploaded_file is not None:
  file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
  org_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  if org_img is None:
    st.error("❌ 이미지를 읽을 수 없습니다. PNG 또는 JPG 파일인지 확인해 주세요.")
  else:
    t0 = time.perf_counter()
    out = detect_cracks(org_img, params)
    elapsed = time.perf_counter() - t0

    display_org_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    display_res_rgb = cv2.cvtColor(out["result"], cv2.COLOR_BGR2RGB)
    current_result_bgr = out["result"]
    crack_contours_count = out["count"]
    removed_by_shape = out["removed_by_shape"]
    step_images = out

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
    st.caption(f"ℹ️ {shown_params_text}")
    if elapsed is not None:
      st.caption(f"⏱️ 분석 시간: {elapsed:.2f}초")

  if not is_from_history:
    if crack_contours_count == 0:
      st.success("✅ [검사 결과] 균열이 발견되지 않았습니다. (정상 도로)")
    else:
      st.error(
          f"⚠️ [검사 결과] 균열이 감지되었습니다! (총 {crack_contours_count}개 영역)"
      )
    if params["use_shape"] and removed_by_shape > 0:
      st.caption(f"📏 형상 필터가 균열이 아닌 모양 {removed_by_shape}개를 제외했습니다.")

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
            "params": shown_params_text,
        }
        st.session_state.history.append(new_record)
        st.success(f"'{record_title}' 기록이 사이드바 보관함에 저장되었습니다!")
        st.rerun()