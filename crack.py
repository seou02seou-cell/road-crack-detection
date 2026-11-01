import cv2
import numpy as np
import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="도로 균열 감지 웹 애플리케이션", page_icon="🛣️", layout="wide"
)

# 1. 세션 스테이트 초기화 (기록 보관함 및 모드 관리용)
if "history" not in st.session_state:
  st.session_state.history = []
if "mode" not in st.session_state:
  st.session_state.mode = "realtime"  # 'realtime' 또는 'history'
if "active_history_title" not in st.session_state:
  st.session_state.active_history_title = None

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
  st.session_state.min_area = 100
  st.session_state.block_size = 11


if "min_area" not in st.session_state:
  st.session_state.min_area = 100
if "block_size" not in st.session_state:
  st.session_state.block_size = 11

if st.sidebar.button("🔄 초기화", on_click=reset_params):
  pass

min_area = st.sidebar.slider(
    "최소 균열 크기",
    min_value=10,
    max_value=500,
    value=st.session_state.min_area,
    step=10,
)
block_size = st.sidebar.slider(
    "명암 비교 범위",
    min_value=3,
    max_value=51,
    value=st.session_state.block_size,
    step=2,
)

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
# 메인 영역: 실시간 입력 방식 선택
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
current_result_bgr = None  # 다운로드용 BGR 이미지 저장 변수

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
    current_result_bgr = selected_item["result_bgr"]  # 저장해둔 BGR 원본
    is_from_history = True

# 2) 실시간 분석 모드이면서 새 파일이 업로드된 경우
elif uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  org_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  if org_img is not None:
    gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size if block_size % 2 == 1 else block_size + 1,
        2,
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result_img = org_img.copy()
    crack_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) >= min_area
    ]
    cv2.drawContours(result_img, crack_contours, -1, (0, 255, 0), 2)

    display_org_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    display_res_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    current_result_bgr = result_img  # 다운로드용으로 보관
    crack_contours_count = len(crack_contours)

# -------------------------------------------------------------------------
# 원래 분석 창 위치에 결과 및 다운로드 버튼 출력
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

  st.divider()

  # 📥 이미지 다운로드 버튼 (실시간 분석 및 보관함 조회 모두에서 상시 제공)
  d_col1, d_col2 = st.columns([1, 1])
  with d_col1:
    if current_result_bgr is not None:
      is_success, buffer = cv2.imencode(".png", current_result_bgr)
      if is_success:
        byte_im = buffer.tobytes()
        st.download_button(
            label="📥 균열 감지 결과 이미지 다운로드",
            data=byte_im,
            file_name="road_crack_result.png",
            mime="image/png",
            use_container_width=True,
        )

  with d_col2:
    st.caption(
        f"ℹ️ 최소 감지 기준 면적: `{min_area}` px | 명암 비교 범위:`{block_size}`"
    )

  # 실시간 분석된 이미지일 때만 결과 요약 및 보관함 저장 폼 활성화
  if not is_from_history:
    if crack_contours_count == 0:
      st.success("✅ [검사 결과] 균열이 발견되지 않았습니다. (정상 도로)")
    else:
      st.error(
          f"⚠️ [검사 결과] 균열이 감지되었습니다! (총 {crack_contours_count}개 영역)"
      )

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
            "result_bgr": current_result_bgr,  # 다운로드용 데이터도 함께 보관
        }
        st.session_state.history.append(new_record)
        st.success(f"'{record_title}' 기록이 사이드바 보관함에 저장되었습니다!")
        st.rerun()