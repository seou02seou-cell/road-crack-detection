import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="도로 균열 감지 시스템", page_icon="🛣️", layout="wide")

st.title("🛣️ 도로 균열 감지 웹 애플리케이션")
st.write("도로 이미지(PNG, JPG)를 업로드하면 관심 영역(ROI) 기반 OpenCV 처리로 균열을 감지합니다.")
st.divider()

# 세션 초기화
if "min_area" not in st.session_state:
    st.session_state.min_area = 100
if "block_size" not in st.session_state:
    st.session_state.block_size = 11

sb_col1, sb_col2 = st.sidebar.columns([1, 0.8])
with sb_col1:
    st.header("⚙️ 탐지 상세 설정")
with sb_col2:
    if st.button("🔄 초기화", use_container_width=True):
        st.session_state.min_area = 100
        st.session_state.block_size = 11
        st.rerun()

min_area = st.sidebar.slider("최소 균열 크기", 10, 1000, key="min_area", step=10)
block_size = st.sidebar.slider("명암 비교 범위", 3, 31, key="block_size", step=2)

file_input = st.file_uploader("검사할 도로 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
if file_input is not None:
    st.session_state.uploaded_file = file_input

uploaded_file = st.session_state.get("uploaded_file", None)

if uploaded_file is not None:
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        h, w, _ = image.shape

        # 1. 전처리 (그레이스케일 및 블러링)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. 적응형 이진화
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 2
        )

        # 3. ROI Mask 생성 (중앙 도로 노면 영역만 추출하여 외곽 풀숲/차선 제외)
        mask = np.zeros_like(thresh)
        # 도로 중앙 영역 좌표 지정 (상황에 맞게 비율 조정 가능)
        polygon = np.array([[
            (int(w * 0.1), h),
            (int(w * 0.3), int(h * 0.4)),
            (int(w * 0.7), int(h * 0.4)),
            (int(w * 0.9), h)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        
        # 마스크 적용
        roi_thresh = cv2.bitwise_and(thresh, mask)

        # 4. 윤곽선 검출 및 필터링
        contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crack_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(max(bw, bh)) / min(bw, bh) if min(bw, bh) > 0 else 0
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # 선 형태(길쭉함) 위주 필터링
                if aspect_ratio >= 1.4 and solidity <= 0.85:
                    crack_contours.append(cnt)

        # 결과 시각화
        result_img = image.copy()
        cv2.drawContours(result_img, crack_contours, -1, (0, 255, 0), 2)

        orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ 원본 이미지")
            st.image(orig_rgb, use_container_width=True)
        with col2:
            st.subheader("🔍 균열 감지 결과")
            st.image(res_rgb, use_container_width=True)

        st.divider()
        if len(crack_contours) == 0:
            st.success("✅ [검사 결과] 균열이 발견되지 않았습니다. (정상 도로)")
        else:
            st.error(f"⚠️ [검사 결과] 균열이 감지되었습니다! (총 {len(crack_contours)}개 영역)")