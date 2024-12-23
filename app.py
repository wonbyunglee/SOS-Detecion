import cv2
import dlib
import numpy as np
import time

import streamlit as st
from PIL import Image

# ---------------------------streamlit 화면 설정---------------------------
st.set_page_config(
    page_title="SOS 신호 관리 페이지",
    page_icon="🚨",
    layout="centered",
)

st.markdown("""
    <style>
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #f0f2f6;
        padding: 10px 0;
        z-index: 100;
        border-bottom: 1px solid #ccc;
    }
    .title {
        font-size: 36px;
        color: #ff4b4b;
        text-align: center;
    }
    .subtitle {
        font-size: 24px;
        color: #333333;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
st.markdown('<div class="title">🚨 SOS 신호 관리 페이지 🚨</div>', unsafe_allow_html=True)
st.info(" 이 페이지는 실시간 신호 감지를 지원합니다. 👀")
st.markdown('</div>', unsafe_allow_html=True)

morse_code = []

# 상태 초기화 함수
def reset_state():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

# 사이드바 버튼 설정
if st.sidebar.button("🗑️ 상태 초기화", key="reset_button"):
    reset_state()
if st.sidebar.button("⚠️ 긴급 도움 요청", key="emergency_button"):
    st.warning(" 112에 연결 중입니다... 📞")

# ---------------------------웹캠 실행---------------------------
cap = cv2.VideoCapture(0)
stframe = st.empty()

# NOTE: Haar Cascade Classifier(얼굴과 눈 감지)
# Source: https://github.com/opencv/opencv/tree/4.x/data/haarcascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# NOTE: Dlib(얼굴 랜드마크 탐지 모델 로드)
# Source: https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/blob/ccfc24642e0210d4d885bc7b3dbc9a68ed948ad6/shape_predictor_68_face_landmarks.dat
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# NOTE: EAR 알고리즘 계산 함수
# Source: https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # 눈 앞쪽 수직 거리
    B = np.linalg.norm(eye[2] - eye[4])  # 눈 뒷쪽 수직 거리
    C = np.linalg.norm(eye[0] - eye[3])  # 수평 거리
    ear = (A + B) / (2.0 * C)
    return ear

# NOTE: This section of code for specifying landmark numbers is based on OpenCV resources
# 눈 랜드마크 추출 함수
def get_eye_landmarks(landmarks, eye_points):
    return np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in eye_points], dtype=np.float32)

THRESHOLD_EAR = 0.2  # EAR 임계값 지정

# 초기값 설정
blink_start_time = None  # 눈 감기 시작
blink_end_time = None    # 눈 감기 종료

def check_sos_signal(morse_code):
    recent_patterns = ''.join(morse_code)
    
    # '_'와 '.'의 개수를 세기
    count_dot = recent_patterns.count('.')
    count_underscore = recent_patterns.count('_')

    # SOS 신호의 대략적인 비율 이상을 포함하고 있으면 어떠한 신호를 보내고 있다고 가정함
    if count_underscore >= 2 and count_dot >= 3:
        return True
    return False

SOS_DETECTED = False
SOS_SENT = False

cv2.namedWindow("Eyes", cv2.WINDOW_NORMAL)

# 일정 시간동안 얼굴 정면이 바라보고 있는지 감지
face_detected_start_time = None  # 얼굴이 감지된 시작 시간
FACE_DETECTION_THRESHOLD = 5    # 얼굴을 정면으로 5초 이상 바라봤을 때 sos 신호로 인지

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar Cascade로 얼굴 감지
    haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 얼굴 감지 여부 확인
    if len(haar_faces) > 0:
        if face_detected_start_time is None:
            face_detected_start_time = time.time() # 감지되면 시간 측정 시작
    else:
        face_detected_start_time = None
        SOS_DETECTED = False
        SOS_SENT = False
        morse_code.clear()

    if face_detected_start_time:
        elapsed_time = time.time() - face_detected_start_time
        if elapsed_time >= FACE_DETECTION_THRESHOLD:
            cv2.putText(frame, "Receive signals", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 5초 이상 화면을 바라봤으면 눈 깜빡임 감지
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)

                # NOTE: This section of code for specifying landmark numbers is based on OpenCV resources and PyImageSearch tutorials
                # 왼쪽 눈과 오른쪽 눈의 랜드마크 가져오기
                left_eye_points = list(range(36, 42))
                right_eye_points = list(range(42, 48))
                left_eye = get_eye_landmarks(landmarks, left_eye_points)
                right_eye = get_eye_landmarks(landmarks, right_eye_points)

                # 눈 영역 자르기
                left_eye_center = np.mean(left_eye, axis=0).astype(int)
                right_eye_center = np.mean(right_eye, axis=0).astype(int)

                left_eye_img = frame[left_eye_center[1] - 20:left_eye_center[1] + 20, left_eye_center[0] - 20:left_eye_center[0] + 20]
                right_eye_img = frame[right_eye_center[1] - 20:right_eye_center[1] + 20, right_eye_center[0] - 20:right_eye_center[0] + 20]

                # NOTE: The code for calculating eye size using scale_factor is based on OpenCV resources
                #스케일 1.5배 확대
                scale_factor = 1.5
                left_eye_scaled = (left_eye - left_eye_center) * scale_factor + left_eye_center
                right_eye_scaled = (right_eye - right_eye_center) * scale_factor + right_eye_center

                # 확대된 눈을 기준으로 랜드마크 포인트 표시하기
                for i in range(36, 42):  # 왼쪽 눈
                    x_point = landmarks.part(i).x
                    y_point = landmarks.part(i).y
                    x_scaled = int((x_point - left_eye_center[0]) * scale_factor + left_eye_center[0])
                    y_scaled = int((y_point - left_eye_center[1]) * scale_factor + left_eye_center[1])
                    cv2.circle(frame, (x_scaled, y_scaled), 1, (0, 0, 255), -1)

                for i in range(42, 48):  # 오른쪽 눈
                    x_point = landmarks.part(i).x
                    y_point = landmarks.part(i).y
                    x_scaled = int((x_point - right_eye_center[0]) * scale_factor + right_eye_center[0])
                    y_scaled = int((y_point - right_eye_center[1]) * scale_factor + right_eye_center[1])
                    cv2.circle(frame, (x_scaled, y_scaled), 1, (0, 0, 255), -1)

                # EAR 계산
                left_ear = calculate_ear(left_eye_scaled)
                right_ear = calculate_ear(right_eye_scaled)
                ear = (left_ear + right_ear) / 2.0

                # 두 개의 눈 이미지를 가로로 합치기
                combined_eye_img = np.hstack((left_eye_img, right_eye_img))
                cv2.imshow("Eyes", combined_eye_img)
                
                cooldown_time = 0.2  # 200ms 동안은 중복 감지 못하도록 설정
                last_blink_time = 0

                if time.time() - last_blink_time > cooldown_time:
                    if ear < THRESHOLD_EAR:
                        # 눈 감김 시작 처리
                        if blink_start_time is None:
                            blink_start_time = time.time()
                    else:
                        if blink_start_time is not None:
                            blink_end_time = time.time()
                            duration = blink_end_time - blink_start_time
                            blink_start_time = None
                            last_blink_time = time.time()

                            if duration <= 0.35:
                                morse_code.append('.')
                                print("Morse code:", ''.join(morse_code))
                            else:
                                morse_code.append('_')
                                print("Morse code:", ''.join(morse_code))

                            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}], Blink Duration: {duration:.2f}s, EAR: {ear:.2f}, morse_code: {' '.join(morse_code)}")

                # EAR 값 및 패턴 화면 표시
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"morse_code: {' '.join(morse_code)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if ''.join(morse_code).count('_') >= 3:
        SOS_DETECTED = True

    if SOS_DETECTED:
        morse_string = ''.join(morse_code)

        # 정확한 SOS 신호 감지
        if "...___..." in morse_string and not SOS_SENT:
            SOS_TIME = time.strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, "SOS Detected!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 255), thickness=3)

            # Streamlit에서 SOS 신호 출력
            st.write(f"감지 시간: {SOS_TIME}")
            st.write(f"모스 부호: {morse_string}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            st.image(image, use_column_width=True)
            st.error("SOS signal!")

            # SOS 신호가 감지되면 상태를 업데이트
            SOS_SENT = True
            morse_code.clear()

        # 위험 신호 감지
        elif "...___" in morse_string and "...___..." not in morse_string and not SOS_SENT:
            SOS_TIME = time.strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, "Warning signal!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 140, 255), thickness=3)

            # Streamlit에서 위험 신호 출력
            st.write(f"감지 시간: {SOS_TIME}")
            st.write(f"모스 부호: {morse_string}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            st.image(image, use_column_width=True)
            st.success("Warning signal!")

            # 위험 신호가 감지되면 상태를 업데이트
            SOS_SENT = True

        # 위험 신호 이후 SOS 신호 감지 시 상태 리셋
        if "...___..." in morse_string and SOS_SENT:
            SOS_SENT = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 'q' 키를 눌러 초기화
            SOS_DETECTED = False
            SOS_SENT = False
            morse_code.clear()
            print("Reset")
            face_detected_start_time = None

    cv2.imshow("Frame", frame)

    # esc 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()    