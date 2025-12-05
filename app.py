import cv2
import mediapipe as mp
import time
import numpy as np
import streamlit as st
import tempfile
import os
from datetime import datetime
import json
import hashlib




st.set_page_config(page_title="Examination Malpractice Detector", page_icon="🎓", layout="wide")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists("users.json"):
        return {}
    with open("users.json", "r") as f:
        return json.load(f)

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

def signup_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = hash_password(password)
    save_users(users)
    return True

def login_user(username, password):
    users = load_users()
    if username in users and users[username] == hash_password(password):
        return True
    return False


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
         """
    <style>
    /* Hide Streamlit header and menu */
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Background styling */
    [data-testid="stAppViewContainer"] {
        background-color: #0e0e10;
        color: #ffffff;
        padding-top: 2rem;
    }

    /* Login/Signup Box */
    .stForm {
        background-color: #1c1c1e;
        border: 1px solid #2c2c2e;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.05);
    }

    /* Title styling */
    h1, h2, h3 {
        text-align: center;
        color: #ffffff;
    }

    /* Input fields */
    input {
        background-color: #2c2c2e !important;
        color: white !important;
    }

    /* Submit button */
    div.stButton > button {
        width: 100%;
        background-color: #2b65ec !important;
        color: white !important;
        border-radius: 8px;
    }

    /* Error message box */
    .error-box {
        background-color: #3a1f1f;
        padding: 10px;
        border-radius: 5px;
        color: #ffbaba;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True
    )

    left_col, mid_col, right_col = st.columns([1, 0.8, 1])
    with mid_col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='margin:0 0 6px 0; color:#fff;'>🔐 Admin Access Portal</h3>", unsafe_allow_html=True)
        with st.form(key="auth_form"):
            mode = st.radio("", ["Login", "Sign Up"], horizontal=True)
            username = st.text_input("Username", key="auth_username")
            password = st.text_input("Password", type="password", key="auth_password")
            if mode == "Sign Up":
                confirm_password = st.text_input("Confirm Password", type="password", key="auth_confirm")

            submit = st.form_submit_button("Submit")
            if submit:
                if mode == "Login":
                    if not username or not password:
                        st.error("Please enter both username and password.")
                    elif login_user(username, password):
                        st.session_state.authenticated = True
                        st.success("Login successful — loading main app...")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    if not username or not password:
                        st.error("Please provide username and password for sign up.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    elif signup_user(username, password):
                        st.success("Account created. Switch to Login and sign in.")
                    else:
                        st.error("Username already exists. Choose another.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


st.title("🎯 Examination Malpractice Detection System")
st.markdown("""
Welcome to the **Examination Malpractice Detection System**.
This system can analyze live video or uploaded exam footage to detect suspicious activity.
""")

st.sidebar.header("⚙️ Settings")
camera_index = st.sidebar.number_input("Select Camera Index", min_value=0, max_value=5, value=0, step=1)
sms_number = st.sidebar.text_input("Invigilator Phone Number (for SMS alerts)", "+234xxxxxxxxxx")

if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.experimental_rerun()

tab1, tab2 = st.tabs(["📹 Live Detection", "🎞️ Uploaded Video Analysis"])


import mediapipe as mp
import cv2
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

_last_detection_time = 0
_detection_cooldown = 2.0 

def detect_malpractice(frame, yaw_threshold=4.0):
    global _last_detection_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(rgb)
    results_hands = hands_detector.process(rgb)

    h, w, _ = frame.shape
    detected = False

  
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]
            mid_x = (left_eye.x + right_eye.x) / 2
            yaw = (nose_tip.x - mid_x) * 100
            if abs(yaw) > yaw_threshold:
                detected = True

   
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(xs) * w), int(max(xs) * w)
            ymin, ymax = int(min(ys) * h), int(max(ys) * h)
            xmin, ymin = max(0, xmin - 25), max(0, ymin - 25)
            xmax, ymax = min(w, xmax + 25), min(h, ymax + 25)
            hand_region = frame[ymin:ymax, xmin:xmax]

            if hand_region.size > 0:
                gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)

               
                hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
                lower_white = np.array([0, 0, 180])
                upper_white = np.array([180, 50, 255])
                mask_white = cv2.inRange(hsv, lower_white, upper_white)
                white_ratio = np.sum(mask_white > 0) / mask_white.size

               
                _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
                dark_ratio = np.sum(thresh > 0) / thresh.size

                if white_ratio > 0.25:   # paper covers 25%+ of hand box
                    detected = True
                elif dark_ratio > 0.35:  # phone-like dark region
                    detected = True

    
    now = time.time()
    if detected:
        _last_detection_time = now
        return True
    elif now - _last_detection_time < _detection_cooldown:
        return True

    return False

# ---------------------- SEND SMS PLACEHOLDER ----------------------
def send_sms_alert(row, column, phone_number):
    st.warning(f"🚨 SMS Sent to {phone_number}: Suspicious activity detected at Row {row}, Column {column}!")

with tab1:
    st.subheader("🎥 Live Camera Monitoring")
    st.markdown("Use your computer webcam or an external camera for real-time monitoring.")

    if "detections" not in st.session_state:
        st.session_state.detections = []
    if "live_mode" not in st.session_state:
        st.session_state.live_mode = False

    start_live = st.button("▶️ Start Live Detection")
    stop_live = st.button("⏹ Stop Live Detection")

    if start_live:
        st.session_state.live_mode = True
    if stop_live:
        st.session_state.live_mode = False

    frame_placeholder = st.empty()
    os.makedirs("detected_frames", exist_ok=True)

    if st.session_state.live_mode:
        try:
            cap = cv2.VideoCapture(int(camera_index))
        except Exception as e:
            st.error(f"❌ Error opening camera: {e}")
            st.session_state.live_mode = False
            st.stop()

        if cap is None or not cap.isOpened():
            st.error("❌ Could not open camera.")
            st.session_state.live_mode = False
            st.stop()
        else:
            st.info("✅ Camera started successfully.")

            last_alert_time = 0
            alert_cooldown = 5 

            while st.session_state.live_mode:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera.")
                    break

                detected = detect_malpractice(frame)
                now = time.time()

                if detected and now - last_alert_time > alert_cooldown:
                    last_alert_time = now
                    row, column = np.random.randint(1, 5), np.random.randint(1, 5)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detected_frames/malpractice_{row}_{column}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    st.session_state.detections.append({
                        "row": row, "column": column, "time": timestamp, "file": filename
                    })
                    send_sms_alert(row, column, sms_number)
                    st.error(f"🚨 Malpractice Detected at Row {row}, Column {column}")

                
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                time.sleep(0.03)

            cap.release()
            frame_placeholder.empty()
            st.success("🛑 Live detection stopped.")

    st.markdown("---")
    st.subheader("🧠 Detection Analysis")
    if st.button("🗑️ Reset All Detections"):
        st.session_state.detections = []
        st.success("✅ All detections cleared.")

    if st.session_state.detections:
        for det in st.session_state.detections:
            with st.expander(f"🚨 Row {det['row']} | Column {det['column']} | Time: {det['time']}"):
                st.image(det['file'], caption=f"Row {det['row']}, Column {det['column']}", width=400)
    else:
        st.info("No malpractice detected yet.")

with tab2:
    st.subheader("📁 Uploaded Video Analysis")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if "video_detections" not in st.session_state:
        st.session_state.video_detections = []
    if "uploaded_video_path" not in st.session_state:
        st.session_state.uploaded_video_path = None

    if st.session_state.uploaded_video_path or st.session_state.video_detections:
        if st.button("🗑️ Reset Uploaded Video Section", use_container_width=True):
            st.session_state.video_detections = []
            st.session_state.uploaded_video_path = None
            st.experimental_rerun() 

    if uploaded_video:
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.session_state.uploaded_video_path = tfile.name

        st.markdown("<h6>🎬 Uploaded Video Preview</h6>", unsafe_allow_html=True)
        st.video(tfile.name)
        st.markdown(
            "<style>video{max-width:480px !important; height:auto !important;}</style>",
            unsafe_allow_html=True,
        )

        start_analysis = st.button("▶️ Start Analysis", use_container_width=True)
        if start_analysis:
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            os.makedirs("video_detected_frames", exist_ok=True)

            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            last_alert_time = 0
            alert_cooldown = 5  
            detection_count = 0

            st.info("🔍 Analyzing video, please wait...")

            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detected = detect_malpractice(frame)
                now = time.time()

                if detected and now - last_alert_time > alert_cooldown:
                    last_alert_time = now
                    detection_count += 1
                    row, column = np.random.randint(1, 5), np.random.randint(1, 5)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"video_detected_frames/malpractice_{row}_{column}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    st.session_state.video_detections.append({
                        "row": row, "column": column, "time": timestamp, "file": filename
                    })
                    send_sms_alert(row, column, sms_number)
                    st.error(f"🚨 Malpractice Detected at Row {row}, Column {column} (Frame {frame_index})")

                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width=480
                )

                progress_percent = int((frame_index / total_frames) * 100)
                progress_bar.progress(min(progress_percent, 100))
                frame_index += 1
                time.sleep(1 / (fps if fps > 0 else 30))

            cap.release()
            progress_bar.progress(100)
            st.success(f"✅ Video analysis completed with {detection_count} detections.")

    st.markdown("---")
    st.subheader("🧠 Detection Analysis")

    if st.session_state.video_detections:
        for det in st.session_state.video_detections:
            with st.expander(f"🚨 Row {det['row']} | Column {det['column']} | Time: {det['time']}"):
                st.image(det["file"], caption=f"Row {det['row']}, Column {det['column']}", width=400)
    else:
        st.info("No malpractice detected yet.")
