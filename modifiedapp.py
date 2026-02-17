import streamlit as st
import cv2
from ultralytics import YOLO
import sqlite3
import datetime
import tempfile
import os
import time

# Constants for Optimization
SKIP_FRAMES = 5  # Only run AI on every 5th frame
ALERT_COOLDOWN = 2  # Seconds between alarm sounds
PROCESS_SIZE = (640, 480) # Resize internal frames for speed

# Database setup
conn = sqlite3.connect('crowd_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS alerts (timestamp TEXT, count INTEGER, status TEXT)')
conn.commit()

# Session State for tracking
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0
if 'last_status' not in st.session_state:
    st.session_state.last_status = "NORMAL"

st.set_page_config(page_title="AI Crowd Guard", layout="wide")
st.title("🛡️ Smart Crowd Monitoring System")

# Alarm sound with cooldown
def play_alarm():
    current_time = time.time()
    if current_time - st.session_state.last_alert_time > ALERT_COOLDOWN:
        try:
            st_alarm.audio("alarm.mp3", autoplay=True)
            st.session_state.last_alert_time = current_time
        except:
            pass

# Sidebar
st.sidebar.header("System Controls")
mode = st.sidebar.radio("Input Source", ("Webcam", "Video File", "Log History"))
limit = st.sidebar.slider("Set Crowd Limit", 0, 50, 5)

# Placeholders
st_count = st.sidebar.empty()
st_status = st.empty()  
st_alarm = st.empty()   

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Process frame
def process_frame(frame, st_frame):
    if frame is None:
        return

    st.session_state.frame_count += 1
    
    # Only run AI on every Nth frame to save CPU
    should_run_ai = (st.session_state.frame_count % SKIP_FRAMES == 0)
    
    if should_run_ai:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, PROCESS_SIZE)
        results = model(small_frame, verbose=False)

        person_count = 0
        if results and len(results) > 0 and results[0].boxes is not None:
            person_count = int(sum(
                1 for b in results[0].boxes.cls
                if model.names[int(b)] == 'person'
            ))

        st_count.metric("People Detected", person_count)

        if person_count > limit:
            st_status.error(f"🚨 ALERT: {person_count} People Detected! (Limit: {limit})")
            play_alarm()

            # State-change Database Logging (Only log once per crowded event)
            if st.session_state.last_status == "NORMAL":
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                c.execute('INSERT INTO alerts VALUES (?, ?, ?)', (now, person_count, "CROWDED"))
                conn.commit()
                st.session_state.last_status = "CROWDED"
        else:
            st_status.success(f"✅ Status: Normal ({person_count} people)")
            st_alarm.empty()
            st.session_state.last_status = "NORMAL"

        # Render annotated frame
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(annotated_frame, use_column_width=True)
    else:
        # Just show raw frame BGR->RGB for skipped frames (keeps video fluid)
        raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(raw_rgb, use_column_width=True)


# Webcam mode
if mode == "Webcam":
    st.warning("Webcam may not work on cloud deployments.")
    run = st.checkbox("Start Live Stream")
    st_frame = st.empty()
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        process_frame(frame, st_frame)
    cap.release()


# Video file mode
elif mode == "Video File":
    uploaded = st.file_uploader("Upload MP4 Video", type=['mp4'])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            process_frame(frame, st_frame)

        cap.release()
        os.unlink(tfile.name)


# Log history
elif mode == "Log History":
    st.header("📋 System Log History")
    data = c.execute('SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 25').fetchall()
    if data:
        st.table(data)
    else:
        st.info("No logs found yet.")
