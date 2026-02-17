import streamlit as st
import cv2
from ultralytics import YOLO
import sqlite3
import datetime
import tempfile
import os

# Database setup
conn = sqlite3.connect('crowd_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS alerts (timestamp TEXT, count INTEGER, status TEXT)')
conn.commit()

st.set_page_config(page_title="AI Crowd Guard", layout="wide")
st.title("🛡️ Smart Crowd Monitoring System")

# Alarm sound (cloud compatible)
def play_alarm():
    try:
        st.audio("alarm.mp3", autoplay=True)
    except:
        pass

# Sidebar controls
st.sidebar.header("System Controls")
mode = st.sidebar.radio("Input Source", ("Webcam", "Video File", "Log History"))
limit = st.sidebar.slider("Set Crowd Limit", 0, 50, 5)
st_count = st.sidebar.empty()

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

def process_frame(frame, st_frame):

    if frame is None:
        return

    results = model(frame, verbose=False)

    person_count = 0

if results and len(results) > 0 and results[0] is not None:

        person_count = sum(
            1 for b in results[0].boxes.cls
            if model.names[int(b)] == 'person'
        )

    st_count.metric("People Detected", person_count)

    if person_count > limit:

        st.error(f"🚨 ALERT: {person_count} People Detected! (Limit: {limit})")
        play_alarm()

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        c.execute(
            'INSERT INTO alerts VALUES (?, ?, ?)',
            (now, person_count, "CROWDED")
        )

        conn.commit()

    else:

        st.success(f"✅ Status: Normal ({person_count} people)")


    # Safe frame plotting
    if results and results[0] is not None:

        annotated_frame = results[0].plot()

        if annotated_frame is not None:

            annotated_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            st_frame.image(
                annotated_frame,
                use_container_width=True
            )


# Webcam mode
if mode == "Webcam":

    st.warning("Webcam may not work on cloud deployments.")

    run = st.checkbox("Start Live Stream")

    st_frame = st.empty()

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()

        if not ret:
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

    data = c.execute(
        'SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 25'
    ).fetchall()

    if data:
        st.table(data)
    else:
        st.info("No logs found yet.")
