import streamlit as st
import cv2
from ultralytics import YOLO
import sqlite3
import datetime
import tempfile
import os
import winsound  # Direct hardware beep for Windows

# 1. Database Setup - Saves alerts permanently
conn = sqlite3.connect('crowd_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS alerts (timestamp TEXT, count INTEGER, status TEXT)')
conn.commit()

# 2. Page Configuration
st.set_page_config(page_title="AI Crowd Guard", layout="wide")
st.title("🛡️ Smart Crowd Monitoring System")

# 3. Hardware Alert Function (Bypasses browser blocks)
def play_alarm():
    # Frequency: 1000Hz, Duration: 500ms
    winsound.Beep(1000, 500) 

# 4. Sidebar Controls
st.sidebar.header("System Controls")
mode = st.sidebar.radio("Input Source", ("Webcam", "Video File", "Log History"))
limit = st.sidebar.slider("Set Crowd Limit", 0, 50, 5)
st_count = st.sidebar.empty() # Placeholder for the big live number

# 5. Load AI Model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")
model = load_model()

# 6. Central Processing Function
def process_frame(frame, st_frame):
    # Perform detection
    results = model(frame, verbose=False) 
    
    # Count only 'person' class
    person_count = sum(1 for b in results[0].boxes.cls if model.names[int(b)] == 'person')
    
    # Update the sidebar count
    st_count.metric("People Detected", person_count)
    
    # Alert Logic
    if person_count > limit:
        st.error(f"🚨 ALERT: {person_count} People Detected! (Limit: {limit})")
        play_alarm() # The hardware beep
        
        # Save to Database
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute('INSERT INTO alerts VALUES (?, ?, ?)', (now, person_count, "CROWDED"))
        conn.commit()
    else:
        st.success(f"✅ Status: Normal ({person_count} people)")

    # FIXED: This renders the video correctly with boxes
    annotated_frame = results[0].plot() 
    st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

# --- APP MODES ---
if mode == "Webcam":
    run = st.checkbox("Start Live Stream")
    st_frame = st.empty()
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret: break
        process_frame(frame, st_frame)
    cap.release()

elif mode == "Video File":
    uploaded = st.file_uploader("Upload MP4 Video", type=['mp4'])
    if uploaded:
        # Create a temp file to store the upload so CV2 can read it
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            process_frame(frame, st_frame)
            
        cap.release()
        os.unlink(tfile.name) # Clean up temp file

elif mode == "Log History":
    st.header("📋 System Log History (Saved Alerts)")
    data = c.execute('SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 25').fetchall()
    if data:
        st.table(data) # Shows the database records
    else:
        st.info("No logs found yet.")