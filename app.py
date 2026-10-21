import csv
import os
import tempfile
import time
from datetime import datetime

import cv2
import streamlit as st

import security_camera as sc   # our OpenCV pipeline

OUTPUT_DIR = "output"
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
LOG_PATH = os.path.join(OUTPUT_DIR, "events.csv")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

st.set_page_config(page_title="Security Camera", page_icon="🎥")


@st.cache_resource
def get_cascades():
    """Load the Haar cascades once instead of on every rerun."""
    return sc.load_cascades()


def open_source(mode, camera_index, uploaded):
    """Open the webcam or an uploaded video. Returns cv2.VideoCapture or None."""
    if mode == "Webcam":
        return cv2.VideoCapture(int(camera_index))
    if uploaded is None:
        return None
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
    return cv2.VideoCapture(tmp.name)


def read_events(limit=10):
    """Last rows of the alert log, newest first."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{"Time": r["timestamp"], "Event": r["event"]} for r in rows[-limit:][::-1]]


# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    mode = st.radio("Source", ["Webcam", "Upload video"])
    camera_index = st.number_input("Camera number", 0, 5, 0) if mode == "Webcam" else 0
    uploaded = st.file_uploader("Video file", type=["mp4", "avi", "mov"]) if mode == "Upload video" else None
    sensitivity = st.slider("Sensitivity", 1, 10, 5, help="Higher = detects smaller movements")
    show_mask = st.checkbox("Show motion mask", value=False)
    run = st.toggle("Start camera")

# Slider 1-10 -> minimum motion area (bigger sensitivity = smaller area)
sc.MIN_CONTOUR_AREA = int(6000 - sensitivity * 500)

# ---------------------------- Main page ----------------------------
st.title("Security Camera")
status_box = st.empty()
if show_mask:
    cam_col, mask_col = st.columns(2)       # camera on the left, mask on the right
else:
    cam_col, mask_col = st.container(), None
frame_box = cam_col.empty()
mask_box = mask_col.empty() if mask_col is not None else None
st.subheader("Alert log")
log_box = st.empty()

if "alerts" not in st.session_state:
    st.session_state.alerts = 0


def show_log():
    events = read_events()
    if events:
        log_box.table(events)
    else:
        log_box.caption("No alerts yet.")


if not run:
    status_box.info("Camera is off. Turn on **Start camera** in the sidebar.")
    show_log()
else:
    cap = open_source(mode, camera_index, uploaded)
    if cap is None:
        status_box.warning("Please upload a video first.")
    elif not cap.isOpened():
        status_box.error("Could not open the camera or video.")
    else:
        face_cascade, body_cascade = get_cascades()
        bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        frame_count, last_alert = 0, 0.0
        show_log()

        while True:
            ok, raw = cap.read()
            if not ok:
                status_box.success("Video finished.")
                break
            frame_count += 1

            # preprocess -> segment -> find motion regions
            frame, gray, blurred = sc.preprocess(raw)
            mask = sc.segment_motion(bg, blurred)
            armed = frame_count > sc.WARMUP_FRAMES
            regions = sc.find_motion_regions(mask) if armed else []

            # label each region (PERSON / MOTION) and draw boxes
            labels = []
            for (x, y, w, h, area) in regions:
                label, detections = sc.classify_region(gray, (x, y, w, h), face_cascade, body_cascade)
                labels.append(label)
                color = sc.RED if label == "PERSON" else sc.ORANGE
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                sc.draw_text(frame, label, (x, max(15, y - 8)), color)
                for _, (dx, dy, dw, dh) in detections:
                    cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), sc.GREEN, 2)

            # save a snapshot and log it (with cooldown)
            now = time.time()
            if regions and now - last_alert > sc.ALERT_COOLDOWN_SEC:
                event = "PERSON" if "PERSON" in labels else "MOTION"
                path = os.path.join(SNAPSHOT_DIR, f"{event.lower()}_{datetime.now():%Y%m%d_%H%M%S}.jpg")
                cv2.imwrite(path, frame)
                sc.log_event(LOG_PATH, event, path, len(regions))
                last_alert = now
                st.session_state.alerts += 1
                show_log()

            # update the page
            if not armed:
                status_box.info("Learning background... keep still.")
            elif "PERSON" in labels:
                status_box.error("🔴 Person detected")
            elif regions:
                status_box.warning("🟠 Motion detected")
            else:
                status_box.success("🟢 All clear")

            frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            if mask_box is not None:
                mask_box.image(mask, caption="Motion mask", use_container_width=True)

        cap.release()
