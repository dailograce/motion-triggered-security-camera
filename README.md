# Motion-Triggered Security Camera

CSC 126  - Midterm Project
Members: Grace V. Dailo, Jhon Paul P. Duro

# Description

A webcam / video security monitor built with Python and classical OpenCV. It
learns the background of a scene, detects anything that moves, decides whether
the moving object is a PERSON (using Haar cascades) or just GENERIC MOTION, tracks each object with an ID, and saves a snapshot plus a CSV log
entry whenever an alert is triggered.

REAL-WOLRD USE CASE: low-cost home or small-office monitoring that records
only when something happens, instead of storing hours of empty footage.

# OpenCV techniques used

| Topic | Technique | Where |
|-------|-----------|-------|
| a. Image Operations | resize, grayscale, Gaussian blur, histogram equalization, drawing overlays | `preprocess()`, `draw_text()` |
| b. Image Segmentation | MOG2 background subtraction, thresholding, morphological opening/dilation, contours | `segment_motion()` |
| c. Haar Cascade Classifier | frontal face + upper-body cascades run inside motion regions | `classify_region()` |
| d. Image Analysis | contour area filtering, bounding boxes | `find_motion_regions()` |
| e. Motion and Object Tracking | frame-by-frame motion detection, centroid tracker with IDs and trails | `CentroidTracker` |

# Setup

```bash
git clone <your-repo-link>
cd motion-security-camera
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

# How to run

Web interface (main app)

```bash
streamlit run app.py
```

It opens in your browser. Pick a webcam or upload a video, adjust the sliders,
and turn on **Run camera** in the sidebar.

Command-line version (same pipeline, OpenCV windows):

```bash
python security_camera.py                       # default webcam
python security_camera.py --source sample_inputs/demo.mp4
python security_camera.py --record              # also save annotated video
```

Keep the scene still for the first ~2 seconds while the background is learned.

| Key | Action |
|-----|--------|
| `q` | quit (command-line version) |
| `m` | show / hide the motion mask |
| `r` | reset the background model |
| `s` | save a manual snapshot |

# Output

- Live dashboard (or window) with boxes, labels (`MOTION` / `PERSON`), object IDs, trails, status, FPS
- `output/snapshots/` - an image for every alert
- `output/events.csv` - timestamp, event type, object count, snapshot path
- `output/recording.mp4` - only if `--record` is used

# Project structure

```
motion-security-camera/
  app.py               Streamlit web interface
  security_camera.py   OpenCV pipeline + command-line version
  requirements.txt
  README.md
  sample_inputs/       put a short test video here
  screenshots/         put result screenshots here
```

# Limitations

- Lighting changes (clouds, lamps turning on) can cause false motion.
- Haar cascades work best on frontal faces and upright people; side views or people far from the camera may be labeled `MOTION`.
- The simple centroid tracker can swap IDs when two people cross paths.
- Tune `MIN_CONTOUR_AREA` and `ALERT_COOLDOWN_SEC` at the top of the script for your camera.
