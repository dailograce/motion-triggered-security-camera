"""
Motion-Triggered Security Camera

Pipeline (OpenCV topics used):
  (a) Image Operations ............ resize, grayscale, Gaussian blur, overlays/text
  (b) Image Segmentation .......... background subtraction (MOG2), thresholding,
                                    morphological opening/dilation, contours
  (c) Haar Cascade Classifier ..... face + upper-body detection inside motion regions
                                    to tell "PERSON" apart from generic "MOTION"
  (d) Image Analysis .............. contour area / bounding-box analysis, area filtering
  (e) Motion and Object Tracking .. motion detection per frame + simple centroid tracker

Keys:  q = quit | m = show/hide mask | r = reset background | s = manual snapshot
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime

import cv2

# Settings (can be changed here or through command-line arguments)

FRAME_WIDTH = 640          # every frame is resized to this width for speed
WARMUP_FRAMES = 40         # frames used to learn the background before arming
MIN_CONTOUR_AREA = 1500    # ignore motion blobs smaller than this (noise)
ALERT_COOLDOWN_SEC = 5     # minimum seconds between two saved alerts
MAX_TRACK_DISTANCE = 80    # max pixel distance to match a blob to an existing track
MAX_MISSED_FRAMES = 15     # drop a track after it is missing this long

# BGR colors used for drawing
GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# Helper classes / functions

class CentroidTracker:
    """Very small centroid tracker: gives each moving object a persistent ID.

    Each frame we receive the centroids of detected motion blobs. A blob is
    matched to the nearest existing track (if close enough), otherwise a new ID
    is created. Tracks that are not seen for MAX_MISSED_FRAMES are removed.
    """

    def __init__(self, max_distance=MAX_TRACK_DISTANCE, max_missed=MAX_MISSED_FRAMES):
        self.next_id = 1
        self.tracks = {}          # id -> {"centroid": (x, y), "missed": int, "path": [...]}
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, centroids):
        """Update tracks with the current centroids. Returns {id: centroid}."""
        unmatched = list(range(len(centroids)))

        # Match each existing track to the nearest unmatched centroid
        for track_id, track in list(self.tracks.items()):
            best_idx, best_dist = None, self.max_distance
            for idx in unmatched:
                dist = math.dist(track["centroid"], centroids[idx])
                if dist < best_dist:
                    best_idx, best_dist = idx, dist

            if best_idx is None:
                track["missed"] += 1
                if track["missed"] > self.max_missed:
                    del self.tracks[track_id]
            else:
                track["centroid"] = centroids[best_idx]
                track["missed"] = 0
                track["path"].append(centroids[best_idx])
                track["path"] = track["path"][-30:]   # keep a short trail
                unmatched.remove(best_idx)

        # Any centroid left over is a new object
        for idx in unmatched:
            self.tracks[self.next_id] = {
                "centroid": centroids[idx],
                "missed": 0,
                "path": [centroids[idx]],
            }
            self.next_id += 1

        return {tid: t["centroid"] for tid, t in self.tracks.items() if t["missed"] == 0}


def load_cascades():
    """Load the Haar cascade classifiers that ship with OpenCV."""
    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    body_path = cv2.data.haarcascades + "haarcascade_upperbody.xml"
    face_cascade = cv2.CascadeClassifier(face_path)
    body_cascade = cv2.CascadeClassifier(body_path)
    if face_cascade.empty() or body_cascade.empty():
        raise RuntimeError("Could not load Haar cascade files from OpenCV.")
    return face_cascade, body_cascade


def preprocess(frame):
    """Image operations: resize -> grayscale -> blur. Returns (frame, blurred_gray)."""
    h, w = frame.shape[:2]
    scale = FRAME_WIDTH / w
    frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)   # smooths out sensor noise
    return frame, gray, blurred


def segment_motion(bg_subtractor, blurred):
    """Image segmentation: background subtraction + threshold + morphology.

    MOG2 marks shadows as gray (127), so thresholding at 200 keeps only true
    foreground (255). Opening removes small specks, dilation fills holes.
    """
    fg_mask = bg_subtractor.apply(blurred)
    _, mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=3)
    return mask


def find_motion_regions(mask):
    """Image analysis: extract contours and keep only the large ones.

    Returns a list of (x, y, w, h, area) bounding boxes.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue                      # too small -> probably noise
        x, y, w, h = cv2.boundingRect(contour)
        regions.append((x, y, w, h, area))
    return regions


def classify_region(gray, box, face_cascade, body_cascade):
    """Haar cascades: is the moving blob a person? Returns (label, detections).

    We only search inside the (padded) motion box, which is faster and reduces
    false positives compared to scanning the entire frame.
    """
    x, y, w, h = box
    pad_x, pad_y = int(w * 0.15), int(h * 0.15)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(gray.shape[1], x + w + pad_x), min(gray.shape[0], y + h + pad_y)
    roi = gray[y0:y1, x0:x1]

    detections = []
    if roi.shape[0] < 60 or roi.shape[1] < 60:
        return "MOTION", detections       # region too small for the classifiers

    roi = cv2.equalizeHist(roi)           # improves detection in uneven lighting

    faces = face_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (fx, fy, fw, fh) in faces:
        detections.append(("face", (x0 + fx, y0 + fy, fw, fh)))

    bodies = body_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
    for (bx, by, bw, bh) in bodies:
        detections.append(("body", (x0 + bx, y0 + by, bw, bh)))

    label = "PERSON" if detections else "MOTION"
    return label, detections


def draw_text(frame, text, origin, color=WHITE, scale=0.6):
    """Draw text with a dark outline so it is readable on any background."""
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def log_event(log_path, label, snapshot_path, object_count):
    """Append one alert row to the CSV event log (creates the header if new)."""
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "event", "objects", "snapshot"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, object_count, snapshot_path])


def parse_args():
    parser = argparse.ArgumentParser(description="Motion-triggered security camera (OpenCV)")
    parser.add_argument("--source", default="0",
                        help="webcam index (0, 1, ...) or path to a video file (default: 0)")
    parser.add_argument("--output-dir", default="output",
                        help="folder for snapshots, event log and recorded video")
    parser.add_argument("--record", action="store_true",
                        help="also save the annotated video to the output folder")
    return parser.parse_args()



# Main loop
def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    snapshot_dir = os.path.join(args.output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "events.csv")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    face_cascade, body_cascade = load_cascades()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    tracker = CentroidTracker()

    writer = None
    frame_count = 0
    last_alert_time = 0.0
    alert_total = 0
    show_mask = False
    prev_time = time.time()
    fps = 0.0

    print("Camera started. Learning background... keep the scene still.")

    while True:
        ok, raw = cap.read()
        if not ok:
            print("End of video / camera disconnected.")
            break
        frame_count += 1

        # 1) Preprocess 
        frame, gray, blurred = preprocess(raw)

        # 2) Segment motion 
        mask = segment_motion(bg_subtractor, blurred)
        armed = frame_count > WARMUP_FRAMES

        # 3) Analyse blobs, classify and track 
        regions = find_motion_regions(mask) if armed else []
        centroids, labels = [], []

        for (x, y, w, h, area) in regions:
            label, detections = classify_region(gray, (x, y, w, h), face_cascade, body_cascade)
            labels.append(label)
            centroids.append((x + w // 2, y + h // 2))

            color = RED if label == "PERSON" else ORANGE
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            draw_text(frame, f"{label} ({int(area)} px)", (x, max(15, y - 8)), color)

            # Draw the Haar detections found inside this motion box
            for kind, (dx, dy, dw, dh) in detections:
                cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), GREEN, 2)
                draw_text(frame, kind, (dx, dy + dh + 15), GREEN, 0.5)

        objects = tracker.update(centroids)
        for obj_id, (cx, cy) in objects.items():
            cv2.circle(frame, (cx, cy), 4, WHITE, -1)
            draw_text(frame, f"ID {obj_id}", (cx + 6, cy), WHITE, 0.5)
            # draw the recent path as a trail
            path = tracker.tracks[obj_id]["path"]
            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], WHITE, 1)

        # 4) Trigger an alert (with cooldown) 
        now = time.time()
        if regions and (now - last_alert_time) > ALERT_COOLDOWN_SEC:
            event = "PERSON" if "PERSON" in labels else "MOTION"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(snapshot_dir, f"{event.lower()}_{stamp}.jpg")
            cv2.imwrite(snapshot_path, frame)
            log_event(log_path, event, snapshot_path, len(regions))
            last_alert_time = now
            alert_total += 1
            print(f"[ALERT] {event} detected -> {snapshot_path}")

        # 5) HUD overlay 
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        if not armed:
            status, status_color = f"LEARNING BACKGROUND {frame_count}/{WARMUP_FRAMES}", ORANGE
        elif "PERSON" in labels:
            status, status_color = "ALERT: PERSON DETECTED", RED
        elif regions:
            status, status_color = "ALERT: MOTION DETECTED", ORANGE
        else:
            status, status_color = "ARMED - no motion", GREEN

        draw_text(frame, status, (10, 25), status_color, 0.7)
        draw_text(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 50))
        draw_text(frame, f"FPS: {fps:.1f}  Alerts: {alert_total}  Tracked: {len(objects)}",
                  (10, frame.shape[0] - 12))

        # 6) Display / record 
        cv2.imshow("Security Camera", frame)
        if show_mask:
            cv2.imshow("Motion Mask", mask)

        if args.record:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(os.path.join(args.output_dir, "recording.mp4"),
                                         fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Motion Mask")
        elif key == ord("r"):
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
            frame_count = 0
            print("Background reset.")
        elif key == ord("s"):
            path = os.path.join(snapshot_dir, f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(path, frame)
            print(f"Saved {path}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done. {alert_total} alert(s) logged in {log_path}")


if __name__ == "__main__":
    main()
