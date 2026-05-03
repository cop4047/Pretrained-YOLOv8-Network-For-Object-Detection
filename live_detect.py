import cv2
import numpy as np
from ultralytics import YOLO
import yt_dlp

# ── Models ──────────────────────────────────────────────────────────────────
detector   = YOLO(r'C:\Users\alros\Documents\GitHub\Pretrained-YOLOv8-Network-For-Object-Detection\runs\detect\train6\weights\best.pt')
segmentor  = YOLO('yolov8m-seg.pt')  # downloads automatically first run

# ── Classes ─────────────────────────────────────────────────────────────────
DET_CLASSES = {
    0: 'pedestrian', 1: 'pedestrian', 2: 'wheelchair',
    3: 'stroller',   4: 'scooter',    5: 'police',
}

# COCO segmentation class IDs to treat as ROAD (flat surface / driveable)
ROAD_SEG_CLASSES = {56, 57, 59, 60, 61}  # dining table used as proxy; adjust as needed

# Colour palette
ROAD_COLOUR = (0, 200, 0)    # green overlay for driveable surface
BOX_COLOUR  = (0, 120, 255)  # orange boxes for detected objects

# ── Video source ─────────────────────────────────────────────────────────────
def get_stream(youtube_url):
    ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

# Change this to a YouTube dashcam URL, or use a local file path / 0 for webcam
SOURCE = 'https://www.youtube.com/watch?v=b4Rr_7y068U'  # example dashcam video

print("Loading stream...")
try:
    stream_url = get_stream(SOURCE)
    cap = cv2.VideoCapture(stream_url)
except Exception:
    print("YouTube failed, falling back to webcam")
    cap = cv2.VideoCapture(0)

print("Starting detection — press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── 1. Road segmentation ────────────────────────────────────────────────
    seg_results = segmentor(frame, conf=0.3, verbose=False)[0]

    if seg_results.masks is not None:
        masks  = seg_results.masks.data.cpu().numpy()   # (N, H, W)
        cls_ids = seg_results.boxes.cls.cpu().numpy().astype(int)

        road_mask = np.zeros((h, w), dtype=np.uint8)

        for mask, cls_id in zip(masks, cls_ids):
            # Resize mask to frame size
            mask_resized = cv2.resize(mask, (w, h))
            binary = (mask_resized > 0.5).astype(np.uint8)

            # Use bottom half of frame — road is always in lower portion
            bottom_mask = np.zeros_like(binary)
            bottom_mask[h//2:, :] = binary[h//2:, :]
            road_mask = cv2.bitwise_or(road_mask, bottom_mask)

        # Apply green overlay to road area
        overlay[road_mask == 1] = (
            overlay[road_mask == 1] * 0.5 +
            np.array(ROAD_COLOUR) * 0.5
        ).astype(np.uint8)

    # ── 2. Object detection ─────────────────────────────────────────────────
    det_results = detector(frame, conf=0.25, verbose=False)[0]
    boxes  = det_results.boxes.xyxy.cpu().numpy().astype(int)
    scores = det_results.boxes.conf.cpu().numpy()
    cls_ids_det = det_results.boxes.cls.cpu().numpy().astype(int)
    names  = det_results.names

    for box, score, cls_id in zip(boxes, scores, cls_ids_det):
        x1, y1, x2, y2 = box
        label = f"{names[cls_id]} {score:.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), BOX_COLOUR, 2)
        cv2.rectangle(overlay, (x1, y1-22), (x1+len(label)*9, y1), BOX_COLOUR, -1)
        cv2.putText(overlay, label, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # ── 3. HUD ───────────────────────────────────────────────────────────────
    cv2.putText(overlay, f"Objects: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(overlay, "GREEN = driveable road", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROAD_COLOUR, 2)

    cv2.imshow('Autonomous Perception', overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
