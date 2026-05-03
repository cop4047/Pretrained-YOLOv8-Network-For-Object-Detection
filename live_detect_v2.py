import cv2
import numpy as np
import torch
import mss
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading nuScenes object detector...")
detector = YOLO(r'C:\Users\alros\Documents\GitHub\Pretrained-YOLOv8-Network-For-Object-Detection\runs\detect\train8\weights\best.pt')

print("Loading traffic light / stop sign detector (COCO)...")
coco_detector = YOLO('yolov8m.pt')
COCO_CLASSES_KEEP = {9: 'traffic light', 11: 'stop sign', 12: 'parking meter'}

print("Loading road segmentation model...")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
seg_model.eval()

ROAD_ID     = 0
SIDEWALK_ID = 1

SEG_EVERY   = 3
frame_count = 0
seg_map     = None

print("Running — press Q to quit")
print("Play a dashcam video on your screen and this window will detect in real time")

with mss.mss() as sct:
    # Capture your primary monitor
    monitor = sct.monitors[1]

    while True:
        # Grab screen
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Resize to something manageable if your screen is large
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))
            h, w = frame.shape[:2]

        overlay = frame.copy()

        # ── 1. Road segmentation (every N frames) ────────────────────────────
        if frame_count % SEG_EVERY == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt")
            with torch.no_grad():
                logits = seg_model(**inputs).logits
            seg_map = torch.argmax(logits, dim=1).squeeze().numpy().astype(np.uint8)
            seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)

        if seg_map is not None:
            road_mask = (seg_map == ROAD_ID)
            overlay[road_mask] = (
                overlay[road_mask] * 0.5 + np.array([0, 200, 0]) * 0.5
            ).astype(np.uint8)

            sidewalk_mask = (seg_map == SIDEWALK_ID)
            overlay[sidewalk_mask] = (
                overlay[sidewalk_mask] * 0.5 + np.array([0, 140, 255]) * 0.5
            ).astype(np.uint8)

        # ── 2. nuScenes object detection ─────────────────────────────────────
        results  = detector(frame, conf=0.25, verbose=False)[0]
        boxes    = results.boxes.xyxy.cpu().numpy().astype(int)
        scores   = results.boxes.conf.cpu().numpy()
        cls_ids  = results.boxes.cls.cpu().numpy().astype(int)
        names    = results.names

        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box
            label = f"{names[cls_id]} {score:.2f}"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 80, 0), 2)
            cv2.rectangle(overlay, (x1, y1-22), (x1+len(label)*9, y1), (255, 80, 0), -1)
            cv2.putText(overlay, label, (x1+2, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── 3. Traffic lights and signs (COCO) ───────────────────────────────
        coco_results = coco_detector(frame, conf=0.3, verbose=False)[0]
        coco_boxes   = coco_results.boxes.xyxy.cpu().numpy().astype(int)
        coco_scores  = coco_results.boxes.conf.cpu().numpy()
        coco_cls     = coco_results.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(coco_boxes, coco_scores, coco_cls):
            if cls_id not in COCO_CLASSES_KEEP:
                continue
            x1, y1, x2, y2 = box
            label = f"{COCO_CLASSES_KEEP[cls_id]} {score:.2f}"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.rectangle(overlay, (x1, y1-22), (x1+len(label)*9, y1), (0, 255, 255), -1)
