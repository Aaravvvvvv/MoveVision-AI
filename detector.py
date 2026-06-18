# detector.py
# Handles video/image processing and YOLO object detection.

from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from estimator import INVENTORY_ITEMS, canonical_inventory_item


MODEL_PATH = Path(__file__).resolve().parent / "weights" / "household_v2_best.pt"
model = YOLO(str(MODEL_PATH))

# Exported for the Streamlit inventory editor.
HOUSEHOLD_ITEMS = set(INVENTORY_ITEMS)


def confidence_level(confidence):
    """Bucket a detection confidence into a review-friendly status."""
    if confidence >= 0.65:
        return "High"
    if confidence >= 0.35:
        return "Review"
    return "Low"


def _build_confidence_summary(confidences_by_item):
    summary = {}
    for item_name, confidences in confidences_by_item.items():
        if not confidences:
            continue
        avg_confidence = sum(confidences) / len(confidences)
        summary[item_name] = {
            "avg": round(avg_confidence, 3),
            "max": round(max(confidences), 3),
            "min": round(min(confidences), 3),
            "level": confidence_level(avg_confidence),
        }
    return summary


def _box_xyxy(box, frame):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return (
        max(0, min(x1, w - 1)),
        max(0, min(y1, h - 1)),
        max(0, min(x2, w - 1)),
        max(0, min(y2, h - 1)),
    )


def _area(coords):
    x1, y1, x2, y2 = coords
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = _area((ix1, iy1, ix2, iy2))
    union = _area(a) + _area(b) - intersection
    return intersection / union if union else 0


def _center_inside(inner, outer):
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def _is_duplicate_detection(candidate, kept_detection):
    candidate_box = candidate["coords"]
    kept_box = kept_detection["coords"]
    candidate_area = _area(candidate_box)
    kept_area = _area(kept_box)
    smaller = min(candidate_area, kept_area)
    larger = max(candidate_area, kept_area)

    if smaller == 0:
        return True

    area_ratio = smaller / larger
    if _iou(candidate_box, kept_box) >= 0.20:
        return True
    if area_ratio >= 0.25 and (
        _center_inside(candidate_box, kept_box) or _center_inside(kept_box, candidate_box)
    ):
        return True
    return False


def _inventory_detections(result, frame, confidence):
    detections = []
    for box in result.boxes:
        raw_label = model.names[int(box.cls)]
        item_name = canonical_inventory_item(raw_label)
        conf_score = float(box.conf)

        if item_name and conf_score >= confidence:
            detections.append({
                "item": item_name,
                "confidence": conf_score,
                "box": box,
                "coords": _box_xyxy(box, frame),
            })

    kept = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if any(
            detection["item"] == existing["item"]
            and _is_duplicate_detection(detection, existing)
            for existing in kept
        ):
            continue
        kept.append(detection)
    return kept


def _draw_detection(frame, box, label, confidence):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"{label} {confidence:.0%}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def process_video(video_path, frame_interval=30, confidence=0.25, imgsz=960,
                  progress_callback=None, return_details=False):
    """
    Process a video file using ByteTrack object tracking.

    Each detected object receives a persistent tracker ID across frames,
    so the final count is the number of *unique* tracked objects per class
    rather than an averaged estimate.

    Tip: smaller frame_interval values (5–15) give the tracker more
    temporal context and improve accuracy.  The default 30 still works
    but may over-count if the camera revisits a room.

    Returns:
        detected_items: dict of {canonical_item_name: count}
        sample_frames: list of up to 3 annotated frame images
        confidence_summary: optional dict when return_details=True
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = round(total_frames / fps, 1) if fps > 0 else 0
    print(f"Video loaded: {total_frames} frames, {fps:.1f} fps, {duration}s duration")

    # Reset tracker state from any previous video
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor = None

    # {tracker_id: canonical_item_name}  — one entry per unique object
    tracked_objects = {}
    track_confidences = {}
    sample_frames = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=confidence,
                iou=0.5,
                imgsz=imgsz,
            )[0]

            frame_items = []

            # results.boxes.id is None when no tracks are found
            if results.boxes.id is not None:
                for box in results.boxes:
                    raw_label = model.names[int(box.cls)]
                    item_name = canonical_inventory_item(raw_label)
                    conf_score = float(box.conf)
                    track_id = int(box.id)

                    if item_name and conf_score >= confidence:
                        tracked_objects[track_id] = item_name
                        track_confidences.setdefault(track_id, []).append(conf_score)
                        frame_items.append(item_name)
                        _draw_detection(
                            frame, box,
                            f"{item_name} #{track_id}",
                            conf_score,
                        )

            if frame_items and len(sample_frames) < 3:
                sample_frames.append(frame.copy())

        frame_number += 1

        if progress_callback and frame_number % frame_interval == 0:
            progress_callback(frame_number, total_frames)

    cap.release()

    # Count unique objects per class
    detected_items = {}
    confidences_by_item = {}
    for item_name in tracked_objects.values():
        detected_items[item_name] = detected_items.get(item_name, 0) + 1

    for track_id, item_name in tracked_objects.items():
        track_scores = track_confidences.get(track_id, [])
        if track_scores:
            confidences_by_item.setdefault(item_name, []).append(max(track_scores))

    print(
        f"Detection complete. Tracked {len(tracked_objects)} unique objects "
        f"→ {detected_items}"
    )
    confidence_summary = _build_confidence_summary(confidences_by_item)

    if return_details:
        return detected_items, sample_frames, confidence_summary
    return detected_items, sample_frames


def deduplicate_detections(all_detections, total_frames, frame_interval):
    """
    Legacy fallback: estimate item counts by averaging detections per frame.
    No longer used by process_video (which uses ByteTrack instead) but kept
    for backward compatibility.
    """
    if not all_detections:
        return {}

    frames_processed = max(1, total_frames // frame_interval)
    raw_counts = Counter(all_detections)
    clean_counts = {}

    for item, raw_count in raw_counts.items():
        avg_per_frame = raw_count / frames_processed
        clean_counts[item] = max(1, round(avg_per_frame))

    return clean_counts


def process_image(image_path, confidence=0.25, imgsz=960, return_details=False):
    """Process a single image and return canonical inventory detections."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not open image: {image_path}")

    results = model(
        frame,
        verbose=False,
        conf=confidence,
        iou=0.5,
        imgsz=imgsz,
    )[0]
    detected = {}
    confidences_by_item = {}

    for detection in _inventory_detections(results, frame, confidence):
        item_name = detection["item"]
        detected[item_name] = detected.get(item_name, 0) + 1
        confidences_by_item.setdefault(item_name, []).append(detection["confidence"])
        _draw_detection(frame, detection["box"], item_name, detection["confidence"])

    if return_details:
        return detected, frame, _build_confidence_summary(confidences_by_item)
    return detected, frame
