import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CORRECTIONS_PATH = Path("corrections.jsonl")


def _clean_value(value):
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "nan", "null"} else text


def _clean_count(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _row_at(frame, index):
    if index >= len(frame):
        return None
    row = frame.iloc[index]
    return {
        "item": _clean_value(row.get("Item", "")).lower(),
        "count": _clean_count(row.get("Count", 0)),
        "confidence": _clean_value(row.get("Confidence", "")),
        "review_status": _clean_value(row.get("Review Status", "")),
    }


def _event_id(event):
    payload = json.dumps(event, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_correction(event, output_path=DEFAULT_CORRECTIONS_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def log_inventory_corrections(
    original_frame,
    corrected_frame,
    room,
    image_filename,
    seen_event_ids=None,
    output_path=DEFAULT_CORRECTIONS_PATH,
):
    """
    Compare original detector rows with edited Streamlit rows and save changes.

    Each JSONL record captures the original detection, corrected value, room,
    source image/video filename, confidence metadata, and timestamp.
    """
    seen_event_ids = seen_event_ids if seen_event_ids is not None else set()
    logged = 0
    max_rows = max(len(original_frame), len(corrected_frame))

    for index in range(max_rows):
        original = _row_at(original_frame, index)
        corrected = _row_at(corrected_frame, index)

        if original == corrected:
            continue
        if not original and not corrected:
            continue
        if original and corrected:
            item_changed = original["item"] != corrected["item"]
            count_changed = original["count"] != corrected["count"]
            if not item_changed and not count_changed:
                continue
            change_type = []
            if item_changed:
                change_type.append("item")
            if count_changed:
                change_type.append("count")
        elif original and not corrected:
            change_type = ["deleted"]
        else:
            change_type = ["added"]

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "room": room,
            "image_filename": image_filename,
            "row_index": index,
            "change_type": change_type,
            "original_detection": original,
            "corrected_value": corrected,
        }

        event_key = _event_id({
            key: value
            for key, value in event.items()
            if key != "timestamp"
        })
        if event_key in seen_event_ids:
            continue

        event["event_id"] = event_key
        append_correction(event, output_path)
        seen_event_ids.add(event_key)
        logged += 1

    return logged
