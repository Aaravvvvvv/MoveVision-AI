import base64
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from detector import MODEL_PATH, process_image
from estimator import summarise_items
from quote import calculate_quote, get_distance


app = FastAPI(
    title="MoveVision AI API",
    description="Computer vision household inventory and relocation estimate API.",
    version="1.0.0",
)


class EstimateRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPG/PNG image")
    origin_city: str = "Delhi"
    destination_city: str = "Mumbai"
    distance_km: Optional[int] = None
    confidence: float = Field(0.25, ge=0.05, le=0.90)
    imgsz: int = Field(960, ge=320, le=1280)
    floor_number: int = Field(0, ge=0)
    has_lift: bool = True
    extra_boxes: int = Field(0, ge=0)
    box_type: str = "light"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_present": Path(MODEL_PATH).exists(),
    }


@app.post("/estimate")
def estimate(payload: EstimateRequest):
    try:
        image_bytes = base64.b64decode(payload.image_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc

    suffix = ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        image_path = Path(tmp.name)

    try:
        detected_items, _frame, confidence_summary = process_image(
            str(image_path),
            confidence=payload.confidence,
            imgsz=payload.imgsz,
            return_details=True,
        )
        summary = summarise_items(detected_items)
        distance = payload.distance_km or get_distance(
            payload.origin_city,
            payload.destination_city,
        )
        quote = calculate_quote(
            summary,
            payload.origin_city,
            payload.destination_city,
            int(distance),
            floor_number=payload.floor_number,
            has_lift=payload.has_lift,
            extra_boxes=payload.extra_boxes,
            box_type=payload.box_type,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Estimate failed: {exc}") from exc
    finally:
        image_path.unlink(missing_ok=True)

    return {
        "detected_items": detected_items,
        "confidence_summary": confidence_summary,
        "inventory_summary": summary,
        "quote": quote,
    }
