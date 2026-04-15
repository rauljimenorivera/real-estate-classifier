"""FastAPI inference service."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate_ml.inference.predictor import Predictor

app = FastAPI(title="Real Estate Classifier API", version="0.1.0")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/best_model.pth"))
DEVICE = os.getenv("DEVICE", "cuda")
BACKBONE = "efficientnet_b0"  # fallback (checkpoint may override)
NUM_CLASSES = 15  # fallback (checkpoint may override)
predictor: Predictor | None = None


@app.on_event("startup")
def startup_event():
    global predictor
    if MODEL_PATH.exists():
        predictor = Predictor(
            checkpoint_path=MODEL_PATH,
            backbone=BACKBONE,
            num_classes=NUM_CLASSES,
            device=DEVICE,
        )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Expected checkpoint at '{MODEL_PATH.as_posix()}'. Train model first.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Input file must be an image.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload.") from exc

    predictions = predictor.predict(image, top_k=3)
    return {"filename": file.filename, "predictions": predictions}

