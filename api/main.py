"""FastAPI inference service."""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
import wandb

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate_ml.inference.predictor import Predictor

app = FastAPI(title="Real Estate Classifier API", version="0.1.0")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/best_model.pth"))
MODEL_ARTIFACT = os.getenv("MODEL_ARTIFACT", "").strip()
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "202529987-universidad-pontificia-comillas").strip()
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "real-estate-classifier").strip()
DEVICE = os.getenv("DEVICE", "cuda")
BACKBONE = "efficientnet_b0"  # fallback (checkpoint may override)
NUM_CLASSES = 15  # fallback (checkpoint may override)
predictor: Predictor | None = None
loaded_model_source: str | None = None


@dataclass
class ModelSelection:
    checkpoint_path: Path
    source: str


class LoadModelRequest(BaseModel):
    model_path: str | None = Field(default=None, description="Local checkpoint .pth path.")
    artifact_ref: str | None = Field(
        default=None,
        description="W&B artifact reference, for example entity/project/best-model:v12.",
    )


def _download_artifact_checkpoint(artifact_ref: str) -> Path:
    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    download_dir = Path(artifact.download(root="artifacts/wandb_models"))
    checkpoints = sorted(download_dir.rglob("*.pth"))
    if not checkpoints:
        raise RuntimeError(f"Artifact '{artifact_ref}' does not contain any .pth checkpoint.")
    return checkpoints[0]


def _list_model_artifacts(entity: str, project: str, limit: int = 100) -> list[str]:
    api = wandb.Api()
    path = f"{entity}/{project}"
    collections = api.artifact_type("model", path).collections()
    refs: list[str] = []
    for collection in collections:
        try:
            versions = collection.versions()
        except Exception:
            continue
        for version in versions:
            refs.append(f"{entity}/{project}/{collection.name}:{version.version}")
            if len(refs) >= limit:
                return refs
    return refs


def _select_model(model_path: str | None = None, artifact_ref: str | None = None) -> ModelSelection:
    if artifact_ref:
        checkpoint = _download_artifact_checkpoint(artifact_ref.strip())
        return ModelSelection(checkpoint_path=checkpoint, source=f"wandb:{artifact_ref.strip()}")

    selected_path = Path((model_path or str(MODEL_PATH)).strip())
    if not selected_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{selected_path.as_posix()}'.")
    return ModelSelection(checkpoint_path=selected_path, source=f"local:{selected_path.as_posix()}")


def _load_predictor(selection: ModelSelection) -> None:
    global predictor, loaded_model_source
    predictor = Predictor(
        checkpoint_path=selection.checkpoint_path,
        backbone=BACKBONE,
        num_classes=NUM_CLASSES,
        device=DEVICE,
    )
    loaded_model_source = selection.source


@app.on_event("startup")
def startup_event():
    try:
        selection = _select_model(artifact_ref=MODEL_ARTIFACT) if MODEL_ARTIFACT else _select_model()
        _load_predictor(selection)
    except Exception:
        # Service keeps running and can load model later via /load-model
        return


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None, "model_source": loaded_model_source}


@app.post("/load-model")
def load_model(payload: LoadModelRequest):
    try:
        if payload.model_path and payload.artifact_ref:
            raise HTTPException(status_code=400, detail="Provide either model_path or artifact_ref, not both.")
        selection = _select_model(model_path=payload.model_path, artifact_ref=payload.artifact_ref)
        _load_predictor(selection)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to load model: {exc}") from exc
    return {"status": "ok", "model_loaded": True, "model_source": loaded_model_source}


@app.get("/models")
def list_models(entity: str | None = None, project: str | None = None, limit: int = 100):
    resolved_entity = (entity or WANDB_ENTITY).strip()
    resolved_project = (project or WANDB_PROJECT).strip()
    if not resolved_entity or not resolved_project:
        raise HTTPException(
            status_code=400,
            detail="Missing W&B context. Provide entity/project in query params or set WANDB_ENTITY and WANDB_PROJECT.",
        )
    if limit <= 0 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500.")
    try:
        models = _list_model_artifacts(resolved_entity, resolved_project, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to list W&B models: {exc}") from exc
    return {
        "entity": resolved_entity,
        "project": resolved_project,
        "count": len(models),
        "models": models,
    }


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

