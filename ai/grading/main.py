"""FastAPI entry point for five-grade diabetic-retinopathy inference."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.append(os.fspath(PROJECT_ROOT))

from ai.grading.predictor import DRPredictor, load_predictor


app = FastAPI(
    title="Diabetic Retinopathy Grading API",
    description="Inference for five-grade ICDR diabetic-retinopathy classifiers.",
    version="2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEIGHTS_DIR = PROJECT_ROOT / "ai" / "weights"
DEFAULT_MODEL = WEIGHTS_DIR / "dr_grading_model.keras"
MODEL_PATH = Path(os.environ.get("DR_MODEL_PATH", os.fspath(DEFAULT_MODEL)))
MODEL_BACKEND = os.environ.get("DR_MODEL_BACKEND", "auto")
LEGACY_PREPROCESS_ENHANCE = os.environ.get("DR_PREPROCESS_ENHANCE", "0") == "1"

dr_predictor: DRPredictor | None = None
startup_error: str | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global dr_predictor, startup_error
    try:
        dr_predictor = load_predictor(
            MODEL_PATH,
            backend=MODEL_BACKEND,
            legacy_enhance=LEGACY_PREPROCESS_ENHANCE,
        )
        startup_error = None
        print(
            f"Loaded {dr_predictor.info.backend} grading model "
            f"{dr_predictor.info.model_version}"
        )
    except Exception as error:
        dr_predictor = None
        startup_error = str(error)
        print(f"Could not load DR grading model: {startup_error}")


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "DR Grading API is running; POST an image to /analyze."}


@app.get("/health")
def health() -> Dict[str, str]:
    if dr_predictor is None:
        raise HTTPException(status_code=503, detail=startup_error or "Model is not loaded")
    return {"status": "ready", "model_version": dr_predictor.info.model_version}


@app.get("/model-info")
def get_model_info() -> Dict[str, Any]:
    if dr_predictor is None:
        return {"error": startup_error or "Model is not loaded", "model_path": os.fspath(MODEL_PATH)}
    result = dr_predictor.info.to_dict()
    result["model_path"] = os.fspath(MODEL_PATH)
    if MODEL_PATH.is_file():
        result["size_MB"] = round(MODEL_PATH.stat().st_size / (1024 * 1024), 2)
    return result


@app.post("/analyze")
async def analyze_fundus(file: UploadFile = File(...)):
    if dr_predictor is None:
        raise HTTPException(status_code=503, detail=startup_error or "Model is not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        content = await file.read()
        encoded = np.frombuffer(content, dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Image is corrupt or unsupported")

        prediction = dr_predictor.predict(image_bgr)
        result = prediction.to_api_dict()
        succeeded, buffer = cv2.imencode(".png", prediction.preprocessed_bgr)
        if succeeded:
            preview = base64.b64encode(buffer).decode("ascii")
            result["preprocessed_preview_b64"] = f"data:image/png;base64,{preview}"
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"DR analysis failed: {error}") from error
