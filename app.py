# Revised app.py — loads model at startup and exposes model-info and health endpoints.
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import status
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import time
import logging

logger = logging.getLogger("uvicorn.error")

MODEL_PATH = os.environ.get("MODEL_PATH", "energy_model.pkl")

model = None
model_loaded = False
model_load_error = None
model_load_time = None
model_size_bytes = None

def load_model(path: str):
    global model, model_loaded, model_load_error, model_load_time, model_size_bytes
    start = time.time()
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        model = joblib.load(path)
        model_loaded = True
        model_load_time = time.time() - start
        model_size_bytes = os.path.getsize(path)
        logger.info(f"Model loaded from {path} in {model_load_time:.2f}s ({model_size_bytes} bytes)")
    except Exception as e:
        model_loaded = False
        model_load_error = str(e)
        logger.exception("Failed to load model")
        raise

# ==== fastapi app ====
app = FastAPI(title="Kaggle Sklearn Model API (safe startup)")

# allow calls from Rork (adjust allow_origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== load model at startup with clear errors ====
@app.on_event("startup")
def startup_event():
    try:
        load_model(MODEL_PATH)
    except Exception:
        # On startup failure we log the error — the process continues so you can inspect /model-info
        logger.error("Model failed to load at startup. Check /model-info for details.")

# ---- define input schema ----
class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="One row of numeric features in expected order")
    feature_names: list[str] | None = None

class PredictResponse(BaseModel):
    prediction: float

@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {
        "model_loaded": model_loaded,
        "model_load_time": model_load_time,
        "model_size_bytes": model_size_bytes,
        "model_load_error": model_load_error,
        "model_path": MODEL_PATH,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        if not model_loaded:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {model_load_error}")

        if req.feature_names:
            X = pd.DataFrame([req.features], columns=req.feature_names)
        else:
            raise HTTPException(
                status_code=400,
                detail="Missing feature_names. This model needs column names to match training data."
            )
        y_pred = model.predict(X)
        pred = y_pred[0]
        if hasattr(pred, "item"):
            pred = pred.item()
        return PredictResponse(prediction=pred)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=400, detail=str(e))