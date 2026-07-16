"""
Jasmine Flower Detection — FastAPI backend
--------------------------------------------
Wraps the existing YOLOv7 model (previously used inside the Streamlit app)
behind a REST API that the React Native frontend can call.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health   -> simple liveness check
    POST /predict  -> multipart form upload, field name "image"
                       returns detections + base64 annotated image
"""

import base64
import io
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from model import ModelWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jasmine-api")

app = FastAPI(
    title="Jasmine Flower Detection API",
    description="YOLOv7-based flower/bud detection service",
    version="1.0.0",
)

# Allow the mobile app (or Expo dev client) to call this API.
# Lock this down to your actual app's origin(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup and reused across requests (mirrors the
# @st.cache_resource behaviour from the Streamlit version).
model_wrapper = ModelWrapper(
    weights_path="trained_weights/yolov7_best_v3.pt",
    yolov7_repo_path="yolov7",
)


@app.on_event("startup")
def startup_event():
    logger.info("Loading YOLOv7 model...")
    model_wrapper.load()
    logger.info("Model loaded successfully.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_wrapper.is_loaded()}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        raw_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        pil_image = pil_image.resize((640, 640))
        img_array = np.array(pil_image)
    except Exception as exc:
        logger.exception("Failed to read uploaded image")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    try:
        detections, annotated_img_array = model_wrapper.run_inference(img_array)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    annotated_b64 = _encode_image_to_base64(annotated_img_array)

    return JSONResponse(
        {
            "detections": detections,
            "annotated_image_base64": annotated_b64,
        }
    )


def _encode_image_to_base64(img_array: np.ndarray) -> str:
    """Convert a numpy image array (as returned by results.render()) to a
    base64-encoded JPEG string the frontend can render directly."""
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
