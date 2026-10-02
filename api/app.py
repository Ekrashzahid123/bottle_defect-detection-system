import os
import sys
import logging
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BottleDefectAPI")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predictor import BottleDefectPredictor

# Pydantic Response Schemas
class PredictionResponse(BaseModel):
    status: str = Field(..., example="success")
    prediction: str = Field(..., example="Defective")
    class_id: int = Field(..., example=1)
    is_defective: bool = Field(..., example=True)
    confidence: float = Field(..., example=0.9852)
    probabilities: Dict[str, float] = Field(..., example={"Normal": 0.0148, "Defective": 0.9852})
    inference_time_ms: float = Field(..., example=12.4)

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    device: str = Field(..., example="cpu")
    model_variant: str = Field(..., example="mobilenet_v3_small")

# Initialize FastAPI App
app = FastAPI(
    title="Bottle Visual Defect Detection API",
    description="Production-ready REST API for industrial bottle defect inspection (Normal vs. Defective)",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance
predictor: BottleDefectPredictor = None

@app.on_event("startup")
def load_predictor():
    global predictor
    model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
    try:
        predictor = BottleDefectPredictor(model_path=model_path)
        logger.info(f"Model successfully loaded from {model_path} onto {predictor.device}")
    except Exception as e:
        logger.warning(f"Could not load model at startup ({e}). Predictor will initialize once model is trained.")

@app.get("/", tags=["General"])
def root():
    return {
        "service": "Bottle Visual Defect Detection API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    global predictor
    if predictor is None:
        model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
        if os.path.exists(model_path):
            try:
                predictor = BottleDefectPredictor(model_path=model_path)
            except Exception:
                pass
                
    is_loaded = predictor is not None and predictor.model is not None
    device_str = str(predictor.device) if predictor else "none"
    variant_str = f"mobilenet_v3_{predictor.variant}" if predictor else "unknown"

    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "device": device_str,
        "model_variant": variant_str
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_bottle(file: UploadFile = File(...)):
    """
    Accepts an uploaded bottle image, runs MobileNetV3 inference, and returns defect prediction and confidence.
    """
    global predictor
    if predictor is None:
        model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
        if os.path.exists(model_path):
            predictor = BottleDefectPredictor(model_path=model_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not trained or checkpoint file is missing."
            )

    # Input Validation
    allowed_types = ["image/jpeg", "image/png", "image/bmp", "image/webp", "image/jpg"]
    if file.content_type not in allowed_types and not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file.content_type}'. Please upload a JPEG or PNG image."
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        logger.info(f"Processing inference request for file: {file.filename} ({len(contents)} bytes)")
        result = predictor.predict(contents)
        logger.info(f"Result for {file.filename}: {result['prediction']} ({result['confidence']:.2%}) in {result['inference_time_ms']}ms")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error for {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
