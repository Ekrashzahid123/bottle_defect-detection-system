import io
import os
import sys
import time
from typing import Union, Dict, Any
from PIL import Image
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CLASS_NAMES
from src.model import create_model

class BottleDefectPredictor:
    """
    Pure TensorFlow 2.x production inference engine for MobileNetV3 bottle defect detection.
    """
    def __init__(self, model_path: str = "models/best_model.keras"):
        self.model_path = model_path
        self.class_names = CLASS_NAMES
        self.img_size = 224
        self.model = None
        self.framework = "TensorFlow 2.x"
        
        gpus = tf.config.list_physical_devices("GPU")
        self.device = "GPU" if gpus else "CPU"
        
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            # Fallback initialization so API & UI can boot immediately
            self.model = create_model(num_classes=len(self.class_names))

    def _load_image(self, image_input: Union[str, bytes, Image.Image]) -> np.ndarray:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        image = image.resize((self.img_size, self.img_size))
        img_array = np.array(image, dtype=np.float32)
        return np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)

    def predict(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        img_batch = self._load_image(image_input)

        probs = self.model.predict(img_batch, verbose=0)[0]
        class_id = int(np.argmax(probs))
        class_name = self.class_names[class_id]
        confidence = float(probs[class_id])
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        probabilities_dict = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }

        return {
            "status": "success",
            "prediction": class_name,
            "class_id": class_id,
            "is_defective": bool(class_id == 1),
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities_dict.items()},
            "inference_time_ms": round(elapsed_ms, 2),
            "framework": self.framework,
            "device": self.device
        }
