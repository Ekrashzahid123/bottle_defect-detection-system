import io
import os
import sys
import time
from typing import Union, Dict, Any
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CLASS_NAMES

class BottleDefectPredictor:
    """
    Production-grade inference engine supporting both TensorFlow (.keras) and PyTorch (.pth) backbones.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Auto-detect best model: prioritize .keras (TensorFlow) or .pth (PyTorch)
            if os.path.exists("models/best_model.keras"):
                model_path = "models/best_model.keras"
            elif os.path.exists("models/best_model.pth"):
                model_path = "models/best_model.pth"
            else:
                model_path = "models/best_model.keras"

        self.model_path = model_path
        self.class_names = CLASS_NAMES
        self.img_size = 224
        self.framework = "unknown"
        self.model = None
        self.device = "cpu"
        
        self._load_model()

    def _load_model(self):
        if self.model_path.endswith((".keras", ".h5")):
            self.framework = "tensorflow"
            import tensorflow as tf
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
            else:
                from src.tf_model import create_tf_mobilenet_v3
                self.model = create_tf_mobilenet_v3(num_classes=len(self.class_names))
            gpus = tf.config.list_physical_devices("GPU")
            self.device = "gpu" if gpus else "cpu"

        else:
            self.framework = "pytorch"
            import torch
            import torchvision.transforms as transforms
            from src.model import create_model

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pytorch_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                variant = checkpoint.get("variant", "small")
                self.model = create_model(num_classes=len(self.class_names), variant=variant, pretrained=False)
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model = create_model(num_classes=len(self.class_names), variant="small", pretrained=True)
                
            self.model.to(self.device)
            self.model.eval()

    def _load_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

    def predict(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        image = self._load_image(image_input)

        if self.framework == "tensorflow":
            import tensorflow as tf
            img_resized = image.resize((self.img_size, self.img_size))
            img_arr = np.array(img_resized, dtype=np.float32)
            img_batch = np.expand_dims(img_arr, axis=0)
            
            # Predict
            probs = self.model.predict(img_batch, verbose=0)[0]
            
        else:
            import torch
            tensor = self.pytorch_transforms(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

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
            "framework": self.framework
        }
