import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    mobilenet_v3_large,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights
)

class BottleDefectClassifier(nn.Module):
    """
    MobileNetV3 transfer learning model for bottle visual defect classification.
    
    Why MobileNetV3?
    1. Edge & Production Ready: Tailored for fast CPU/edge inference with low latency (5-15ms).
    2. High Efficiency: Utilizes Hard-Swish activations and Squeeze-and-Excitation attention modules.
    3. Strong Generalization: Fine-tuning pretrained ImageNet backbone provides rapid convergence
       even on small manufacturing datasets.
    """
    def __init__(self, num_classes: int = 2, variant: str = "small", pretrained: bool = True, dropout: float = 0.2):
        super(BottleDefectClassifier, self).__init__()
        self.variant = variant.lower()
        self.num_classes = num_classes

        if self.variant == "large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_large(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            last_channel = self.backbone.classifier[0].out_features
        else:
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            last_channel = self.backbone.classifier[0].out_features

        # Replace classification head with custom dropout and linear projection for 2 classes
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

def create_model(num_classes: int = 2, variant: str = "small", pretrained: bool = True, dropout: float = 0.2) -> BottleDefectClassifier:
    """
    Factory function to instantiate MobileNetV3 model.
    """
    return BottleDefectClassifier(
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained,
        dropout=dropout
    )
