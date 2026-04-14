"""Inference helper shared by API and Streamlit."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from real_estate_ml.constants import IDX_TO_CLASS
from real_estate_ml.data.dataset import get_transforms
from real_estate_ml.models.classifier import build_model


class Predictor:
    def __init__(self, checkpoint_path: str | Path, backbone: str, num_classes: int, device: str = "cpu", image_size: int = 224):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.transform = get_transforms(split="test", image_size=image_size)
        self.model = build_model(backbone=backbone, num_classes=num_classes, pretrained=False)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image, top_k: int = 3):
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            values, indices = torch.topk(probs, k=top_k)

        return [
            {
                "class_name": IDX_TO_CLASS[index.item()],
                "probability": round(value.item(), 6),
            }
            for value, index in zip(values, indices)
        ]

