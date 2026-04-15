"""Inference helper shared by API and Streamlit."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from real_estate_ml.data.dataset import get_transforms
from real_estate_ml.models.classifier import build_model


class Predictor:
    def __init__(self, checkpoint_path: str | Path, backbone: str, num_classes: int, device: str = "cpu", image_size: int = 224):
        requested_device = device
        self.device = torch.device(requested_device if (requested_device == "cpu" or torch.cuda.is_available()) else "cpu")

        state = torch.load(checkpoint_path, map_location=self.device)
        ckpt_backbone = state.get("backbone", backbone)
        ckpt_num_classes = int(state.get("num_classes", num_classes))
        self.classes = list(state.get("classes") or [])
        if not self.classes:
            # Fallback to numeric labels if checkpoint has no class names
            self.classes = [str(i) for i in range(ckpt_num_classes)]

        self.transform = get_transforms(split="test", image_size=image_size)
        self.model = build_model(backbone=ckpt_backbone, num_classes=ckpt_num_classes, pretrained=False)
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
                "class_name": self.classes[index.item()] if 0 <= index.item() < len(self.classes) else str(index.item()),
                "probability": round(value.item(), 6),
            }
            for value, index in zip(values, indices)
        ]

