"""Model builder for transfer learning classifiers."""

import timm
import torch.nn as nn


def build_model(backbone: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3, freeze_backbone: bool = False):
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, "get_classifier"):
            classifier = model.get_classifier()
            for param in classifier.parameters():
                param.requires_grad = True

    return model

