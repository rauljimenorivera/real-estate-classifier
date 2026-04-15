"""Training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from real_estate_ml.constants import CLASSES


@dataclass
class EpochResult:
    loss: float
    macro_f1: float
    report: dict
    confusion_matrix: np.ndarray


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    train: bool,
    *,
    mixed_precision: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            use_amp = bool(mixed_precision and device.type == "cuda")
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                if use_amp:
                    if scaler is None:
                        raise ValueError("mixed_precision=True requires a GradScaler instance.")
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / max(1, len(loader.dataset))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))

    return EpochResult(loss=avg_loss, macro_f1=macro_f1, report=report, confusion_matrix=cm)

