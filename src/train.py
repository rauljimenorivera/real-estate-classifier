"""Train transfer-learning model and log to Weights & Biases."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import wandb
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from real_estate_ml.config import load_config
from real_estate_ml.constants import CLASSES
from real_estate_ml.data.dataset import get_dataloaders
from real_estate_ml.models.classifier import build_model
from real_estate_ml.training.engine import run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument(
        "--wandb",
        default="offline",
        choices=["offline", "online", "disabled"],
        help="Weights & Biases mode. Use 'offline' to avoid login prompts.",
    )
    return parser.parse_args()


def resolve_device(cfg: dict) -> torch.device:
    requested = cfg["hardware"].get("device", "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = resolve_device(cfg)
    run = None
    if args.wandb != "disabled":
        mode = args.wandb
        if mode == "online" and not (os.getenv("WANDB_API_KEY") or os.getenv("WANDB_ACCESS_TOKEN")):
            print("W&B online requested but no API key found; falling back to offline.")
            mode = "offline"
        run = wandb.init(
            project=cfg["project_name"],
            entity=cfg.get("entity"),
            config=cfg,
            job_type="train",
            mode=mode,
        )

    dataloaders = get_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = build_model(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = None
    if cfg["training"]["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    patience = 0
    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model.pth"

    for epoch in range(cfg["training"]["epochs"]):
        train_metrics = run_epoch(model, dataloaders["train"], criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, dataloaders["val"], criterion, optimizer, device, train=False)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{cfg['training']['epochs']} | "
            f"train_loss={train_metrics.loss:.4f} train_macro_f1={train_metrics.macro_f1:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_macro_f1={val_metrics.macro_f1:.4f} | "
            f"lr={lr:.2e}"
        )

        if run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics.loss,
                    "train/macro_f1": train_metrics.macro_f1,
                    "val/loss": val_metrics.loss,
                    "val/macro_f1": val_metrics.macro_f1,
                    "lr": lr,
                }
            )

        for class_name in CLASSES:
            class_report = val_metrics.report.get(class_name, {})
            if class_report and run is not None:
                wandb.log(
                    {
                        f"val/{class_name}/precision": class_report.get("precision", 0.0),
                        f"val/{class_name}/recall": class_report.get("recall", 0.0),
                        f"val/{class_name}/f1-score": class_report.get("f1-score", 0.0),
                    }
                )

        if val_metrics.macro_f1 > best_macro_f1:
            best_macro_f1 = val_metrics.macro_f1
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": cfg["model"]["backbone"],
                    "num_classes": cfg["data"]["num_classes"],
                    "classes": CLASSES,
                },
                best_model_path,
            )
        else:
            patience += 1

        if patience >= cfg["training"]["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, dataloaders["test"], criterion, optimizer=None, device=device, train=False)
    if run is not None:
        wandb.log(
            {
                "test/loss": test_metrics.loss,
                "test/macro_f1": test_metrics.macro_f1,
            }
        )
    cm_plot = ConfusionMatrixDisplay(test_metrics.confusion_matrix, display_labels=CLASSES).plot(xticks_rotation=45)
    if run is not None:
        wandb.log({"test/confusion_matrix": wandb.Image(cm_plot.figure_)})

    if run is not None:
        artifact = wandb.Artifact("best-model", type="model")
        artifact.add_file(str(best_model_path))
        run.log_artifact(artifact)
        run.finish()
    print(f"Training finished. Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()

