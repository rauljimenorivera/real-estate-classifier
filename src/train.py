"""Train transfer-learning model and log to Weights & Biases."""

from __future__ import annotations

import argparse
import os
import json
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


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values. Repeatable. Example: --set model.backbone=convnext_large --set training.learning_rate=5e-5",
    )
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


def _coerce_scalar(value: str):
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if v.lower() in {"none", "null"}:
        return None
    try:
        if any(ch in v for ch in [".", "e", "E"]):
            return float(v)
        return int(v)
    except ValueError:
        return v


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set '{item}'. Use key.path=value")
        key_path, raw = item.split("=", 1)
        keys = [k for k in key_path.strip().split(".") if k]
        if not keys:
            raise ValueError(f"Invalid --set '{item}'. Empty key path.")
        cur = cfg
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = _coerce_scalar(raw)
    return cfg


def apply_wandb_sweep_overrides(cfg: dict, wb_cfg) -> dict:
    """If running under a W&B sweep, apply common hyperparams from wandb.config.

    We do this to avoid platform-specific CLI templating issues (e.g. ${var} not expanding on Windows).
    """
    if wb_cfg is None:
        return cfg

    mapping = {
        "backbone": ("model", "backbone"),
        "freeze_backbone": ("model", "freeze_backbone"),
        "dropout": ("model", "dropout"),
        "lr": ("training", "learning_rate"),
        "learning_rate": ("training", "learning_rate"),
        "weight_decay": ("training", "weight_decay"),
        "epochs": ("training", "epochs"),
        "batch_size": ("data", "batch_size"),
        "image_size": ("data", "image_size"),
        "num_workers": ("data", "num_workers"),
    }

    for src_key, dst_path in mapping.items():
        if src_key not in wb_cfg:
            continue
        value = wb_cfg.get(src_key)
        cur = cfg
        for k in dst_path[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[dst_path[-1]] = value

    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)

    device = resolve_device(cfg)
    run = None
    if args.wandb != "disabled":
        mode = args.wandb
        run = wandb.init(
            project=cfg["project_name"],
            entity=cfg.get("entity"),
            config=cfg,
            job_type="train",
            mode=mode,
        )
        # If this run is launched by a sweep agent, sweep parameters live in wandb.config.
        # Apply them to our nested cfg dict so the rest of the script uses them.
        cfg = apply_wandb_sweep_overrides(cfg, wandb.config)

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
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {cfg['model']['backbone']} | total_params={total_params:,} | trainable_params={trainable_params:,}")
    if run is not None:
        wandb.summary["model/total_params"] = total_params
        wandb.summary["model/trainable_params"] = trainable_params

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = None
    if cfg["training"]["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    criterion = nn.CrossEntropyLoss()
    mixed_precision = bool(cfg.get("hardware", {}).get("mixed_precision", False))
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    best_macro_f1 = -1.0
    patience = 0
    base_save_dir = Path(cfg["training"]["save_dir"])
    base_save_dir.mkdir(parents=True, exist_ok=True)
    run_id = (getattr(run, "id", None) or getattr(wandb, "run", None) and wandb.run and wandb.run.id) or "local"
    run_save_dir = base_save_dir / "runs" / str(run_id)
    run_save_dir.mkdir(parents=True, exist_ok=True)

    # Per-run best checkpoint (never overwritten by other runs)
    best_model_path = run_save_dir / "best_model.pth"
    # Global best pointer for the whole project (optional, only updated if better)
    global_best_path = base_save_dir / "best_model.pth"
    global_best_meta = base_save_dir / "best_model.json"

    try:
        for epoch in range(cfg["training"]["epochs"]):
            train_metrics = run_epoch(
                model,
                dataloaders["train"],
                criterion,
                optimizer,
                device,
                train=True,
                mixed_precision=mixed_precision,
                scaler=scaler,
            )
            val_metrics = run_epoch(
                model,
                dataloaders["val"],
                criterion,
                optimizer,
                device,
                train=False,
                mixed_precision=mixed_precision,
            )

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
                payload = {
                    "model_state_dict": model.state_dict(),
                    "backbone": cfg["model"]["backbone"],
                    "num_classes": cfg["data"]["num_classes"],
                    "classes": CLASSES,
                    "best_val_macro_f1": float(best_macro_f1),
                    "run_id": str(run_id),
                }
                torch.save(
                    {
                        **payload,
                    },
                    best_model_path,
                )

                # Update global best only if this run beats the previous global best
                prev_best = None
                if global_best_meta.exists():
                    try:
                        prev_best = json.loads(global_best_meta.read_text(encoding="utf-8")).get("best_val_macro_f1")
                    except Exception:
                        prev_best = None
                if prev_best is None or float(best_macro_f1) > float(prev_best):
                    torch.save(payload, global_best_path)
                    global_best_meta.write_text(
                        json.dumps(
                            {
                                "best_val_macro_f1": float(best_macro_f1),
                                "run_id": str(run_id),
                                "checkpoint": global_best_path.as_posix(),
                                "source_checkpoint": best_model_path.as_posix(),
                                "backbone": cfg["model"]["backbone"],
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            else:
                patience += 1

            # Sweep agent can request stop; exit gracefully to avoid noisy stack traces
            if run is not None and getattr(run, "should_stop", False):
                print("STOP REASON: W&B sweep (early_terminate) requested stop. This was NOT a manual Ctrl+C.")
                wandb.summary["stopped_by_sweep"] = True
                break

            if patience >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break
    except KeyboardInterrupt:
        # On Windows, sweep stop can surface as KeyboardInterrupt (e.g. while spawning DataLoader workers).
        print("STOP REASON: KeyboardInterrupt (very likely W&B sweep stop, not you). Ending run gracefully.")
        if run is not None:
            wandb.summary["stopped_by_sweep"] = True
            run.finish(exit_code=0)
        return

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(
        model,
        dataloaders["test"],
        criterion,
        optimizer=None,
        device=device,
        train=False,
        mixed_precision=mixed_precision,
    )
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
    print(f"Training finished. Run-best model saved at: {best_model_path}")
    if global_best_path.exists():
        print(f"Global-best pointer (artifacts) at: {global_best_path}")


if __name__ == "__main__":
    main()

