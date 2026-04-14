"""CLI wrapper for reproducible train/val/test split creation."""

from pathlib import Path

from real_estate_ml.config import load_config
from real_estate_ml.data.prepare_splits import prepare_splits


def main():
    cfg = load_config("configs/base_config.yaml")
    prepare_splits(
        raw_training_dir=Path(cfg["data"]["raw_training_dir"]),
        raw_validation_dir=Path(cfg["data"]["raw_validation_dir"]),
        output_dir=Path(cfg["data"]["data_dir"]),
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        seed=cfg["training"]["seed"],
    )


if __name__ == "__main__":
    main()

