"""Create reproducible train/val/test splits from raw folders."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from real_estate_ml.constants import CLASSES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _collect_images(class_dir: Path) -> list[Path]:
    return [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def _safe_copy(files: list[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for src_path in files:
        shutil.copy2(src_path, target_dir / src_path.name)


def prepare_splits(
    raw_training_dir: Path,
    raw_validation_dir: Path,
    output_dir: Path,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> None:
    if abs((train_split + val_split + test_split) - 1.0) > 1e-6:
        raise ValueError("train_split + val_split + test_split must be 1.0")

    rng = random.Random(seed)

    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        training_class_dir = raw_training_dir / class_name
        validation_class_dir = raw_validation_dir / class_name
        all_images = []

        if training_class_dir.exists():
            all_images.extend(_collect_images(training_class_dir))
        if validation_class_dir.exists():
            all_images.extend(_collect_images(validation_class_dir))

        if not all_images:
            print(f"[WARN] No images found for class '{class_name}'.")
            continue

        rng.shuffle(all_images)
        total = len(all_images)
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)

        train_files = all_images[:train_end]
        val_files = all_images[train_end:val_end]
        test_files = all_images[val_end:]

        _safe_copy(train_files, output_dir / "train" / class_name)
        _safe_copy(val_files, output_dir / "val" / class_name)
        _safe_copy(test_files, output_dir / "test" / class_name)

        print(
            f"{class_name}: total={total}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reproducible train/val/test splits.")
    parser.add_argument("--raw-training-dir", default="data/raw/dataset/training")
    parser.add_argument("--raw-validation-dir", default="data/raw/dataset/validation")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_splits(
        raw_training_dir=Path(args.raw_training_dir),
        raw_validation_dir=Path(args.raw_validation_dir),
        output_dir=Path(args.output_dir),
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

