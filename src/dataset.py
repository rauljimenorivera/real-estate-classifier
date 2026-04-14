"""Backward-compatible dataset exports.

Prefer importing from `real_estate_ml.data.dataset`.
"""

from real_estate_ml.constants import CLASS_TO_IDX, CLASSES
from real_estate_ml.data.dataset import RealEstateDataset, get_dataloaders, get_transforms

__all__ = [
    "CLASSES",
    "CLASS_TO_IDX",
    "RealEstateDataset",
    "get_transforms",
    "get_dataloaders",
]