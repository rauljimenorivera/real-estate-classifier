"""Project-level constants used across training and inference."""

CLASSES = [
    "Bedroom",
    "Coast",
    "Forest",
    "Highway",
    "Industrial",
    "Inside city",
    "Kitchen",
    "Living room",
    "Mountain",
    "Office",
    "Open country",
    "Store",
    "Street",
    "Suburb",
    "Tall building",
]

CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_IDX.items()}

