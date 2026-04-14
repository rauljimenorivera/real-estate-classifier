"""Dataset and dataloader utilities."""

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from real_estate_ml.constants import CLASS_TO_IDX, CLASSES

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


class RealEstateDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_name in CLASSES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for pattern in IMAGE_EXTENSIONS:
                for img_path in class_dir.glob(pattern):
                    self.samples.append((img_path, CLASS_TO_IDX[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(split: str, image_size: int = 224):
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_dataloaders(data_dir: str, batch_size: int = 32, image_size: int = 224, num_workers: int = 4):
    dataloaders = {}
    for split in ["train", "val", "test"]:
        dataset = RealEstateDataset(
            root_dir=data_dir,
            split=split,
            transform=get_transforms(split, image_size),
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"{split}: {len(dataset)} images")
    return dataloaders

