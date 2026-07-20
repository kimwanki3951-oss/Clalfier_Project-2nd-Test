from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import ManifestImageDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_params() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_default_transforms(img_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def create_dataloaders(
    train_transform=None,
    valid_transform=None,
    test_transform=None,
):
    params = load_params()

    seed = int(params["data"]["seed"])
    img_size = int(params["loader"]["img_size"])
    batch_size = int(params["loader"]["batch_size"])
    num_workers = int(params["loader"]["num_workers"])

    set_seed(seed)

    default_train_transform, default_eval_transform = build_default_transforms(
        img_size=img_size
    )

    train_transform = train_transform or default_train_transform
    valid_transform = valid_transform or default_eval_transform
    test_transform = test_transform or default_eval_transform

    train_dataset = ManifestImageDataset(
        manifest_path=SPLIT_DIR / "train.csv",
        project_root=PROJECT_ROOT,
        transform=train_transform,
    )

    class_to_idx = train_dataset.class_to_idx

    valid_dataset = ManifestImageDataset(
        manifest_path=SPLIT_DIR / "valid.csv",
        project_root=PROJECT_ROOT,
        transform=valid_transform,
        class_to_idx=class_to_idx,
    )

    test_dataset = ManifestImageDataset(
        manifest_path=SPLIT_DIR / "test.csv",
        project_root=PROJECT_ROOT,
        transform=test_transform,
        class_to_idx=class_to_idx,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train_ds": train_dataset,
        "valid_ds": valid_dataset,
        "test_ds": test_dataset,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "classes": train_dataset.classes,
        "num_classes": len(train_dataset.classes),
    }
