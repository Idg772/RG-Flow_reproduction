#!/usr/bin/env python3
# src/data/celebA.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# <repo-root>/data by default
default_data_dir = Path.cwd().parent.parent / "data"


def _load_celeba_once(root: Path, split: str, transform, try_download: bool):
    """
    Helper that wraps a single call to torchvision.datasets.CelebA.
    """
    return datasets.CelebA(
        root=root,
        split=split,
        transform=transform,
        download=try_download,
    )


def _load_or_download_celeba(root: Path, split: str, transform):
    """
    1) Attempt to load with download=False.
    2) If that raises the canonical 'Dataset not found or corrupted' RuntimeError,
       retry once with download=True.
    3) Propagate any exception from the second attempt.
    """
    try:
        return _load_celeba_once(root, split, transform, try_download=False)
    except RuntimeError as e:
        msg = str(e).lower()
        if "not found" in msg and "download=true" in msg:
            # dataset missing – one retry with download=True
            return _load_celeba_once(root, split, transform, try_download=True)
        raise  # some other runtime error – let it propagate


def get_celeba_dataloaders(
    data_dir: Path | str = default_data_dir,
    batch_size: int = 512,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train / val / test DataLoaders for CelebA.

    • If the dataset already exists in `data_dir`, it is loaded directly.  
    • Otherwise a download is attempted exactly once; any failure is raised.
    """
    data_dir = Path(data_dir)

    train_transform = transforms.Compose(
        [
            transforms.CenterCrop(148),
            transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(148),
            transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )

    train_ds = _load_or_download_celeba(data_dir, "train",  train_transform)
    val_ds   = _load_or_download_celeba(data_dir, "valid",  test_transform)
    test_ds  = _load_or_download_celeba(data_dir, "test",   test_transform)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
