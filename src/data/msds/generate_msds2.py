#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

default_data_dir = Path.cwd().parent.parent / "data"
_NTRAIN, _NTEST = 9 * 10**4, 10**4


def _root(data_dir: Path) -> Path:
    return data_dir / "msds2"


def _exists(root: Path) -> bool:
    return (
        (root / "train/0").is_dir()
        and any((root / "train/0").glob("*.png"))
        and (root / "test/0").is_dir()
        and any((root / "test/0").glob("*.png"))
        and (root / "train_labels.hdf5").is_file()
        and (root / "test_labels.hdf5").is_file()
    )


def _generate(root: Path) -> None:
    try:
        import generate_msds2 as gm2
    except ImportError as e:
        raise RuntimeError(
            "msds2 dataset is missing and generate_msds2.py could not be imported."
        ) from e

    root.mkdir(parents=True, exist_ok=True)
    gm2.generate("train", _NTRAIN, base_dir=root)
    gm2.generate("test",  _NTEST,  base_dir=root)


def _ensure(root: Path) -> None:
    if _exists(root):
        return
    _generate(root)
    if not _exists(root):
        raise RuntimeError("Generating msds2 failed – dataset still missing.")


def get_msds2_dataloaders(
    data_dir: Path | str = default_data_dir,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    root = _root(data_dir)
    _ensure(root)

    train_tf = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    test_tf = transforms.ToTensor()

    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(root / "test",  transform=test_tf)
    test_ds  = val_ds

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    return train_loader, val_loader, test_loader
