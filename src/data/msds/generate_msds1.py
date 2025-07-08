#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

# Path setup
here = Path(__file__).parent
for p in [here] + list(here.parents):
    if (p / "src").is_dir():
        sys.path.insert(0, str(p / "src"))
        break
else:
    raise RuntimeError("Could not locate 'src' folder to add to PYTHONPATH")

from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Find project root and set data directory
def find_project_root() -> Path:
    """Find the project root (directory containing 'src' folder)."""
    current = Path(__file__).parent
    for p in [current] + list(current.parents):
        if (p / "src").is_dir():
            return p
    # Fallback to current working directory
    return Path.cwd()

# <repo-root>/data by default
project_root = find_project_root()
default_data_dir = project_root / "data"

# Image counts used by the original generator
NTRAIN, NTEST = 9 * 10**4, 10**4

# ───────────────────────── dataset availability ──────────────────────────

def get_root(data_dir: Path) -> Path:
    return data_dir / "msds1"

def _exists(root: Path) -> bool:
    """Rough-and-ready: look for both splits, at least one PNG, and the HDF5s."""
    return (
        (root / "train/0").is_dir()
        and any((root / "train/0").glob("*.png"))
        and (root / "test/0").is_dir()
        and any((root / "test/0").glob("*.png"))
        and (root / "train_labels.hdf5").is_file()
        and (root / "test_labels.hdf5").is_file()
    )

def generate(split: str, n_samples: int, base_dir: Path) -> None:
    """
    Generate synthetic MSDS1 dataset.
    This is a placeholder - implement your actual dataset generation logic here.
    """
    print(f"Generating {n_samples} {split} samples in {base_dir.absolute()}")
    
    import numpy as np
    from PIL import Image
    import h5py
    
    # Create directories
    split_dir = base_dir / split / "0"
    split_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {split_dir.absolute()}")
    
    # Generate simple synthetic images (replace with your actual generation logic)
    np.random.seed(42 if split == "train" else 123)
    
    labels = []
    for i in range(n_samples):
        # Generate a simple synthetic image (32x32 RGB)
        # Replace this with your actual MSDS1 generation algorithm
        img_data = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        
        # Add some structure to make it less random
        center_x, center_y = 16, 16
        y, x = np.ogrid[:32, :32]
        mask = (x - center_x)**2 + (y - center_y)**2 <= (8 + 4 * np.sin(i * 0.1))**2
        img_data[mask] = [255, 255, 255]  # White circle with varying radius
        
        # Save image
        img = Image.fromarray(img_data)
        img_path = split_dir / f"img_{i:06d}.png"
        img.save(img_path)
        
        # Generate a label (replace with your actual labeling logic)
        label = i % 10  # Simple label for demo
        labels.append(label)
        
        if i % 10000 == 0 and i > 0:
            print(f"Generated {i}/{n_samples} {split} images")
    
    # Save labels to HDF5
    labels_path = base_dir / f"{split}_labels.hdf5"
    with h5py.File(labels_path, "w") as f:
        f.create_dataset("labels", data=np.array(labels))
    
    print(f"Finished generating {n_samples} {split} images at {split_dir.absolute()}")
    print(f"Labels saved to {labels_path.absolute()}")

def _generate(root: Path) -> None:
    """Run the local generator to create msds1 under root."""
    try:
        root.mkdir(parents=True, exist_ok=True)
        # Call the generate function directly (no self-import needed)
        generate("train", NTRAIN, base_dir=root)
        generate("test", NTEST, base_dir=root)
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate msds1 dataset: {e}"
        ) from e

def _ensure(root: Path) -> None:
    """Guarantees the dataset exists or raises RuntimeError."""
    if _exists(root):
        print(f"MSDS1 dataset found at {root}")
        return
    print(f"MSDS1 dataset not found at {root}")
    print(f"Generating MSDS1 dataset at: {root.absolute()}")
    _generate(root)
    if not _exists(root):
        raise RuntimeError(f"Generating msds1 failed – dataset still missing at {root.absolute()}")

# ───────────────────────── public loader function ────────────────────────

def get_msds1_dataloaders(
    data_dir: Path | str = default_data_dir,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    • If data already present → just load it.
    • Else → call the generator once, then load.
    • Any generator failure is propagated.
    """
    data_dir = Path(data_dir)
    root = get_root(data_dir)
    
    print(f"Looking for MSDS1 dataset at: {root.absolute()}")
    print(f"Data directory: {data_dir.absolute()}")
    
    _ensure(root)
    
    train_tf = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    test_tf = transforms.ToTensor()
    
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "test", transform=test_tf)  # uses 'test' split
    test_ds = val_ds  # same underlying set
    
    print(f"Created datasets - Train: {len(train_ds)} samples, Test: {len(val_ds)} samples")
    
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    
    return train_loader, val_loader, test_loader

# For direct execution testing
if __name__ == "__main__":
    print("Testing MSDS1 dataset generation...")
    test_dir = Path("./test_data")
    loaders = get_msds1_dataloaders(test_dir, batch_size=32, num_workers=2)
    print(f"Successfully created dataloaders: {len(loaders)} loaders")
    
    # Test loading a batch
    train_loader = loaders[0]
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch[0].shape}, Labels shape: {batch[1].shape}")