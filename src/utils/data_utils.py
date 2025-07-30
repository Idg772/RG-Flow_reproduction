"""
Data loading utilities.
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

from data.msds.generate_msds1 import get_msds1_dataloaders
from data.msds.generate_msds2 import get_msds2_dataloaders
from data.celeba import get_celeba_dataloaders


def get_dataloaders(dataset: str, data_dir: Path, batch: int, workers: int, test_mode: bool = False):
    """
    Get dataloaders for the specified dataset.
    
    Args:
        dataset: Dataset name ("msds1", "msds2", or "celeba")
        data_dir: Data directory path
        batch: Batch size
        workers: Number of worker processes
        test_mode: If True, use only a small subset of data for quick testing
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Raises:
        ValueError: If dataset name is not recognized
    """
    if dataset == "msds1":
        loaders = get_msds1_dataloaders(data_dir, batch, workers)
    elif dataset == "msds2":
        loaders = get_msds2_dataloaders(data_dir, batch, workers)
    elif dataset == "celeba":
        loaders = get_celeba_dataloaders(data_dir, batch, workers)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")
    
    if test_mode:
        train_loader, val_loader, test_loader = loaders
        
        # Create small subsets for testing
        test_subset_size = 128  # Small subset for quick testing
        
        # Create subset of training data
        train_indices = torch.randperm(len(train_loader.dataset))[:test_subset_size]
        train_subset = Subset(train_loader.dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True, 
                                num_workers=workers, pin_memory=True)
        
        # Create subset of validation data
        val_indices = torch.randperm(len(val_loader.dataset))[:min(64, len(val_loader.dataset))]
        val_subset = Subset(val_loader.dataset, val_indices)
        val_loader = DataLoader(val_subset, batch_size=batch, shuffle=False, 
                              num_workers=workers, pin_memory=True)
        
        return train_loader, val_loader, test_loader
    
    return loaders