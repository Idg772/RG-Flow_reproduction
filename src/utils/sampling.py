"""
Image sampling and saving utilities.
"""
import math
from pathlib import Path
from typing import Tuple
import torch
import torchvision

from .transforms import logit_transform


def sample_and_save(model, prior, epoch: int, img_dims: Tuple[int, int, int],
                   n_samples: int, device, out_dir: Path, log):
    """
    Generate and save sample images from the model.
    
    Args:
        model: Trained model for generation
        prior: Prior distribution for sampling
        epoch: Current epoch number
        img_dims: Image dimensions (channels, height, width)
        n_samples: Number of samples to generate
        device: Device to run sampling on
        out_dir: Output directory for saving images
        log: Logger instance
    """
    nC, H, W = img_dims
    with torch.no_grad():
        z = prior.sample((n_samples, nC, H, W)).to(device)
        xl, _ = model.inverse(z)
        xp, _ = logit_transform(xl, inverse=True)
        grid = torchvision.utils.make_grid(xp.clamp(0, 1).cpu(),
                                           nrow=int(math.sqrt(n_samples)))
        p = out_dir / f"epoch{epoch:03d}.png"
        torchvision.utils.save_image(grid, p)
    log.info("Saved samples → %s", p)