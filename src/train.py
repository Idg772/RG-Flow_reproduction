"""
Training script for RG-Flow models.

This script provides a command-line interface for training RG-Flow models
on various datasets with configurable parameters.
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Path setup - add src directory to Python path
here = Path(__file__).parent
for p in [here] + list(here.parents):
    if (p / "src").is_dir():
        sys.path.insert(0, str(p / "src"))
        break
else:
    raise RuntimeError("Could not locate 'src' folder to add to PYTHONPATH")

# --------------------------------------------------------------------------- #
# Local imports
from layers import build_rg_flow
from distributions import PriorDistribution
from utils import logit_transform, setup_logger, sample_and_save, get_dataloaders


def train(dataset: str, data_root: Path, epochs: int, batch_size: int, 
          lr: float, prior_name: str, out_root: Path, device: str, 
          workers: int = 8, img_samples: int = 16, test_mode: bool = False):
    """
    Main training function for RG-Flow models.
    
    Args:
        dataset: Dataset to train on ("msds1", "msds2", or "celeba")
        data_root: Root directory for dataset storage
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate for optimizer
        prior_name: Prior distribution type ("laplace" or "gaussian")
        out_root: Root directory for output artifacts
        device: Device to train on ("cpu" or "cuda")
        workers: Number of data loading workers
        img_samples: Number of images to sample per epoch
        test_mode: If True, use small subset of data for quick testing
    """
    # Set up logging and directories
    run_dir = out_root / f"{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "ckpt").mkdir(parents=True)
    (run_dir / "samples").mkdir()
    logger = setup_logger(run_dir / "logs")
    
    # Get dataloaders using the dataloader functions
    train_ld, val_ld, _ = get_dataloaders(dataset, data_root, batch_size, workers, test_mode)
    mode_str = " (TEST MODE - small subset)" if test_mode else ""
    logger.info("Dataset %s%s – %d training batches, 1 validation split",
                dataset, mode_str, len(train_ld))

    # Determine image geometry
    _ex = next(iter(train_ld))[0]
    nC, H, W = _ex.shape[1:]
    logger.info("Image dimensions: C=%d, H=%d, W=%d", nC, H, W)

    # Initialize model, prior, and optimizer
    model = build_rg_flow(depth=8, blocks_per_layer=4).to(device)
    prior = PriorDistribution(prior_name).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)

    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("MERA model %.2f M parameters; optimiser AdamW lr %.1e", n_param, lr)

    bpd_const = math.log(2) * nC * H * W

    # Training loop
    for ep in range(1, epochs + 1):
        t0, ep_loss = time.time(), 0.0
        model.train()

        for step, (x, _) in enumerate(train_ld, 1):
            x = x.to(device)
            x, ldj0 = logit_transform(x)
            z, ldj1 = model(x)
            logp = prior.log_prob(z)
            loss = -(logp + ldj0 + ldj1).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_loss += loss.item()

            if step % 50 == 0 or step == len(train_ld):
                bpd = (loss.item() + math.log(256.) * nC * H * W) / bpd_const
                logger.info("[ep %3d] batch %4d/%d  loss %.4f  bpd %.4f",
                            ep, step, len(train_ld), loss.item(), bpd)

        logger.info("Epoch %d finished in %.1fs  avg‑loss %.4f",
                    ep, time.time() - t0, ep_loss / len(train_ld))

        # Validation
        model.eval()
        with torch.no_grad():
            val_ll = 0.0
            for x, _ in val_ld:
                x = x.to(device)
                x, ldj0 = logit_transform(x)
                z, ldj1 = model(x)
                val_ll += (prior.log_prob(z) + ldj0 + ldj1).sum().item()
        val_bpd = (- val_ll / len(val_ld.dataset)
                   + math.log(256.) * nC * H * W) / bpd_const
        logger.info("Validation BPD: %.4f", val_bpd)

        # Sampling and checkpointing
        sample_and_save(model, prior, ep, (nC, H, W),
                        img_samples, device, run_dir / "samples", logger)
        ck = run_dir / "ckpt" / f"rgflow_ep{ep:03d}.pth"
        torch.save({"epoch": ep,
                    "model": model.state_dict(),
                    "optim": optim.state_dict()}, ck)
        logger.info("Checkpoint saved → %s", ck)

    logger.info("Training complete – artefacts in %s", run_dir)


def cli():
    """Command line interface for training RG-Flow models."""
    p = argparse.ArgumentParser(prog="train", description="Train RG‑Flow")
    p.add_argument("--dataset", choices=["msds1", "msds2", "celeba"], default="msds1",
                   help="Dataset to train on")
    p.add_argument("--data-root", type=str, default="data",
                   help="Root folder that will hold the datasets")
    p.add_argument("--epochs", type=int, default=60,
                   help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Training batch size")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--prior", choices=["laplace", "gaussian"], default="laplace",
                   help="Prior distribution type")
    p.add_argument("--out", type=str, default="runs",
                   help="Output directory for artifacts")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device to train on")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of data loading workers")
    p.add_argument("--img-samples", type=int, default=16,
                   help="Images sampled per epoch")
    p.add_argument("--test-mode", action="store_true",
                   help="Use small subset of data for quick testing")
    args = p.parse_args()

    train(dataset=args.dataset,
          data_root=Path(args.data_root),
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          prior_name=args.prior,
          out_root=Path(args.out),
          device=args.device,
          workers=args.workers,
          img_samples=args.img_samples,
          test_mode=args.test_mode)


if __name__ == "__main__":
    cli()