from __future__ import annotations
import argparse, logging, math, os, time, sys
from datetime import datetime
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

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
from data.msds.generate_msds1 import get_msds1_dataloaders
from data.msds.generate_msds2 import get_msds2_dataloaders
from data.celeba import get_celeba_dataloaders

# --------------------------------------------------------------------------- #
# Add your utility functions here (logit_transform, sample_and_save, etc.)

def logit_transform(x, *, c: float = 0.9, inverse: bool = False):
    import math, torch
    if inverse:                          # logits → [0,1]
        logx = x
        pre = math.log(c) - math.log1p(-c)
        ldj = (F.softplus(logx) + F.softplus(-logx) -
               F.softplus(-pre)).flatten(1).sum(1)
        x = torch.sigmoid(logx)
        x = (x * 2 - 1) / c
        x = (x + 1) / 2
        return x, ldj

    # forward: [0,1] → logits
    x = (x * 255. + torch.rand_like(x)) / 256.
    x = (x * 2 - 1) * c
    x = (x + 1) / 2
    x = x.clamp(1e-6, 1 - 1e-6)
    logx = torch.log(x) - torch.log1p(-x)
    pre  = math.log(c) - math.log1p(-c)
    ldj  = (F.softplus(logx) + F.softplus(-logx) -
            F.softplus(-pre)).flatten(1).sum(1)
    return logx, ldj

def get_dataloaders(dataset: str, data_dir: Path, batch: int, workers: int):
    """Get dataloaders for the specified dataset"""
    if dataset == "msds1":
        return get_msds1_dataloaders(data_dir, batch, workers)
    elif dataset == "msds2":
        return get_msds2_dataloaders(data_dir, batch, workers)
    elif dataset == "celeba":
        return get_celeba_dataloaders(data_dir, batch, workers)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")

def setup_logger(out: Path, name: str = "rgflow"):
    """Set up logging"""
    out.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fmt  = "%(asctime)s  %(levelname)-8s  %(message)s"
    date = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_h = logging.FileHandler(out / f"{name}_{ts}.log")
    file_h.setFormatter(logging.Formatter(fmt, date))
    cons_h = logging.StreamHandler()
    cons_h.setFormatter(logging.Formatter(fmt, date))

    if not logger.handlers:
        logger.addHandler(file_h)
        logger.addHandler(cons_h)
    return logger

def sample_and_save(model, prior, epoch: int, img_dims: Tuple[int, int, int],
                   n_samples: int, device, out_dir: Path, log):
    """Generate and save samples"""
    nC, H, W = img_dims
    with torch.no_grad():
        z  = prior.sample((n_samples, nC, H, W)).to(device)
        xl, _ = model.inverse(z)
        xp, _ = logit_transform(xl, inverse=True)
        grid = torchvision.utils.make_grid(xp.clamp(0, 1).cpu(),
                                           nrow=int(math.sqrt(n_samples)))
        p = out_dir / f"epoch{epoch:03d}.png"
        torchvision.utils.save_image(grid, p)
    log.info("Saved samples → %s", p)

def train(dataset: str, data_root: Path, epochs: int, batch_size: int, 
          lr: float, prior_name: str, out_root: Path, device: str, 
          workers: int = 8, img_samples: int = 16):
    """Main training function"""
    
    # Set up logging and directories
    run_dir = out_root / f"{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "ckpt").mkdir(parents=True)
    (run_dir / "samples").mkdir()
    logger = setup_logger(run_dir / "logs")
    
    # Get dataloaders using the dataloader functions
    train_ld, val_ld, _ = get_dataloaders(dataset, data_root, batch_size, workers)
    logger.info("Dataset %s – %d training batches, 1 validation split",
                dataset, len(train_ld))

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
            logp    = prior.log_prob(z)
            loss    = -(logp + ldj0 + ldj1).mean()

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
        val_bpd = ( - val_ll / len(val_ld.dataset)
                    + math.log(256.) * nC * H * W ) / bpd_const
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
    """Command line interface"""
    p = argparse.ArgumentParser(prog="train", description="Train RG‑Flow")
    p.add_argument("--dataset", choices=["msds1", "msds2", "celeba"], default="msds1")
    p.add_argument("--data-root", type=str, default="data",
                   help="Root folder that will hold the datasets")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--prior", choices=["laplace", "gaussian"], default="laplace")
    p.add_argument("--out", type=str, default="runs")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--img-samples", type=int, default=16,
                   help="Images sampled per epoch")
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
          img_samples=args.img_samples)

if __name__ == "__main__":
    cli()


# from __future__ import annotations
# import argparse, logging, math, os, time, sys
# from datetime import datetime
# from pathlib import Path
# from typing import Tuple
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision
# from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor

# # --------------------------------------------------------------------------- #
# # Path setup - add src directory to Python path
# here = Path(__file__).parent
# for p in [here] + list(here.parents):
#     if (p / "src").is_dir():
#         sys.path.insert(0, str(p / "src"))
#         break
# else:
#     raise RuntimeError("Could not locate 'src' folder to add to PYTHONPATH")

# # --------------------------------------------------------------------------- #
# # local imports – fixed based on your file structure
# from layers import build_rg_flow
# from distributions import PriorDistribution  # Fixed: removed .prior
# from data.msds.generate_msds1 import get_msds1_dataloaders  # Fixed: use the actual function
# from data.msds.generate_msds2 import get_msds2_dataloaders
# from data.celeba import get_celeba_dataloaders  # Fixed: match your file structure

# # --------------------------------------------------------------------------- #

# # ════════════════════════════════════════════════════════════════════════════ #
# #                               ––  Logging  ––                               #
# # ════════════════════════════════════════════════════════════════════════════ #
# def setup_logger(log_dir: Path, name: str = "rgflow") -> logging.Logger:
#     """
#     Return a logger that writes *both* to console and to
#     `<log_dir>/<name>_<timestamp>.log`.
#     """
#     log_dir.mkdir(exist_ok=True, parents=True)
#     stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     log_file = log_dir / f"{name}_{stamp}.log"

#     fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
#     datefmt = "%Y-%m-%d %H:%M:%S"

#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
#     logger.propagate = False         # do not duplicate to root

#     # ‑‑ file
#     fh = logging.FileHandler(log_file)
#     fh.setFormatter(logging.Formatter(fmt, datefmt))
#     # ‑‑ console
#     ch = logging.StreamHandler()
#     ch.setFormatter(logging.Formatter(fmt, datefmt))

#     if not logger.handlers:          # idempotent
#         logger.addHandler(fh)
#         logger.addHandler(ch)

#     logger.info("Logger initialised. Saving to %s", log_file)
#     return logger


# # ════════════════════════════════════════════════════════════════════════════ #
# #                         ––  Logit  ⇄  Pixel space  ––                        #
# # ════════════════════════════════════════════════════════════════════════════ #
# def logit_transform(
#     x: torch.Tensor,
#     *,
#     constraint: float = 0.9,
#     inverse: bool = False,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Bidirectional logit‑transform with dequantisation and LDJ computation.
#     """
#     if inverse:
#         logit_x = x
#         pre_logit_scale = math.log(constraint) - math.log(1 - constraint)
#         ldj = (F.softplus(logit_x) + F.softplus(-logit_x)
#                - F.softplus(-pre_logit_scale)).flatten(1).sum(1)
#         x = torch.sigmoid(logit_x)           # [0.05, 0.95]
#         x = (x * 2 - 1) / constraint         # [-1,1]
#         x = (x + 1) / 2                      # [0,1]
#         return x, ldj

#     # forward direction
#     x = (x * 255. + torch.rand_like(x)) / 256.
#     x = (x * 2 - 1) * constraint
#     x = (x + 1) / 2
#     x = torch.clamp(x, 1e-6, 1 - 1e-6)
#     logit_x = torch.log(x) - torch.log1p(-x)
#     pre_logit_scale = math.log(constraint) - math.log(1 - constraint)
#     ldj = (F.softplus(logit_x) + F.softplus(-logit_x)
#            - F.softplus(-pre_logit_scale)).flatten(1).sum(1)
#     return logit_x, ldj


# # ════════════════════════════════════════════════════════════════════════════ #
# #                       ––  Image generation helper  ––                       #
# # ════════════════════════════════════════════════════════════════════════════ #
# @torch.no_grad()
# def generate_images(
#     model,
#     prior,
#     epoch: int,
#     n_images: int,
#     image_dims: Tuple[int, int, int],
#     device: torch.device | str,
#     out_dir: Path,
#     logger: logging.Logger,
# ):
#     """
#     Samples `n_images` from PRIOR → inverse(flow) → pixel space,
#     writes a PNG grid to `out_dir/sample_epochXXX.png`.
#     """
#     n_channels, H, W = image_dims
#     z = prior.sample((n_images, n_channels, H, W)).to(device)
#     x_logits, _ = model.inverse(z)
#     x_pixels, _ = logit_transform(x_logits, inverse=True)
#     x_pixels = torch.clamp(x_pixels.cpu(), 0, 1)

#     grid = torchvision.utils.make_grid(x_pixels, nrow=int(math.sqrt(n_images)))
#     out_path = out_dir / f"sample_epoch{epoch:03d}.png"
#     torchvision.utils.save_image(grid, out_path)
#     logger.info("Saved sample grid → %s", out_path)


# # ════════════════════════════════════════════════════════════════════════════ #
# #                                ––  Trainer  ––                              #
# # ════════════════════════════════════════════════════════════════════════════ #
# def train(
#     *,
#     data_root: str | Path = "msds1",
#     epochs: int = 60,
#     batch_size: int = 512,
#     lr: float = 1e-3,
#     prior_type: str = "laplace",
#     out_dir: str | Path = "artifacts",
#     device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
#     img_samples: int = 16,
# ):
#     out_dir = Path(out_dir)
#     ckpt_dir = out_dir / "checkpoints"
#     sample_dir = out_dir / "samples"
#     log_dir = out_dir / "logs"
#     ckpt_dir.mkdir(parents=True, exist_ok=True)
#     sample_dir.mkdir(exist_ok=True)
#     logger = setup_logger(log_dir)

#     # ---------- data ------------------------------------------------------- #
#     train_set = MSDS1(root=data_root, split="train", transform=ToTensor())
#     train_loader = DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True,
#     )
#     logger.info("Loaded MSD‑1: %d training images", len(train_set))

#     # ---------- model ------------------------------------------------------ #
#     model = build_rg_flow(depth=8, blocks_per_layer=4).to(device)
#     prior = PriorDistribution(prior_type).to(device)
#     optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
#     logger.info("Model has %.2f M parameters",
#                 sum(p.numel() for p in model.parameters()) / 1e6)

#     n_chan, H, W = 3, 32, 32
#     bpd_const = math.log(2) * n_chan * H * W

#     # ---------- training --------------------------------------------------- #
#     for epoch in range(1, epochs + 1):
#         t0 = time.time()
#         epoch_loss = 0.0
#         model.train()

#         for i, (x, _) in enumerate(train_loader, 1):
#             x = x.to(device)
#             x, ldj_logit = logit_transform(x)

#             z, ldj_flow = model(x)
#             logp_prior = prior.log_prob(z)

#             loss = -(logp_prior + ldj_flow + ldj_logit).mean()

#             optim.zero_grad(set_to_none=True)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optim.step()

#             epoch_loss += loss.item()

#             if i % 50 == 0 or i == len(train_loader):
#                 bpd = (loss.item() + math.log(256.) * n_chan * H * W) / bpd_const
#                 logger.info("[ep %3d/%d] batch %4d/%d  loss %.4f  bpd %.4f",
#                             epoch, epochs, i, len(train_loader), loss.item(), bpd)

#         # ------ end‑of‑epoch summary -------------------------------------- #
#         logger.info("Epoch %d done in %.1fs  avg‑loss %.4f",
#                     epoch, time.time() - t0, epoch_loss / len(train_loader))

#         # sample images
#         generate_images(
#             model=model,
#             prior=prior,
#             epoch=epoch,
#             n_images=img_samples,
#             image_dims=(n_chan, H, W),
#             device=device,
#             out_dir=sample_dir,
#             logger=logger,
#         )

#         # checkpoint
#         ckpt_path = ckpt_dir / f"rgflow_epoch{epoch:03d}.pth"
#         torch.save(
#             {
#                 "model": model.state_dict(),
#                 "optim": optim.state_dict(),
#                 "epoch": epoch,
#             },
#             ckpt_path,
#         )
#         logger.info("Checkpoint saved → %s", ckpt_path)

#     logger.info("Training completed – all artifacts in %s", out_dir)


# # ════════════════════════════════════════════════════════════════════════════ #
# #                                  ––  CLI  ––                                #
# # ════════════════════════════════════════════════════════════════════════════ #
# def cli():
#     p = argparse.ArgumentParser(description="Train MERA / RG‑Flow on MSD‑1")
#     p.add_argument("--data-root", type=str, default="msds1")
#     p.add_argument("--epochs", type=int, default=60)
#     p.add_argument("--batch-size", type=int, default=512)
#     p.add_argument("--lr", type=float, default=1e-3)
#     p.add_argument("--prior", choices=["laplace", "gaussian"], default="laplace")
#     p.add_argument("--out", type=str, default="artifacts")
#     p.add_argument("--device", type=str,
#                    default="cuda" if torch.cuda.is_available() else "cpu")
#     p.add_argument("--img-samples", type=int, default=16,
#                    help="Number of images to sample per epoch")
#     args = p.parse_args()

#     train(
#         data_root=args.data_root,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         lr=args.lr,
#         prior_type=args.prior,
#         out_dir=args.out,
#         device=args.device,
#         img_samples=args.img_samples,
#     )


# if __name__ == "__main__":
#     cli()
