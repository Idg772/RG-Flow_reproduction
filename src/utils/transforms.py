"""
Image transformation utilities for training.
"""
import math
import torch
import torch.nn.functional as F


def logit_transform(x, *, c: float = 0.9, inverse: bool = False):
    """
    Bidirectional logit transform with dequantization and log-determinant-Jacobian computation.
    
    Args:
        x: Input tensor
        c: Constraint parameter for logit transform (default: 0.9)
        inverse: Whether to apply inverse transform (default: False)
    
    Returns:
        Tuple of (transformed_tensor, log_determinant_jacobian)
    """
    if inverse:  # logits → [0,1]
        logx = x
        pre = torch.tensor(math.log(c) - math.log1p(-c), device=x.device, dtype=x.dtype)
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
    pre = torch.tensor(math.log(c) - math.log1p(-c), device=x.device, dtype=x.dtype)
    ldj = (F.softplus(logx) + F.softplus(-logx) -
           F.softplus(-pre)).flatten(1).sum(1)
    return logx, ldj