import torch
import torch.nn as nn


class Decimator(nn.Module):
    def __init__(self, m: int, num_channels: int, network):
        super().__init__()
        self.m = m
        self.num_channels = num_channels
        self.network = network
        self.unfold = nn.Unfold(kernel_size=(m, m), stride=(m, m))
        self.fold = None  # Will be set dynamically per input shape

    def _extract_patches(self, x: torch.Tensor):
        # Input: (B, C, H, W)
        B, C, H, W = x.shape
        unfolded = self.unfold(x)  # (B, C*m*m, N)
        N = unfolded.shape[-1]
        patch_size = self.m * self.m
        unfolded = unfolded.view(B, C, patch_size, N)     # (B, C, m*m, N)
        unfolded = unfolded.permute(0, 3, 1, 2)           # (B, N, C, m*m)
        unfolded = unfolded.reshape(B * N, C, self.m, self.m) # (B*N, C, m, m)
        return unfolded, B, C, H, W, N

    def _assemble_from_patches(self, x_patched, B, C, H, W, N):
        patch_size = self.m * self.m
        x = x_patched.view(B, N, C, patch_size)          # (B, N, C, m*m)
        x = x.permute(0, 2, 3, 1)                       # (B, C, m*m, N)
        x = x.reshape(B, C * patch_size, N)             # (B, C*m*m, N)
        if self.fold is None or self.fold.output_size != (H, W):
            self.fold = nn.Fold(output_size=(H, W), kernel_size=(self.m, self.m), stride=(self.m, self.m))
        x = self.fold(x)                                # (B, C, H, W)
        return x

    def _check_channels(self, x):
        if x.shape[1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {x.shape[1]}.")

    def _core(self, x, net_fn):
        self._check_channels(x)
        x_unfolded, B, C, H, W, N = self._extract_patches(x)
        x_transformed, ldj = net_fn(x_unfolded)
        x_folded = self._assemble_from_patches(x_transformed, B, C, H, W, N)
        ldj = ldj.view(B, -1).sum(dim=1)
        return x_folded, ldj

    def forward(self, x):
        return self._core(x, self.network)

    def inverse(self, x):
        return self._core(x, self.network.inverse)
