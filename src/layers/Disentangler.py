import torch
import torch.nn as nn

def disentangler_grid(x: torch.Tensor, m: int, inverse: bool = False) -> torch.Tensor:
    """Rolls tensor spatially by m//2 for disentangling."""
    shift = m // 2
    shifts = (-shift, -shift) if not inverse else (shift, shift)
    return torch.roll(x, shifts=shifts, dims=(-2, -1))


class Disentangler(nn.Module):
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
        unfolded = unfolded.view(B, C, patch_size, N)      # (B, C, m*m, N)
        unfolded = unfolded.permute(0, 3, 1, 2)            # (B, N, C, m*m)
        unfolded = unfolded.reshape(B * N, C, self.m, self.m) # (B*N, C, m, m)
        return unfolded, B, C, H, W, N

    def _assemble_from_patches(self, x_patched, B, C, H, W, N):
        patch_size = self.m * self.m
        x = x_patched.view(B, N, C, patch_size)           # (B, N, C, m*m)
        x = x.permute(0, 2, 3, 1)                         # (B, C, m*m, N)
        x = x.reshape(B, C * patch_size, N)               # (B, C*m*m, N)
        if self.fold is None or self.fold.output_size != (H, W):
            self.fold = nn.Fold(output_size=(H, W), kernel_size=(self.m, self.m), stride=(self.m, self.m))
        x = self.fold(x)                                  # (B, C, H, W)
        return x

    def _check_channels(self, x):
        if x.shape[1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {x.shape[1]}.")

    def _core(self, x, net_fn):
        self._check_channels(x)
        x = disentangler_grid(x, self.m, inverse=False)
        x_unfolded, B, C, H, W, N = self._extract_patches(x)
        x_transformed, ldj = net_fn(x_unfolded)
        x_folded = self._assemble_from_patches(x_transformed, B, C, H, W, N)
        x_folded = disentangler_grid(x_folded, self.m, inverse=True)
        ldj = ldj.view(B, -1).sum(dim=1)
        return x_folded, ldj

    def forward(self, x):
        return self._core(x, self.network)

    def inverse(self, x):
        return self._core(x, self.network.inverse)


# import torch
# import torch.nn as nn

# def disentangler_grid(x: torch.Tensor, m: int, inverse: bool = False) -> torch.Tensor:
#     shift = m // 2
#     # dim(x): (B, C, H, W), axes -2 (height), -1 (width)
#     if not inverse:
#         return torch.roll(x, shifts=(-shift, -shift), dims=(-2, -1))
#     else:
#         return torch.roll(x, shifts=(shift, shift), dims=(-2, -1))

# class Disentangler(nn.Module):
#     def __init__(self, m: int, num_channels: int, network):
#         super(Disentangler, self).__init__()
#         self.m = m
#         self.num_channels = num_channels
#         self.network = network
#         self.unfold = nn.Unfold(kernel_size=(m, m), stride=(m, m))
    
#     def forward(self, x):
#         B, C, H, W = x.shape
#         if C != self.num_channels:
#             raise ValueError(f"Input tensor must have {self.num_channels} channels, but got {C} channels.")
        
#         # Roll the input tensor for the correct grid arrangement
#         x = disentangler_grid(x, self.m, inverse=False)
        
#         # Unfold the tensor to extract patches
#         # dim(x): (B, C, H, W) -> dim(x_unfolded): (B, C*m*m, N), where N is the number of patches
#         x_unfolded = self.unfold(x)
#         N = x_unfolded.shape[-1]
        
#         # Reshape to (B, C, m*m, N)
#         x_unfolded = x_unfolded.view(B, C, self.m * self.m, N)
        
#         # Permute to (B, N, C, m*m)
#         x_unfolded = x_unfolded.permute(0, 3, 1, 2)
        
#         # Reshape to (B*N, C*m*m)
#         x_unfolded = x_unfolded.reshape(B * N, C * self.m * self.m)
        
#         # Pass through the network
#         x_transformed, ldj = self.network(x_unfolded)
        
#         # Reshape back to (B, N, C, m*m)
#         x_transformed = x_transformed.view(B, N, C, self.m * self.m)
#         # Permute back to (B, C, m*m, N)
#         x_transformed = x_transformed.permute(0, 2, 3, 1)
        
#         # Reshape to (B, C*m*m, N)
#         x_transformed = x_transformed.reshape(B, C * self.m * self.m, N)
        
#         # Fold the transformed patches back to the original shape
#         # dim(x_transformed): (B, C*m*m, N) -> dim(x_folded): (B, C, H, W)
        
#         fold = nn.Fold(output_size=(H, W), kernel_size=(self.m, self.m), stride=(self.m, self.m))
#         x_folded = fold(x_transformed)
        
#         # Roll back to the original grid arrangement
#         x_folded = disentangler_grid(x_folded, self.m, inverse=True)
#         return x_folded, ldj
    
#     def inverse(self, x):
#         B, C, H, W = x.shape
#         if C != self.num_channels:
#             raise ValueError(f"Input tensor must have {self.num_channels} channels, but got {C} channels.")
        
#         # Roll the input tensor for the correct grid arrangement
#         x = disentangler_grid(x, self.m, inverse=False)
        
#         # Unfold the tensor to extract patches
#         # dim(x): (B, C, H, W) -> dim(x_unfolded): (B, C*m*m, N), where N is the number of patches
#         x_unfolded = self.unfold(x)
#         N = x_unfolded.shape[-1]
        
#         # Reshape to (B, C, m*m, N)
#         x_unfolded = x_unfolded.view(B, C, self.m * self.m, N)
        
#         # Permute to (B, N, C, m*m)
#         x_unfolded = x_unfolded.permute(0, 3, 1, 2)
        
#         # Reshape to (B*N, C*m*m)
#         x_unfolded = x_unfolded.reshape(B * N, C * self.m * self.m)
        
#         # Pass through the network
#         x_transformed, ldj = self.network.inverse(x_unfolded)
        
#         # Reshape back to (B, N, C, m*m)
#         x_transformed = x_transformed.view(B, N, C, self.m * self.m)
#         # Permute back to (B, C, m*m, N)
#         x_transformed = x_transformed.permute(0, 2, 3, 1)
        
#         # Reshape to (B, C*m*m, N)
#         x_transformed = x_transformed.reshape(B, C * self.m * self.m, N)
        
#         # Fold the transformed patches back to the original shape
#         # dim(x_transformed): (B, C*m*m, N) -> dim(x_folded): (B, C, H, W)
        
#         fold = nn.Fold(output_size=(H, W), kernel_size=(self.m, self.m), stride=(self.m, self.m))
#         x_folded = fold(x_transformed)
        
#         # Roll back to the original grid arrangement
#         x_folded = disentangler_grid(x_folded, self.m, inverse=True)
#         return x_folded, ldj