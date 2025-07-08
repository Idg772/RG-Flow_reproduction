
import torch
import torch.nn as nn
import pytest
from src.layers.Disentangler import Disentangler, disentangler_grid


class IdentityNetwork(nn.Module):
    """A simple identity network that returns input unchanged with zero log-det-jacobian."""
    
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
    
    def forward(self, x):
        """Forward pass - returns input unchanged."""
        # x has shape (B*N, C*m*m) where B*N is the batch of patches
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return x, ldj
    
    def inverse(self, x):
        """Inverse pass - same as forward for identity."""
        # x has shape (B*N, C*m*m) where B*N is the batch of patches
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return x, ldj


class TestDisentangler:
    """Test suite for the Disentangler class."""
    
    @pytest.fixture
    def setup_disentangler(self):
        """Setup fixture for creating disentangler instances."""
        m = 4  # patch size
        num_channels = 3
        input_dim = num_channels * m * m  # C * m * m
        
        identity_net = IdentityNetwork(input_dim)
        disentangler = Disentangler(m=m, num_channels=num_channels, network=identity_net)
        
        return {
            'disentangler': disentangler,
            'identity_net': identity_net,
            'm': m,
            'num_channels': num_channels,
            'input_dim': input_dim
        }
    
    def test_identity_recovery_small_input(self, setup_disentangler):
        """Test identity recovery with a small input tensor."""
        setup = setup_disentangler
        disentangler = setup['disentangler']
        m = setup['m']
        
        # Create a small input tensor (divisible by m)
        B, C, H, W = 2, 3, 8, 8  # 8x8 is divisible by 4x4 patches
        x_input = torch.randn(B, C, H, W)
        
        # Calculate expected number of patches
        N = (H // m) * (W // m)  # Number of patches per image
        
        # Forward pass
        x_output, ldj = disentangler.forward(x_input)
        
        # Check shapes
        assert x_output.shape == x_input.shape, f"Output shape {x_output.shape} != input shape {x_input.shape}"
        # LDJ might be per patch (B*N,) and needs to be aggregated to (B,)
        expected_ldj_shape = (B,) if ldj.shape == (B,) else (B * N,)
        assert ldj.shape in [(B,), (B * N,)], f"LDJ shape {ldj.shape} not in expected shapes {[(B,), (B * N,)]}"
        
        # Check if we recover the input (within numerical precision)
        torch.testing.assert_close(x_output, x_input, rtol=1e-5, atol=1e-6)
        
        # Check that LDJ is zero (identity transformation)
        if ldj.shape == (B * N,):
            # If LDJ is per patch, reshape and sum over patches
            ldj_reshaped = ldj.view(B, N).sum(dim=1)
            expected_ldj = torch.zeros(B)
            torch.testing.assert_close(ldj_reshaped, expected_ldj, rtol=1e-5, atol=1e-6)
        else:
            # If LDJ is already aggregated per batch
            expected_ldj = torch.zeros(B)
            torch.testing.assert_close(ldj, expected_ldj, rtol=1e-5, atol=1e-6)
    
    def test_identity_recovery_various_sizes(self, setup_disentangler):
        """Test identity recovery with various input sizes."""
        setup = setup_disentangler
        disentangler = setup['disentangler']
        m = setup['m']
        
        # Test different sizes (all divisible by m=4)
        test_sizes = [
            (1, 3, 4, 4),    # Minimal size
            (1, 3, 16, 16),  # Larger square
            (3, 3, 12, 8),   # Different batch size
        ]
        
        for B, C, H, W in test_sizes:
            x_input = torch.randn(B, C, H, W)
            x_output, ldj = disentangler.forward(x_input)
            
            # Calculate expected number of patches
            N = (H // m) * (W // m)
            
            # Check recovery
            torch.testing.assert_close(
                x_output, x_input, 
                rtol=1e-5, atol=1e-6,
                msg=f"Failed for size ({B}, {C}, {H}, {W})"
            )
            
            # Check LDJ is zero (handle both aggregated and per-patch cases)
            if ldj.shape == (B * N,):
                # If LDJ is per patch, reshape and sum over patches
                ldj_aggregated = ldj.view(B, N).sum(dim=1)
                expected_ldj = torch.zeros(B)
                torch.testing.assert_close(
                    ldj_aggregated, expected_ldj, 
                    rtol=1e-5, atol=1e-6,
                    msg=f"LDJ not zero for size ({B}, {C}, {H}, {W})"
                )
            else:
                # If LDJ is already aggregated per batch
                expected_ldj = torch.zeros(B)
                torch.testing.assert_close(
                    ldj, expected_ldj, 
                    rtol=1e-5, atol=1e-6,
                    msg=f"LDJ not zero for size ({B}, {C}, {H}, {W})"
                )
    
    def test_forward_inverse_consistency(self, setup_disentangler):
        """Test that forward and inverse operations are consistent."""
        setup = setup_disentangler
        disentangler = setup['disentangler']
        
        B, C, H, W = 2, 3, 8, 8
        x_input = torch.randn(B, C, H, W)
        
        # Forward then inverse
        x_forward, ldj_forward = disentangler.forward(x_input)
        x_reconstructed, ldj_inverse = disentangler.inverse(x_forward)
        
        # Should recover original input
        torch.testing.assert_close(x_reconstructed, x_input, rtol=1e-5, atol=1e-6)
        
        # Handle LDJ comparison (they should have the same shape and be zero for identity)
        assert ldj_forward.shape == ldj_inverse.shape, f"LDJ shapes don't match: {ldj_forward.shape} vs {ldj_inverse.shape}"
        
        # For identity network, forward and inverse LDJ should be the same (zero)
        torch.testing.assert_close(ldj_forward, ldj_inverse, rtol=1e-5, atol=1e-6)
    
    def test_disentangler_grid_invertability(self):
        """Test the disentangler_grid function separately."""
        B, C, H, W = 1, 2, 6, 6
        x = torch.randn(B, C, H, W)
        m = 4
        
        # Apply grid transformation and its inverse
        x_shifted = disentangler_grid(x, m, inverse=False)
        x_recovered = disentangler_grid(x_shifted, m, inverse=True)
        
        # Should recover original
        torch.testing.assert_close(x_recovered, x, rtol=1e-5, atol=1e-6)
    
    def test_patch_extraction_assembly(self, setup_disentangler):
        """Test that patch extraction and assembly are inverse operations."""
        setup = setup_disentangler
        disentangler = setup['disentangler']
        
        B, C, H, W = 2, 3, 8, 8
        x = torch.randn(B, C, H, W)
        
        # Extract patches
        patches, B_out, C_out, H_out, W_out, N = disentangler._extract_patches(x)
        
        # Assemble back
        x_recovered = disentangler._assemble_from_patches(patches, B_out, C_out, H_out, W_out, N)
        
        # Should recover original
        torch.testing.assert_close(x_recovered, x, rtol=1e-5, atol=1e-6)
    
    def test_wrong_number_of_channels(self, setup_disentangler):
        """Test that wrong number of channels raises an error."""
        setup = setup_disentangler
        disentangler = setup['disentangler']
        
        # Create input with wrong number of channels
        B, C, H, W = 2, 5, 8, 8  # Expected 3 channels, got 5
        x_wrong_channels = torch.randn(B, C, H, W)
        
        with pytest.raises(ValueError, match="Expected 3 channels, got 5"):
            disentangler.forward(x_wrong_channels)
    
    def test_gradient_flow(self, setup_disentangler):
        """Test that gradients flow through the network properly."""
        setup = setup_disentangler
        disentangler = setup['disentangler']
        
        B, C, H, W = 2, 3, 8, 8
        x = torch.randn(B, C, H, W, requires_grad=True)
        
        # Forward pass
        output, ldj = disentangler.forward(x)
        
        # Create a simple loss
        loss = output.sum() + ldj.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None, "No gradients computed for input"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Gradients are all zero"

