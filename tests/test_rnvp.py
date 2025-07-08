import torch
import torch.nn as nn
import pytest
import numpy as np
from src.layers.RNVP import RNVPBlock, create_masks

class SimpleNetwork(nn.Module):
    """Simple network for testing purposes"""
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class TestRNVPInvertibility:
    
    @pytest.fixture
    def setup_networks(self):
        """Setup test networks and parameters"""
        kernel_size = 4
        num_channels = 3
        num_flows = 2
        
        # Create simple networks for s and t
        s_nets = [SimpleNetwork(num_channels, kernel_size) for _ in range(num_flows)]
        t_nets = [SimpleNetwork(num_channels, kernel_size) for _ in range(num_flows)]
        
        return {
            'kernel_size': kernel_size,
            'num_channels': num_channels,
            'num_flows': num_flows,
            's_nets': s_nets,
            't_nets': t_nets
        }
    
    def test_perfect_invertibility_single_batch(self, setup_networks):
        """Test that forward -> inverse gives back original input for single batch"""
        params = setup_networks
        rnvp = RNVPBlock(
            params['kernel_size'], 
            params['num_channels'], 
            params['s_nets'], 
            params['t_nets']
        )
        rnvp.eval()  # Set to eval mode for consistent behavior
        
        # Create test input
        batch_size = 1
        x_original = torch.randn(batch_size, params['num_channels'], 
                               params['kernel_size'], params['kernel_size'])
        
        with torch.no_grad():
            # Forward pass
            x_forward, ldj_forward = rnvp.forward(x_original)
            
            # Inverse pass
            x_reconstructed, ldj_inverse = rnvp.inverse(x_forward)
            
            # Check reconstruction error
            reconstruction_error = torch.abs(x_original - x_reconstructed).max().item()
            assert reconstruction_error < 1e-5, f"Reconstruction error too large: {reconstruction_error}"
            
            # Check log determinant jacobian consistency
            ldj_total = ldj_forward + ldj_inverse
            assert torch.abs(ldj_total).max().item() < 1e-5, f"LDJ inconsistency: {ldj_total.max().item()}"
    
    def test_perfect_invertibility_multiple_batches(self, setup_networks):
        """Test invertibility with multiple batch sizes"""
        params = setup_networks
        rnvp = RNVPBlock(
            params['kernel_size'], 
            params['num_channels'], 
            params['s_nets'], 
            params['t_nets']
        )
        rnvp.eval()
        
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            x_original = torch.randn(batch_size, params['num_channels'], 
                                   params['kernel_size'], params['kernel_size'])
            
            with torch.no_grad():
                x_forward, ldj_forward = rnvp.forward(x_original)
                x_reconstructed, ldj_inverse = rnvp.inverse(x_forward)
                
                reconstruction_error = torch.abs(x_original - x_reconstructed).max().item()
                assert reconstruction_error < 1e-5, \
                    f"Batch size {batch_size}: Reconstruction error {reconstruction_error}"
                
                ldj_total = ldj_forward + ldj_inverse
                assert torch.abs(ldj_total).max().item() < 1e-5, \
                    f"Batch size {batch_size}: LDJ inconsistency {ldj_total.max().item()}"
    
    def test_inverse_then_forward(self, setup_networks):
        """Test that inverse -> forward also gives back original input"""
        params = setup_networks
        rnvp = RNVPBlock(
            params['kernel_size'], 
            params['num_channels'], 
            params['s_nets'], 
            params['t_nets']
        )
        rnvp.eval()
        
        batch_size = 4
        x_original = torch.randn(batch_size, params['num_channels'], 
                               params['kernel_size'], params['kernel_size'])
        
        with torch.no_grad():
            # Inverse pass first
            x_inverse, ldj_inverse = rnvp.inverse(x_original)
            
            # Then forward pass
            x_reconstructed, ldj_forward = rnvp.forward(x_inverse)
            
            reconstruction_error = torch.abs(x_original - x_reconstructed).max().item()
            assert reconstruction_error < 1e-5, f"Reconstruction error: {reconstruction_error}"
            
            ldj_total = ldj_forward + ldj_inverse
            assert torch.abs(ldj_total).max().item() < 1e-5, f"LDJ inconsistency: {ldj_total.max().item()}"
    
    def test_different_kernel_sizes(self):
        """Test invertibility with different kernel sizes"""
        kernel_sizes = [2, 3, 4, 5, 8]
        num_channels = 2
        
        for kernel_size in kernel_sizes:
            s_nets = [SimpleNetwork(num_channels, kernel_size)]
            t_nets = [SimpleNetwork(num_channels, kernel_size)]
            
            rnvp = RNVPBlock(kernel_size, num_channels, s_nets, t_nets)
            rnvp.eval()
            
            batch_size = 2
            x_original = torch.randn(batch_size, num_channels, kernel_size, kernel_size)
            
            with torch.no_grad():
                x_forward, ldj_forward = rnvp.forward(x_original)
                x_reconstructed, ldj_inverse = rnvp.inverse(x_forward)
                
                reconstruction_error = torch.abs(x_original - x_reconstructed).max().item()
                assert reconstruction_error < 1e-5, \
                    f"Kernel size {kernel_size}: Reconstruction error {reconstruction_error}"
    
    def test_different_channel_numbers(self):
        """Test invertibility with different numbers of channels"""
        kernel_size = 4
        channel_numbers = [1, 2, 3, 4, 8]
        
        for num_channels in channel_numbers:
            s_nets = [SimpleNetwork(num_channels, kernel_size)]
            t_nets = [SimpleNetwork(num_channels, kernel_size)]
            
            rnvp = RNVPBlock(kernel_size, num_channels, s_nets, t_nets)
            rnvp.eval()
            
            batch_size = 2
            x_original = torch.randn(batch_size, num_channels, kernel_size, kernel_size)
            
            with torch.no_grad():
                x_forward, ldj_forward = rnvp.forward(x_original)
                x_reconstructed, ldj_inverse = rnvp.inverse(x_forward)
                
                reconstruction_error = torch.abs(x_original - x_reconstructed).max().item()
                assert reconstruction_error < 1e-5, \
                    f"Channels {num_channels}: Reconstruction error {reconstruction_error}"
    
    def test_multiple_flows_invertibility(self):
        """Test invertibility with different numbers of flows"""
        kernel_size = 4
        num_channels = 3
        flow_numbers = [1, 2, 3, 4, 5]
        
        for num_flows in flow_numbers:
            s_nets = [SimpleNetwork(num_channels, kernel_size) for _ in range(num_flows)]
            t_nets = [SimpleNetwork(num_channels, kernel_size) for _ in range(num_flows)]
            
            rnvp = RNVPBlock(kernel_size, num_channels, s_nets, t_nets)
            rnvp.eval()
            
            batch_size = 2
            x_original = torch.randn(batch_size, num_channels, kernel_size, kernel_size)
            
            with torch.no_grad():
                x_forward, ldj_forward = rnvp.forward(x_original)
                x_reconstructed, ldj_inverse = rnvp.inverse(x_forward)
                
                reconstruction_error = torch.abs(x_original - x_reconstructed).max().item()
                assert reconstruction_error < 1e-5, \
                    f"Flows {num_flows}: Reconstruction error {reconstruction_error}"
    
    def test_gradient_consistency(self, setup_networks):
        """Test that gradients are consistent through forward and inverse"""
        params = setup_networks
        rnvp = RNVPBlock(
            params['kernel_size'], 
            params['num_channels'], 
            params['s_nets'], 
            params['t_nets']
        )
        
        batch_size = 4
        x_original = torch.randn(batch_size, params['num_channels'], 
                               params['kernel_size'], params['kernel_size'], 
                               requires_grad=True)
        
        # Forward pass
        x_forward, ldj_forward = rnvp.forward(x_original)
        
        # Create a simple loss
        loss = x_forward.sum() + ldj_forward.sum()
        loss.backward()
        
        grad_original = x_original.grad.clone()
        
        # Clear gradients
        x_original.grad.zero_()
        
        # Test through inverse path
        x_forward_detached = x_forward.detach().requires_grad_(True)
        x_reconstructed, ldj_inverse = rnvp.inverse(x_forward_detached)
        
        loss2 = x_reconstructed.sum() + ldj_inverse.sum()
        loss2.backward()
        
        # The gradients should flow back properly
        assert x_forward_detached.grad is not None, "Gradients not flowing through inverse"
    
    def test_numerical_stability(self, setup_networks):
        """Test numerical stability with extreme inputs"""
        params = setup_networks
        rnvp = RNVPBlock(
            params['kernel_size'], 
            params['num_channels'], 
            params['s_nets'], 
            params['t_nets']
        )
        rnvp.eval()
        
        batch_size = 2
        
        # Test with small values
        x_small = torch.randn(batch_size, params['num_channels'], 
                            params['kernel_size'], params['kernel_size']) * 1e-3
        
        with torch.no_grad():
            x_forward, _ = rnvp.forward(x_small)
            x_reconstructed, _ = rnvp.inverse(x_forward)
            
            assert torch.isfinite(x_forward).all(), "Forward pass produced non-finite values"
            assert torch.isfinite(x_reconstructed).all(), "Inverse pass produced non-finite values"
            
            reconstruction_error = torch.abs(x_small - x_reconstructed).max().item()
            assert reconstruction_error < 1e-4, f"Small values reconstruction error: {reconstruction_error}"
        
        # Test with large values
        x_large = torch.randn(batch_size, params['num_channels'], 
                            params['kernel_size'], params['kernel_size']) * 10
        
        with torch.no_grad():
            x_forward, _ = rnvp.forward(x_large)
            x_reconstructed, _ = rnvp.inverse(x_forward)
            
            assert torch.isfinite(x_forward).all(), "Forward pass produced non-finite values with large input"
            assert torch.isfinite(x_reconstructed).all(), "Inverse pass produced non-finite values with large input"

