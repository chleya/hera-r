# -*- coding: utf-8 -*-
"""Tests for SAE interface module."""

import pytest
import torch
from unittest.mock import Mock, patch
from hera.modules.sae_interface import SAEInterface


class MockSAE:
    """Mock SAE model for testing."""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.W_dec = torch.randn(100, 768)  # 100 features, 768 dimensions
        self.eval_called = False
    
    def eval(self):
        self.eval_called = True
    
    def encode(self, activations):
        # Simple mock encoding: return random feature activations
        batch_size, seq_len, hidden_dim = activations.shape
        return torch.randn(batch_size, seq_len, 100)  # 100 features


@pytest.fixture
def mock_sae():
    """Provide a mock SAE instance."""
    return MockSAE()


@pytest.fixture
def test_config():
    """Provide test configuration for SAE."""
    return {
        "sae": {
            "release": "test-release",
            "id": "blocks.0.hook_resid_pre",
            "threshold": 2.0,
        },
        "experiment": {
            "device": "cpu",
        },
    }


def test_sae_interface_initialization(test_config):
    """Test SAEInterface initialization."""
    with patch('sae_lens.SAE.from_pretrained') as mock_from_pretrained:
        # Setup mock return value
        mock_sae_instance = MockSAE()
        mock_from_pretrained.return_value = (mock_sae_instance, None, None)
        
        # Initialize interface
        interface = SAEInterface(test_config)
        
        # Verify initialization
        assert interface.cfg == test_config
        assert interface.sae == mock_sae_instance
        assert mock_sae_instance.eval_called is True
        
        # Verify SAE was loaded with correct parameters
        mock_from_pretrained.assert_called_once_with(
            release="test-release",
            sae_id="blocks.0.hook_resid_pre",
            device="cpu"
        )


def test_sae_interface_encode(test_config):
    """Test encode method."""
    with patch('sae_lens.SAE.from_pretrained') as mock_from_pretrained:
        mock_sae_instance = MockSAE()
        mock_from_pretrained.return_value = (mock_sae_instance, None, None)
        
        interface = SAEInterface(test_config)
        
        # Create test activations
        activations = torch.randn(2, 10, 768)  # batch=2, seq_len=10, hidden=768
        
        # Test encoding
        with torch.no_grad():
            features = interface.encode(activations)
        
        # Verify output shape
        assert features.shape == (2, 10, 100)  # Should have 100 features
        
        # Verify no gradients (torch.no_grad context)
        assert not features.requires_grad


def test_sae_interface_decode_direction(test_config):
    """Test decode_direction method."""
    with patch('sae_lens.SAE.from_pretrained') as mock_from_pretrained:
        mock_sae_instance = MockSAE()
        mock_from_pretrained.return_value = (mock_sae_instance, None, None)
        
        interface = SAEInterface(test_config)
        
        # Test decoding for different feature indices
        for feature_idx in [0, 50, 99]:  # Test first, middle, last features
            direction = interface.decode_direction(feature_idx)
            
            # Verify shape
            assert direction.shape == (768,)  # Should match hidden dimension
            
            # Verify no gradients
            assert not direction.requires_grad
            
            # Verify it comes from W_dec
            expected_direction = mock_sae_instance.W_dec[feature_idx].detach()
            assert torch.allclose(direction, expected_direction)


def test_sae_interface_edge_cases(test_config):
    """Test edge cases for SAEInterface."""
    with patch('sae_lens.SAE.from_pretrained') as mock_from_pretrained:
        mock_sae_instance = MockSAE()
        mock_from_pretrained.return_value = (mock_sae_instance, None, None)
        
        interface = SAEInterface(test_config)
        
        # Test with empty activations (edge case)
        empty_activations = torch.randn(0, 0, 768)
        with torch.no_grad():
            features = interface.encode(empty_activations)
        assert features.shape == (0, 0, 100)
        
        # Test with single token
        single_activation = torch.randn(1, 1, 768)
        with torch.no_grad():
            features = interface.encode(single_activation)
        assert features.shape == (1, 1, 100)


def test_sae_interface_device_handling():
    """Test SAEInterface with different devices."""
    test_cases = [
        {"device": "cpu"},
        {"device": "cuda"},
    ]
    
    for device_config in test_cases:
        config = {
            "sae": {
                "release": "test-release",
                "id": "blocks.0.hook_resid_pre",
                "threshold": 2.0,
            },
            "experiment": device_config,
        }
        
        with patch('sae_lens.SAE.from_pretrained') as mock_from_pretrained:
            mock_sae_instance = MockSAE(device=device_config["device"])
            mock_from_pretrained.return_value = (mock_sae_instance, None, None)
            
            interface = SAEInterface(config)
            
            # Verify SAE was loaded with correct device
            mock_from_pretrained.assert_called_with(
                release="test-release",
                sae_id="blocks.0.hook_resid_pre",
                device=device_config["device"]
            )


def test_sae_interface_error_handling(test_config):
    """Test error handling in SAEInterface."""
    # Test initialization error
    with patch('sae_lens.SAE.from_pretrained') as mock_from_pretrained:
        mock_from_pretrained.side_effect = RuntimeError("SAE loading failed")
        
        with pytest.raises(RuntimeError, match="SAE loading failed"):
            SAEInterface(test_config)
    
    # Test encode with invalid input
    with patch('sae_lens.SAE.from_pretrained') as mock_from_pretrained:
        mock_sae_instance = MockSAE()
        mock_from_pretrained.return_value = (mock_sae_instance, None, None)
        
        interface = SAEInterface(test_config)
        
        # Test with wrong dimension activations
        wrong_dims = torch.randn(2, 10)  # 2D instead of 3D
        with pytest.raises(Exception):
            with torch.no_grad():
                interface.encode(wrong_dims)


@pytest.mark.slow
def test_sae_interface_integration():
    """Integration test with actual SAE (marked as slow)."""
    # This test would actually load a small SAE model
    # For now, we'll skip it in regular test runs
    pytest.skip("Integration test with actual SAE - run manually")


def test_sae_interface_config_validation():
    """Test that SAEInterface validates config structure."""
    invalid_configs = [
        {},  # Empty config
        {"sae": {}},  # Missing SAE fields
        {"sae": {"release": "test"}},  # Missing id
        {"sae": {"id": "test"}},  # Missing release
    ]
    
    for config in invalid_configs:
        with patch('sae_lens.SAE.from_pretrained') as mock_from_pretrained:
            # SAE.from_pretrained should raise an error with invalid config
            mock_from_pretrained.side_effect = KeyError("Missing config key")
            
            with pytest.raises((KeyError, RuntimeError)):
                SAEInterface(config)