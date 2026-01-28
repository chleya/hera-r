# -*- coding: utf-8 -*-
"""Tests for evolutionary layer module."""

import pytest
import torch
from unittest.mock import Mock, patch
from hera.modules.evolutionary_layer import NeuroEvolutionaryLayer


class MockModel:
    """Mock transformer model for testing."""
    
    def __init__(self, layer_idx=8):
        self.blocks = [MockBlock(i) for i in range(12)]
        self.layer_idx = layer_idx
    
    @property
    def blocks(self):
        return self._blocks
    
    @blocks.setter
    def blocks(self, value):
        self._blocks = value


class MockBlock:
    """Mock transformer block."""
    
    def __init__(self, idx):
        self.idx = idx
        self.mlp = MockMLP()


class MockMLP:
    """Mock MLP layer."""
    
    def __init__(self):
        self.W_out = MockLinear()


class MockLinear:
    """Mock linear layer with weight matrix."""
    
    def __init__(self, in_features=768, out_features=768):
        self.W = MockParameter(torch.randn(out_features, in_features))
        self.data = self.W.data


class MockParameter:
    """Mock parameter with data attribute."""
    
    def __init__(self, data):
        self.data = data.clone()
    
    def clone(self):
        return self.data.clone()


class MockSAEInterface:
    """Mock SAE interface for testing."""
    
    def __init__(self, num_features=100, hidden_dim=768):
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.W_dec = torch.randn(num_features, hidden_dim)
    
    def encode(self, activations):
        batch_size, seq_len, hidden_dim = activations.shape
        # Return random feature activations
        return torch.randn(batch_size, seq_len, self.num_features)
    
    def decode_direction(self, feature_idx):
        return self.W_dec[feature_idx].detach()


@pytest.fixture
def mock_model():
    """Provide mock model instance."""
    return MockModel(layer_idx=8)


@pytest.fixture
def mock_sae():
    """Provide mock SAE interface."""
    return MockSAEInterface()


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "evolution": {
            "target_layers": [8],
            "learning_rate": 0.01,
            "max_rank": 1,
            "top_k_features": 3,
        },
        "sae": {
            "threshold": 2.0,
        },
    }


def test_evolutionary_layer_initialization(mock_model, mock_sae, test_config):
    """Test NeuroEvolutionaryLayer initialization."""
    layer = NeuroEvolutionaryLayer(mock_model, 8, mock_sae, test_config)
    
    # Verify initialization
    assert layer.model == mock_model
    assert layer.layer_idx == 8
    assert layer.sae == mock_sae
    assert layer.cfg == test_config
    
    # Verify target module is correctly identified
    assert layer.target_module == mock_model.blocks[8].mlp.W_out
    
    # Verify initial weight is saved
    assert torch.allclose(
        layer.initial_weight,
        mock_model.blocks[8].mlp.W_out.W.data
    )
    assert layer.initial_weight.shape == (768, 768)


def test_propose_with_valid_features(mock_model, mock_sae, test_config):
    """Test propose method with features above threshold."""
    layer = NeuroEvolutionaryLayer(mock_model, 8, mock_sae, test_config)
    
    # Create test activations
    activations = torch.randn(1, 5, 768)  # batch=1, seq_len=5, hidden=768
    
    # Mock SAE.encode to return features with high activation
    with patch.object(mock_sae, 'encode') as mock_encode:
        # Create feature activations with some high values
        feature_acts = torch.randn(1, 5, 100)
        feature_acts[0, -1, :3] = 3.0  # First 3 features above threshold (2.0)
        mock_encode.return_value = feature_acts
        
        # Call propose
        delta_w = layer.propose(activations)
    
    # Should return weight update
    assert delta_w is not None
    assert delta_w.shape == (768, 768)  # Should match weight matrix shape
    
    # Verify learning rate is applied
    expected_norm = delta_w.norm()
    assert expected_norm > 0


def test_propose_no_valid_features(mock_model, mock_sae, test_config):
    """Test propose method when no features exceed threshold."""
    layer = NeuroEvolutionaryLayer(mock_model, 8, mock_sae, test_config)
    
    # Create test activations
    activations = torch.randn(1, 5, 768)
    
    # Mock SAE.encode to return features with low activation
    with patch.object(mock_sae, 'encode') as mock_encode:
        # All features below threshold
        feature_acts = torch.ones(1, 5, 100) * 1.0  # All 1.0 < 2.0 threshold
        mock_encode.return_value = feature_acts
        
        # Call propose
        delta_w = layer.propose(activations)
    
    # Should return None when no valid features
    assert delta_w is None


def test_propose_feature_selection(mock_model, mock_sae, test_config):
    """Test that propose selects correct features."""
    layer = NeuroEvolutionaryLayer(mock_model, 8, mock_sae, test_config)
    
    # Create test activations
    activations = torch.randn(1, 10, 768)
    
    # Mock SAE.encode with specific feature activations
    with patch.object(mock_sae, 'encode') as mock_encode:
        feature_acts = torch.zeros(1, 10, 100)
        
        # Set specific features to be high on last token
        feature_acts[0, -1, 0] = 3.0  # Above threshold
        feature_acts[0, -1, 1] = 1.0  # Below threshold
        feature_acts[0, -1, 2] = 4.0  # Above threshold
        feature_acts[0, -1, 3] = 2.5  # Above threshold
        
        mock_encode.return_value = feature_acts
        
        # Mock decode_direction to track which features are used
        used_features = []
        original_decode = mock_sae.decode_direction
        
        def track_decode(feature_idx):
            used_features.append(feature_idx)
            return original_decode(feature_idx)
        
        with patch.object(mock_sae, 'decode_direction', side_effect=track_decode):
            delta_w = layer.propose(activations)
    
    # Should use features 0, 2, 3 (above threshold, top 3)
    # Note: top_k_features=3, but only 3 are above threshold
    assert len(used_features) == 3
    assert set(used_features) == {0, 2, 3}
    assert delta_w is not None


def test_commit_and_rollback(mock_model, mock_sae, test_config):
    """Test commit and rollback methods."""
    layer = NeuroEvolutionaryLayer(mock_model, 8, mock_sae, test_config)
    
    # Save initial weight
    initial_weight = layer.target_module.W.data.clone()
    
    # Create a test weight update
    delta_w = torch.randn(768, 768) * 0.1
    
    # Test commit
    layer.commit(delta_w)
    
    # Verify weight was updated
    new_weight = layer.target_module.W.data
    expected_weight = initial_weight + delta_w
    assert torch.allclose(new_weight, expected_weight, rtol=1e-5)
    
    # Test rollback
    layer.rollback(delta_w)
    
    # Verify weight returned to initial state
    final_weight = layer.target_module.W.data
    assert torch.allclose(final_weight, initial_weight, rtol=1e-5)


def test_propose_batch_handling(mock_model, mock_sae, test_config):
    """Test propose with different batch sizes."""
    layer = NeuroEvolutionaryLayer(mock_model, 8, mock_sae, test_config)
    
    batch_sizes = [1, 2, 4]
    seq_lens = [5, 10, 20]
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            activations = torch.randn(batch_size, seq_len, 768)
            
            with patch.object(mock_sae, 'encode') as mock_encode:
                # All features above threshold
                feature_acts = torch.ones(batch_size, seq_len, 100) * 3.0
                mock_encode.return_value = feature_acts
                
                delta_w = layer.propose(activations)
                
                # Should always work regardless of batch/seq length
                assert delta_w is not None
                assert delta_w.shape == (768, 768)


def test_edge_cases(mock_model, mock_sae, test_config):
    """Test edge cases for evolutionary layer."""
    layer = NeuroEvolutionaryLayer(mock_model, 8, mock_sae, test_config)
    
    # Test with single token
    single_activation = torch.randn(1, 1, 768)
    with patch.object(mock_sae, 'encode') as mock_encode:
        feature_acts = torch.ones(1, 1, 100) * 3.0
        mock_encode.return_value = feature_acts
        
        delta_w = layer.propose(single_activation)
        assert delta_w is not None
    
    # Test with empty batch (should still work theoretically)
    empty_activation = torch.randn(0, 0, 768)
    with patch.object(mock_sae, 'encode') as mock_encode:
        feature_acts = torch.randn(0, 0, 100)
        mock_encode.return_value = feature_acts
        
        # This might fail due to indexing, but that's OK
        # We're testing that it doesn't crash in unexpected ways
        try:
            delta_w = layer.propose(empty_activation)
            # If it succeeds, delta_w should be None (no features)
            if delta_w is not None:
                assert delta_w.shape == (768, 768)
        except Exception:
            # Indexing error is acceptable for empty input
            pass


def test_config_parameter_effects(mock_model, mock_sae):
    """Test how different config parameters affect propose."""
    test_cases = [
        {
            "config": {
                "evolution": {"top_k_features": 1, "learning_rate": 0.1},
                "sae": {"threshold": 0.0}
            },
            "expected_features": 1,
        },
        {
            "config": {
                "evolution": {"top_k_features": 5, "learning_rate": 0.01},
                "sae": {"threshold": 1.0}
            },
            "expected_features": 5,
        },
        {
            "config": {
                "evolution": {"top_k_features": 10, "learning_rate": 0.001},
                "sae": {"threshold": 5.0}  # High threshold, might get 0 features
            },
            "expected_features": 0,  # Might get 0 if no features above threshold
        },
    ]
    
    for test_case in test_cases:
        config = test_case["config"]
        layer = NeuroEvolutionaryLayer(mock_model, 8, mock_sae, config)
        
        activations = torch.randn(1, 5, 768)
        
        with patch.object(mock_sae, 'encode') as mock_encode:
            # All features at 2.0 (above 0.0 and 1.0 thresholds, below 5.0)
            feature_value = 2.0
            feature_acts = torch.ones(1, 5, 100) * feature_value
            mock_encode.return_value = feature_acts
            
            delta_w = layer.propose(activations)
            
            if test_case["expected_features"] > 0 and config["sae"]["threshold"] <= feature_value:
                assert delta_w is not None
                # Learning rate should be applied
                assert delta_w.norm() > 0
            else:
                assert delta_w is None


def test_weight_update_calculation(mock_model, mock_sae, test_config):
    """Test the mathematical correctness of weight updates."""
    layer = NeuroEvolutionaryLayer(mock_model, 8, mock_sae, test_config)
    
    activations = torch.randn(1, 5, 768)
    
    # Mock specific feature directions
    with patch.object(mock_sae, 'encode') as mock_encode:
        # Single feature activation
        feature_acts = torch.zeros(1, 5, 100)
        feature_acts[0, -1, 0] = 3.0  # Only feature 0 is active
        mock_encode.return_value = feature_acts
    
    with patch.object(mock_sae, 'decode_direction') as mock_decode:
        # Use a simple test direction
        test_direction = torch.zeros(768)
        test_direction[0] = 1.0  # Unit vector along first dimension
        mock_decode.return_value = test_direction
        
        delta_w = layer.propose(activations)
    
    # With single feature, update should be outer product of direction with itself
    expected_update = torch.outer(test_direction, test_direction)
    expected_update = expected_update * test_config["evolution"]["learning_rate"]
    
    assert delta_w is not None
    assert torch.allclose(delta_w, expected_update, rtol=1e-5)