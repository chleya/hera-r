# -*- coding: utf-8 -*-
"""Tests for utility modules."""

import torch
import numpy as np
from hera.utils.metrics import js_divergence, cosine_drift


def test_js_divergence_basic():
    """Test basic JS divergence calculation."""
    # Create simple probability distributions
    p_logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    q_logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    
    js = js_divergence(p_logits, q_logits)
    
    # JS divergence should be symmetric and between 0 and 1
    assert 0 <= js <= 1
    assert isinstance(js, float)
    
    # Test symmetry
    js_reverse = js_divergence(q_logits, p_logits)
    assert np.isclose(js, js_reverse, rtol=1e-6)


def test_js_divergence_identical():
    """Test JS divergence for identical distributions."""
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    js = js_divergence(logits, logits)
    
    # JS divergence of identical distributions should be 0
    assert np.isclose(js, 0.0, atol=1e-10)


def test_cosine_drift_basic():
    """Test basic cosine drift calculation."""
    # Create test tensors
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    
    drift = cosine_drift(a, b)
    
    # Identical vectors should have 0 drift
    assert np.isclose(drift, 0.0, atol=1e-10)
    assert isinstance(drift, float)


def test_cosine_drift_orthogonal():
    """Test cosine drift for orthogonal vectors."""
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    
    drift = cosine_drift(a, b)
    
    # Orthogonal vectors should have drift = 1 (cosine similarity = 0)
    assert np.isclose(drift, 1.0, atol=1e-10)


def test_cosine_drift_high_dim():
    """Test cosine drift with higher dimensional tensors."""
    # 3D tensor (batch, seq_len, features)
    a = torch.randn(2, 5, 10)
    b = torch.randn(2, 5, 10)
    
    drift = cosine_drift(a, b)
    
    # Should return a float between 0 and 1
    assert 0 <= drift <= 1
    assert isinstance(drift, float)


def test_cosine_drift_batch_consistency():
    """Test that cosine drift handles batches consistently."""
    batch_size = 4
    feature_dim = 8
    
    # Create identical batches
    a = torch.randn(batch_size, feature_dim)
    b = a.clone()  # Exact copy
    
    drift = cosine_drift(a, b)
    
    # Identical batches should have 0 drift
    assert np.isclose(drift, 0.0, atol=1e-10)


def test_metrics_stability():
    """Test that metrics are numerically stable."""
    # Test with very small values
    small_logits = torch.tensor([[1e-10, 2e-10, 3e-10]])
    js = js_divergence(small_logits, small_logits)
    
    # Should handle small values without NaN or inf
    assert not np.isnan(js)
    assert not np.isinf(js)
    assert np.isclose(js, 0.0, atol=1e-6)
    
    # Test with large values
    large_logits = torch.tensor([[1e10, 2e10, 3e10]])
    js = js_divergence(large_logits, large_logits)
    
    assert not np.isnan(js)
    assert not np.isinf(js)
    assert np.isclose(js, 0.0, atol=1e-6)