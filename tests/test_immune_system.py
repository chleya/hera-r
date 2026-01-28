# -*- coding: utf-8 -*-
"""Tests for immune system module."""

import pytest
import torch
import json
from unittest.mock import Mock, patch, mock_open
from hera.modules.immune_system import ImmuneSystem


@pytest.fixture
def mock_model():
    """Provide mock model instance."""
    return Mock()


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "immune": {
            "max_js_divergence": 0.15,
            "max_ppl_spike": 1.5,
            "max_cosine_drift": 0.05,
        }
    }


@pytest.fixture
def test_baseline_state():
    """Provide test baseline state."""
    return {
        "logits": torch.randn(1, 10, 50257),  # GPT-2 vocab size
        "ppl": 10.0,
        "activations": torch.randn(1, 10, 768),
    }


@pytest.fixture
def test_mutated_state():
    """Provide test mutated state."""
    return {
        "logits": torch.randn(1, 10, 50257),
        "ppl": 12.0,
        "activations": torch.randn(1, 10, 768),
    }


def test_immune_system_initialization(test_config, mock_model):
    """Test ImmuneSystem initialization."""
    # Test with probes file
    probes_data = ["probe1", "probe2", "probe3"]
    probes_json = json.dumps(probes_data)
    
    with patch("builtins.open", mock_open(read_data=probes_json)):
        immune = ImmuneSystem(test_config, mock_model)
        
        assert immune.cfg == test_config
        assert immune.model == mock_model
        assert immune.probes == probes_data
    
    # Test without probes file (file not found)
    with patch("builtins.open", side_effect=FileNotFoundError):
        immune = ImmuneSystem(test_config, mock_model)
        
        assert immune.probes == []  # Should be empty list on error
    
    # Test with invalid JSON
    with patch("builtins.open", mock_open(read_data="invalid json")):
        immune = ImmuneSystem(test_config, mock_model)
        
        assert immune.probes == []  # Should be empty list on JSON error


def test_verify_stable_state(test_config, mock_model, test_baseline_state, test_mutated_state):
    """Test verify method with stable state (all metrics within limits)."""
    immune = ImmuneSystem(test_config, mock_model)
    
    # Mock metrics to return values within limits
    with patch("hera.utils.metrics.js_divergence") as mock_js:
        with patch("hera.utils.metrics.cosine_drift") as mock_drift:
            mock_js.return_value = 0.1  # Below max_js_divergence (0.15)
            mock_drift.return_value = 0.03  # Below max_cosine_drift (0.05)
            
            # PPL ratio = 12.0 / 10.0 = 1.2, below max_ppl_spike (1.5)
            is_safe, reason, metrics = immune.verify(
                test_baseline_state, test_mutated_state, "test prompt"
            )
    
    # Should be safe
    assert is_safe is True
    assert reason == "Stable"
    
    # Check metrics
    assert "js_div" in metrics
    assert "ppl_ratio" in metrics
    assert "drift" in metrics
    assert metrics["js_div"] == 0.1
    assert metrics["ppl_ratio"] == pytest.approx(1.2)  # 12.0 / 10.0
    assert metrics["drift"] == 0.03


def test_verify_js_divergence_violation(test_config, mock_model, test_baseline_state, test_mutated_state):
    """Test verify method when JS divergence exceeds limit."""
    immune = ImmuneSystem(test_config, mock_model)
    
    with patch("hera.utils.metrics.js_divergence") as mock_js:
        with patch("hera.utils.metrics.cosine_drift") as mock_drift:
            mock_js.return_value = 0.2  # Above max_js_divergence (0.15)
            mock_drift.return_value = 0.03  # Within limit
            
            is_safe, reason, metrics = immune.verify(
                test_baseline_state, test_mutated_state, "test prompt"
            )
    
    # Should be rejected due to JS divergence
    assert is_safe is False
    assert "Output Divergence" in reason
    assert "0.2000" in reason  # Formatted with 4 decimal places
    
    # Metrics should still be returned
    assert metrics["js_div"] == 0.2


def test_verify_ppl_spike_violation(test_config, mock_model, test_baseline_state):
    """Test verify method when perplexity spike exceeds limit."""
    immune = ImmuneSystem(test_config, mock_model)
    
    # Create mutated state with high perplexity
    mutated_state = test_baseline_state.copy()
    mutated_state["ppl"] = 20.0  # PPL ratio = 20.0 / 10.0 = 2.0 > 1.5
    
    with patch("hera.utils.metrics.js_divergence") as mock_js:
        with patch("hera.utils.metrics.cosine_drift") as mock_drift:
            mock_js.return_value = 0.1  # Within limit
            mock_drift.return_value = 0.03  # Within limit
            
            is_safe, reason, metrics = immune.verify(
                test_baseline_state, mutated_state, "test prompt"
            )
    
    # Should be rejected due to PPL spike
    assert is_safe is False
    assert "PPL Spike" in reason
    assert "2.00" in reason  # Formatted with 2 decimal places
    
    # Check PPL ratio
    assert metrics["ppl_ratio"] == pytest.approx(2.0)


def test_verify_cosine_drift_violation(test_config, mock_model, test_baseline_state, test_mutated_state):
    """Test verify method when cosine drift exceeds limit."""
    immune = ImmuneSystem(test_config, mock_model)
    
    with patch("hera.utils.metrics.js_divergence") as mock_js:
        with patch("hera.utils.metrics.cosine_drift") as mock_drift:
            mock_js.return_value = 0.1  # Within limit
            mock_drift.return_value = 0.1  # Above max_cosine_drift (0.05)
            
            is_safe, reason, metrics = immune.verify(
                test_baseline_state, test_mutated_state, "test prompt"
            )
    
    # Should be rejected due to cosine drift
    assert is_safe is False
    assert "Internal Drift" in reason
    assert "0.1000" in reason  # Formatted with 4 decimal places
    
    # Check drift metric
    assert metrics["drift"] == 0.1


def test_verify_multiple_violations(test_config, mock_model, test_baseline_state):
    """Test verify method with multiple metric violations."""
    immune = ImmuneSystem(test_config, mock_model)
    
    # Create mutated state with high perplexity
    mutated_state = test_baseline_state.copy()
    mutated_state["ppl"] = 20.0  # Will cause PPL spike
    
    with patch("hera.utils.metrics.js_divergence") as mock_js:
        with patch("hera.utils.metrics.cosine_drift") as mock_drift:
            mock_js.return_value = 0.2  # JS divergence violation
            mock_drift.return_value = 0.1  # Cosine drift violation
            
            is_safe, reason, metrics = immune.verify(
                test_baseline_state, mutated_state, "test prompt"
            )
    
    # Should be rejected
    assert is_safe is False
    
    # Should report the first violation (JS divergence)
    # Order of checks: JS divergence -> PPL spike -> Cosine drift
    assert "Output Divergence" in reason
    
    # All metrics should be in the result
    assert metrics["js_div"] == 0.2
    assert metrics["ppl_ratio"] == pytest.approx(2.0)
    assert metrics["drift"] == 0.1


def test_verify_edge_cases(test_config, mock_model):
    """Test verify method with edge cases."""
    immune = ImmuneSystem(test_config, mock_model)
    
    # Test with zero perplexity baseline (avoid division by zero)
    baseline = {
        "logits": torch.randn(1, 5, 50257),
        "ppl": 0.0,
        "activations": torch.randn(1, 5, 768),
    }
    
    mutated = {
        "logits": torch.randn(1, 5, 50257),
        "ppl": 1.0,
        "activations": torch.randn(1, 5, 768),
    }
    
    with patch("hera.utils.metrics.js_divergence") as mock_js:
        with patch("hera.utils.metrics.cosine_drift") as mock_drift:
            mock_js.return_value = 0.1
            mock_drift.return_value = 0.03
            
            is_safe, reason, metrics = immune.verify(baseline, mutated, "test")
    
    # Should handle zero perplexity with +1e-6 in denominator
    assert "ppl_ratio" in metrics
    # PPL ratio should be large but finite
    assert metrics["ppl_ratio"] > 0
    
    # Test with identical states
    identical_state = baseline.copy()
    with patch("hera.utils.metrics.js_divergence") as mock_js:
        with patch("hera.utils.metrics.cosine_drift") as mock_drift:
            mock_js.return_value = 0.0  # Identical distributions
            mock_drift.return_value = 0.0  # Identical activations
            
            is_safe, reason, metrics = immune.verify(baseline, identical_state, "test")
    
    assert is_safe is True
    assert metrics["js_div"] == 0.0
    assert metrics["drift"] == 0.0
    assert metrics["ppl_ratio"] == pytest.approx(1.0)  # Same perplexity


def test_verify_different_batch_sizes(test_config, mock_model):
    """Test verify with different batch and sequence lengths."""
    immune = ImmuneSystem(test_config, mock_model)
    
    test_cases = [
        # (batch_size, seq_len)
        (1, 1),
        (1, 10),
        (2, 5),
        (4, 20),
    ]
    
    for batch_size, seq_len in test_cases:
        baseline = {
            "logits": torch.randn(batch_size, seq_len, 50257),
            "ppl": 10.0,
            "activations": torch.randn(batch_size, seq_len, 768),
        }
        
        mutated = {
            "logits": torch.randn(batch_size, seq_len, 50257),
            "ppl": 12.0,
            "activations": torch.randn(batch_size, seq_len, 768),
        }
        
        with patch("hera.utils.metrics.js_divergence") as mock_js:
            with patch("hera.utils.metrics.cosine_drift") as mock_drift:
                mock_js.return_value = 0.1
                mock_drift.return_value = 0.03
                
                is_safe, reason, metrics = immune.verify(baseline, mutated, "test")
        
        # Should work for all batch/seq combinations
        assert is_safe is True
        assert "js_div" in metrics
        assert "ppl_ratio" in metrics
        assert "drift" in metrics


def test_config_threshold_variations(mock_model):
    """Test how different config thresholds affect verification."""
    test_cases = [
        {
            "config": {
                "immune": {
                    "max_js_divergence": 0.05,  # Very strict
                    "max_ppl_spike": 1.1,       # Very strict
                    "max_cosine_drift": 0.01,   # Very strict
                }
            },
            "metrics": {"js": 0.1, "drift": 0.03, "ppl_ratio": 1.2},
            "expected_safe": False,  # All metrics exceed strict limits
        },
        {
            "config": {
                "immune": {
                    "max_js_divergence": 0.5,   # Very lenient
                    "max_ppl_spike": 3.0,       # Very lenient
                    "max_cosine_drift": 0.5,    # Very lenient
                }
            },
            "metrics": {"js": 0.1, "drift": 0.03, "ppl_ratio": 1.2},
            "expected_safe": True,  # All metrics within lenient limits
        },
        {
            "config": {
                "immune": {
                    "max_js_divergence": 0.2,   # JS limit not exceeded
                    "max_ppl_spike": 1.0,       # PPL limit exceeded
                    "max_cosine_drift": 0.1,    # Drift limit not exceeded
                }
            },
            "metrics": {"js": 0.1, "drift": 0.03, "ppl_ratio": 1.2},
            "expected_safe": False,  # PPL ratio 1.2 > 1.0
        },
    ]
    
    for test_case in test_cases:
        immune = ImmuneSystem(test_case["config"], mock_model)
        
        baseline = {
            "logits": torch.randn(1, 5, 50257),
            "ppl": 10.0,
            "activations": torch.randn(1, 5, 768),
        }
        
        mutated = {
            "logits": torch.randn(1, 5, 50257),
            "ppl": 10.0 * test_case["metrics"]["ppl_ratio"],  # Apply ratio
            "activations": torch.randn(1, 5, 768),
        }
        
        with patch("hera.utils.metrics.js_divergence") as mock_js:
            with patch("hera.utils.metrics.cosine_drift") as mock_drift:
                mock_js.return_value = test_case["metrics"]["js"]
                mock_drift.return_value = test_case["metrics"]["drift"]
                
                is_safe, reason, metrics = immune.verify(baseline, mutated, "test")
        
        assert is_safe == test_case["expected_safe"], f"Failed for config: {test_case['config']}"


def test_verify_input_text_usage(test_config, mock_model, test_baseline_state, test_mutated_state):
    """Test that input text is passed through (for future extensions)."""
    immune = ImmuneSystem(test_config, mock_model)
    
    test_prompt = "This is a specific test prompt with unique content"
    
    with patch("hera.utils.metrics.js_divergence") as mock_js:
        with patch("hera.utils.metrics.cosine_drift") as mock_drift:
            mock_js.return_value = 0.1
            mock_drift.return_value = 0.03
            
            is_safe, reason, metrics = immune.verify(
                test_baseline_state, test_mutated_state, test_prompt
            )
    
    # Currently input_text is not used, but the method accepts it
    # This test ensures the parameter is properly handled
    assert is_safe is True
    
    # In the future, if input_text is used for analysis,
    # we could add assertions here


def test_immune_system_probes_usage(test_config, mock_model):
    """Test that probes are loaded and could be used (for future extensions)."""
    probes_data = [
        "The capital of France is Paris.",
        "Python is a programming language.",
        "Machine learning involves algorithms that improve automatically."
    ]
    
    with patch("builtins.open", mock_open(read_data=json.dumps(probes_data))):
        immune = ImmuneSystem(test_config, mock_model)
        
        # Verify probes are loaded
        assert len(immune.probes) == 3
        assert immune.probes[0] == "The capital of France is Paris."
        
        # Currently probes are not used in verify method,
        # but they are loaded and available for future extensions
        # This test ensures the infrastructure is in place


def test_error_handling_in_metrics(test_config, mock_model, test_baseline_state, test_mutated_state):
    """Test error handling when metric functions fail."""
    immune = ImmuneSystem(test_config, mock_model)
    
    # Test when js_divergence raises an exception
    with patch("hera.utils.metrics.js_divergence", side_effect=RuntimeError("JS calculation failed")):
        with pytest.raises(RuntimeError, match="JS calculation failed"):
            immune.verify(test_baseline_state, test_mutated_state, "test")
    
    # Test when cosine_drift raises an exception
    with patch("hera.utils.metrics.js_divergence", return_value=0.1):
        with patch("hera.utils.metrics.cosine_drift", side_effect=ValueError("Invalid tensor shape")):
            with pytest.raises(ValueError, match="Invalid tensor shape"):
                immune.verify(test_baseline_state, test_mutated_state, "test")