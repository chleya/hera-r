# -*- coding: utf-8 -*-
"""Performance tests for H.E.R.A.-R."""

import pytest
import torch
import time
import psutil
import os
from unittest.mock import Mock, patch


@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Performance test suite."""
    
    def test_metrics_performance(self):
        """Test performance of metrics calculations."""
        # Create large tensors for testing
        batch_size = 8
        seq_len = 128
        vocab_size = 50257
        hidden_size = 768
        
        p_logits = torch.randn(batch_size, seq_len, vocab_size)
        q_logits = torch.randn(batch_size, seq_len, vocab_size)
        activations_a = torch.randn(batch_size, seq_len, hidden_size)
        activations_b = torch.randn(batch_size, seq_len, hidden_size)
        
        # Import here to avoid dependency issues
        from hera.utils.metrics import js_divergence, cosine_drift
        
        # Time JS divergence
        start_time = time.time()
        js_result = js_divergence(p_logits, q_logits)
        js_time = time.time() - start_time
        
        # Time cosine drift
        start_time = time.time()
        drift_result = cosine_drift(activations_a, activations_b)
        drift_time = time.time() - start_time
        
        print(f"\nPerformance results:")
        print(f"  JS divergence: {js_time:.4f}s (result: {js_result:.6f})")
        print(f"  Cosine drift: {drift_time:.4f}s (result: {drift_result:.6f})")
        
        # Performance assertions
        assert js_time < 1.0, f"JS divergence too slow: {js_time:.4f}s"
        assert drift_time < 0.5, f"Cosine drift too slow: {drift_time:.4f}s"
        
        # Memory usage check
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"  Memory usage: {memory_mb:.1f} MB")
        assert memory_mb < 2000, f"Memory usage too high: {memory_mb:.1f} MB"
    
    def test_evolution_performance(self):
        """Test performance of evolutionary operations."""
        # Mock setup for performance testing
        batch_size = 4
        seq_len = 64
        hidden_size = 768
        num_features = 100
        
        # Create mock objects
        mock_model = Mock()
        mock_sae = Mock()
        
        # Mock SAE encode returns feature activations
        feature_acts = torch.ones(batch_size, seq_len, num_features) * 3.0
        mock_sae.encode.return_value = feature_acts
        
        # Mock decode direction
        mock_sae.decode_direction.return_value = torch.randn(hidden_size)
        
        # Mock model components
        mock_linear = Mock()
        mock_linear.W = Mock()
        mock_linear.W.data = torch.randn(hidden_size, hidden_size)
        
        mock_mlp = Mock()
        mock_mlp.W_out = mock_linear
        
        mock_block = Mock()
        mock_block.mlp = mock_mlp
        
        mock_model.blocks = [Mock() for _ in range(12)]
        mock_model.blocks[8] = mock_block
        
        # Import and test
        from hera.modules.evolutionary_layer import NeuroEvolutionaryLayer
        
        config = {
            "evolution": {
                "top_k_features": 3,
                "learning_rate": 0.01,
            },
            "sae": {
                "threshold": 2.0,
            },
        }
        
        # Create layer
        layer = NeuroEvolutionaryLayer(mock_model, 8, mock_sae, config)
        
        # Test propose performance
        activations = torch.randn(batch_size, seq_len, hidden_size)
        
        start_time = time.time()
        delta_w = layer.propose(activations)
        propose_time = time.time() - start_time
        
        print(f"\nEvolution performance:")
        print(f"  Propose time: {propose_time:.4f}s")
        
        assert propose_time < 0.1, f"Propose too slow: {propose_time:.4f}s"
        assert delta_w is not None
        assert delta_w.shape == (hidden_size, hidden_size)
        
        # Test commit/rollback performance
        start_time = time.time()
        layer.commit(delta_w)
        commit_time = time.time() - start_time
        
        start_time = time.time()
        layer.rollback(delta_w)
        rollback_time = time.time() - start_time
        
        print(f"  Commit time: {commit_time:.6f}s")
        print(f"  Rollback time: {rollback_time:.6f}s")
        
        assert commit_time < 0.01, f"Commit too slow: {commit_time:.6f}s"
        assert rollback_time < 0.01, f"Rollback too slow: {rollback_time:.6f}s"
    
    def test_immune_system_performance(self):
        """Test performance of immune system verification."""
        from hera.modules.immune_system import ImmuneSystem
        
        # Mock setup
        mock_model = Mock()
        config = {
            "immune": {
                "max_js_divergence": 0.15,
                "max_ppl_spike": 1.5,
                "max_cosine_drift": 0.05,
            }
        }
        
        immune = ImmuneSystem(config, mock_model)
        
        # Create test states
        batch_size = 2
        seq_len = 32
        vocab_size = 50257
        hidden_size = 768
        
        baseline = {
            "logits": torch.randn(batch_size, seq_len, vocab_size),
            "ppl": 10.0,
            "activations": torch.randn(batch_size, seq_len, hidden_size),
        }
        
        mutated = {
            "logits": torch.randn(batch_size, seq_len, vocab_size),
            "ppl": 12.0,
            "activations": torch.randn(batch_size, seq_len, hidden_size),
        }
        
        # Mock metrics
        with patch("hera.utils.metrics.js_divergence") as mock_js:
            with patch("hera.utils.metrics.cosine_drift") as mock_drift:
                mock_js.return_value = 0.1
                mock_drift.return_value = 0.03
                
                # Time verification
                start_time = time.time()
                is_safe, reason, metrics = immune.verify(baseline, mutated, "test")
                verify_time = time.time() - start_time
        
        print(f"\nImmune system performance:")
        print(f"  Verify time: {verify_time:.6f}s")
        
        assert verify_time < 0.05, f"Verify too slow: {verify_time:.6f}s"
        assert is_safe is True
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("seq_len", [16, 32, 64, 128])
    def test_scaling_performance(self, batch_size, seq_len):
        """Test how performance scales with batch and sequence length."""
        # Skip large combinations in normal runs
        if batch_size * seq_len > 256:
            pytest.skip("Skipping large test for performance")
        
        hidden_size = 768
        vocab_size = 50257
        
        # Create test data
        p_logits = torch.randn(batch_size, seq_len, vocab_size)
        q_logits = torch.randn(batch_size, seq_len, vocab_size)
        activations = torch.randn(batch_size, seq_len, hidden_size)
        
        from hera.utils.metrics import js_divergence, cosine_drift
        
        # Time operations
        start_time = time.time()
        js_result = js_divergence(p_logits, q_logits)
        js_time = time.time() - start_time
        
        start_time = time.time()
        drift_result = cosine_drift(activations, activations)  # Same tensor
        drift_time = time.time() - start_time
        
        # Log results
        total_elements = batch_size * seq_len
        print(f"\nScaling test (batch={batch_size}, seq={seq_len}, elements={total_elements}):")
        print(f"  JS divergence: {js_time:.6f}s ({js_time/total_elements:.6f}s per element)")
        print(f"  Cosine drift: {drift_time:.6f}s ({drift_time/total_elements:.6f}s per element)")
        
        # Scaling should be roughly linear
        max_time_per_element = 0.001  # 1ms per element max
        assert js_time < total_elements * max_time_per_element, \
            f"JS divergence scaling poor: {js_time:.6f}s for {total_elements} elements"
    
    def test_memory_efficiency(self):
        """Test memory efficiency of operations."""
        import gc
        
        # Track initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Create and process large tensors
        large_tensor = torch.randn(100, 100, 768)  ~61MB
        
        from hera.utils.metrics import cosine_drift
        
        # Process and immediately discard
        result = cosine_drift(large_tensor, large_tensor)
        
        # Force garbage collection
        del large_tensor
        gc.collect()
        
        # Check memory didn't leak
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory efficiency test:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        
        # Allow some overhead but not massive leaks
        assert memory_increase < 100, f"Memory leak detected: {memory_increase:.1f} MB increase"
        
        # Result should be 0 for identical tensors
        assert abs(result) < 1e-6, f"Expected near-zero drift, got {result:.6f}"


@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for critical operations."""
    
    def benchmark_js_divergence(self, benchmark):
        """Benchmark JS divergence calculation."""
        from hera.utils.metrics import js_divergence
        
        # Setup
        p_logits = torch.randn(2, 50, 50257)
        q_logits = torch.randn(2, 50, 50257)
        
        # Benchmark
        result = benchmark(js_divergence, p_logits, q_logits)
        
        # Verify result
        assert 0 <= result <= 1
    
    def benchmark_cosine_drift(self, benchmark):
        """Benchmark cosine drift calculation."""
        from hera.utils.metrics import cosine_drift
        
        # Setup
        a = torch.randn(2, 50, 768)
        b = torch.randn(2, 50, 768)
        
        # Benchmark
        result = benchmark(cosine_drift, a, b)
        
        # Verify result
        assert 0 <= result <= 2  # Cosine drift can be up to 2
    
    def benchmark_feature_encoding(self, benchmark):
        """Benchmark feature encoding (mocked)."""
        # Mock SAE encode
        def mock_encode(activations):
            # Simulate encoding operation
            return torch.randn(*activations.shape[:-1], 100)
        
        # Setup
        activations = torch.randn(2, 32, 768)
        
        # Benchmark
        result = benchmark(mock_encode, activations)
        
        # Verify
        assert result.shape == (2, 32, 100)