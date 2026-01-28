#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Comprehensive Test for Enhanced H.E.R.A.-R Features

This script tests all the enhanced features including:
1. Type hints and error handling
2. Enhanced evolutionary layer with multiple strategies
3. Enhanced immune system with comprehensive safety checks
4. Engine status and reset functionality
"""

import yaml
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from hera.core.engine import HeraEngine
from hera.modules.evolutionary_layer_enhanced import EvolutionStrategy


def create_test_config() -> Dict[str, Any]:
    """Create test configuration."""
    return {
        "experiment": {
            "name": "enhanced-features-test",
            "device": "cpu",
            "seed": 42,
        },
        "model": {
            "name": "gpt2-small",
        },
        "sae": {
            "release": "gpt2-small-res-jb",
            "id": "blocks.8.hook_resid_pre",
            "threshold": 2.0,
        },
        "evolution": {
            "target_layers": [8],
            "learning_rate": 0.01,
            "max_rank": 1,
            "top_k_features": 3,
            "strategy": "hebbian",
        },
        "immune": {
            "max_js_divergence": 0.15,
            "max_ppl_spike": 1.5,
            "max_cosine_drift": 0.05,
            "max_entropy_change": 0.5,
            "max_top_k_disagreement": 0.3,
        },
        "logging": {
            "verbose": True,
            "save_dir": "test_logs",
        },
    }


def test_type_hints_and_validation():
    """Test type hints and input validation."""
    print("=" * 60)
    print("Testing Type Hints and Validation")
    print("=" * 60)
    
    config = create_test_config()
    
    # Test 1: Invalid config path
    print("\n1. Testing invalid config path...")
    try:
        engine = HeraEngine("non_existent_config.yaml")
        print("   [FAIL] Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"   [OK] Correctly raised FileNotFoundError: {e}")
    except Exception as e:
        print(f"   [FAIL] Wrong exception type: {type(e).__name__}: {e}")
        return False
    
    # Test 2: Missing required config section
    print("\n2. Testing missing config section...")
    invalid_config = config.copy()
    del invalid_config["model"]  # Remove required section
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(invalid_config, f)
        invalid_config_path = f.name
    
    try:
        engine = HeraEngine(invalid_config_path)
        print("   [FAIL] Should have raised KeyError")
        return False
    except KeyError as e:
        print(f"   [OK] Correctly raised KeyError: {e}")
    except Exception as e:
        print(f"   [FAIL] Wrong exception type: {type(e).__name__}: {e}")
        return False
    finally:
        Path(invalid_config_path).unlink()
    
    # Test 3: Valid configuration
    print("\n3. Testing valid configuration...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        engine = HeraEngine(config_path)
        print("   [OK] Engine initialized successfully")
        
        # Test type hints
        print("\n4. Testing type hints...")
        status = engine.get_status()
        assert isinstance(status, dict), "get_status() should return dict"
        assert "model" in status, "Status should contain model info"
        assert "device" in status, "Status should contain device info"
        print("   [OK] Type hints working correctly")
        
        engine.shutdown()
        return True
        
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(config_path).unlink()


def test_evolution_strategies():
    """Test different evolution strategies."""
    print("\n" + "=" * 60)
    print("Testing Evolution Strategies")
    print("=" * 60)
    
    config = create_test_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        engine = HeraEngine(config_path)
        
        # Test 1: Hebbian strategy (default)
        print("\n1. Testing Hebbian strategy...")
        engine.evo_layer.set_strategy(EvolutionStrategy.HEBBIAN)
        
        test_prompt = "Artificial intelligence is"
        success = engine.evolve(test_prompt)
        
        metrics = engine.evo_layer.get_metrics()
        print(f"   Success: {success}")
        print(f"   Metrics: {metrics}")
        
        # Test 2: Random strategy
        print("\n2. Testing Random strategy...")
        engine.evo_layer.set_strategy(EvolutionStrategy.RANDOM)
        
        success = engine.evolve("Machine learning algorithms")
        print(f"   Success: {success}")
        
        # Test 3: Mixed strategy
        print("\n3. Testing Mixed strategy...")
        engine.evo_layer.set_strategy(EvolutionStrategy.MIXED)
        
        success = engine.evolve("Deep neural networks")
        print(f"   Success: {success}")
        
        # Test 4: Get feature contributions
        print("\n4. Testing feature contributions...")
        tokens = engine.model.to_tokens("Test prompt for features")
        contributions = engine.evo_layer.get_feature_contributions(tokens)
        print(f"   Feature contributions: {len(contributions)} features")
        
        # Test 5: Get weight change
        weight_change = engine.evo_layer.get_weight_change()
        print(f"   Weight change norm: {weight_change.norm().item():.6f}")
        
        engine.shutdown()
        return True
        
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(config_path).unlink()


def test_immune_system_enhancements():
    """Test enhanced immune system features."""
    print("\n" + "=" * 60)
    print("Testing Enhanced Immune System")
    print("=" * 60)
    
    config = create_test_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        engine = HeraEngine(config_path)
        
        # Test 1: Get immune system stats
        print("\n1. Testing immune system statistics...")
        stats = engine.immune.get_stats()
        print(f"   Initial stats: {stats}")
        
        # Test 2: Run evolution and check stats
        print("\n2. Testing evolution with safety checks...")
        
        test_prompts = [
            "The future of AI research",
            "Neural networks can learn",
            "Language models understand",
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n   Prompt {i+1}: '{prompt}'")
            success = engine.evolve(prompt)
            print(f"   Result: {'[OK] Accepted' if success else '[STOP] Rejected'}")
            
            # Check updated stats
            stats = engine.immune.get_stats()
            print(f"   Stats after: {stats}")
        
        # Test 3: Add and test reference probes
        print("\n3. Testing reference probes...")
        
        # Create test probes directory
        probe_dir = Path("data")
        probe_dir.mkdir(exist_ok=True)
        
        # Add some test probes
        engine.immune.add_probe(
            "The capital of France is",
            "Paris"
        )
        engine.immune.add_probe(
            "Two plus two equals",
            "four"
        )
        
        # Run probe tests
        probe_results = engine.immune.run_probe_tests()
        print(f"   Probe test results: {probe_results}")
        
        engine.shutdown()
        return True
        
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(config_path).unlink()
        # Clean up test directory
        if Path("test_logs").exists():
            shutil.rmtree("test_logs")
        if Path("data").exists():
            shutil.rmtree("data")


def test_engine_functionality():
    """Test engine reset and status functionality."""
    print("\n" + "=" * 60)
    print("Testing Engine Functionality")
    print("=" * 60)
    
    config = create_test_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        engine = HeraEngine(config_path)
        
        # Test 1: Initial status
        print("\n1. Testing initial status...")
        status = engine.get_status()
        print(f"   Initial status: {status}")
        
        # Test 2: Run some evolutions
        print("\n2. Running evolutions...")
        for prompt in ["AI", "ML", "DL"]:
            engine.evolve(prompt)
        
        # Test 3: Check status after evolutions
        status = engine.get_status()
        print(f"   Status after evolutions: {status}")
        
        # Test 4: Reset engine
        print("\n3. Testing engine reset...")
        engine.reset()
        
        status = engine.get_status()
        print(f"   Status after reset: {status}")
        
        # Test 5: Verify reset worked
        print("\n4. Verifying reset...")
        # Evolution layer should have zero updates after reset
        evo_metrics = engine.evo_layer.get_metrics()
        assert evo_metrics["total_updates"] == 0, "Reset should clear updates"
        print(f"   [OK] Reset verified: {evo_metrics}")
        
        engine.shutdown()
        return True
        
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(config_path).unlink()


def test_error_handling():
    """Test error handling and recovery."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    config = create_test_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        engine = HeraEngine(config_path)
        
        # Test 1: Empty input text
        print("\n1. Testing empty input text...")
        try:
            engine.evolve("")
            print("   [FAIL] Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"   [OK] Correctly raised ValueError: {e}")
        
        # Test 2: Whitespace-only input
        print("\n2. Testing whitespace-only input...")
        try:
            engine.evolve("   ")
            print("   [FAIL] Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"   [OK] Correctly raised ValueError: {e}")
        
        # Test 3: Invalid evolution (should handle gracefully)
        print("\n3. Testing graceful error handling...")
        # This should not crash even if something goes wrong
        try:
            # Try to evolve with a very long input that might cause issues
            long_text = "A" * 10000  # Very long text
            success = engine.evolve(long_text)
            print(f"   [OK] Handled gracefully: {'Accepted' if success else 'Rejected'}")
        except Exception as e:
            print(f"   [FAIL] Should have handled gracefully: {e}")
            return False
        
        engine.shutdown()
        return True
        
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(config_path).unlink()


def main():
    """Run all tests."""
    print("H.E.R.A.-R Enhanced Features Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Type Hints & Validation", test_type_hints_and_validation),
        ("Evolution Strategies", test_evolution_strategies),
        ("Immune System", test_immune_system_enhancements),
        ("Engine Functionality", test_engine_functionality),
        ("Error Handling", test_error_handling),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        print("-" * 40)
        
        try:
            passed = test_func()
            test_results[test_name] = "PASS" if passed else "FAIL"
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"   [FAIL] Test crashed: {e}")
            test_results[test_name] = "CRASH"
            all_passed = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        status_icon = "[OK]" if result == "PASS" else "[FAIL]"
        print(f"{status_icon} {test_name}: {result}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All tests passed! Enhanced features are working correctly.")
    else:
        print("[WARN] Some tests failed. Please check the output above.")
    
    # Clean up
    if Path("test_logs").exists():
        shutil.rmtree("test_logs")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)