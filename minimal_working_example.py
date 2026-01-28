#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Working Example (MVE) for H.E.R.A.-R

This is the simplest possible example that demonstrates
H.E.R.A.-R's core functionality working end-to-end.
"""

import sys
import os

# Fix encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import yaml
import torch
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available."""
    print("=" * 60)
    print("Checking Dependencies")
    print("=" * 60)
    
    dependencies = {
        "torch": "PyTorch",
        "transformer_lens": "TransformerLens",
        "sae_lens": "SAE Lens",
        "yaml": "PyYAML",
    }
    
    all_available = True
    
    for module, name in dependencies.items():
        try:
            if module == "yaml":
                import yaml
            else:
                __import__(module)
            print(f"[OK] {name} available")
        except ImportError as e:
            print(f"[FAIL] {name} not available: {e}")
            all_available = False
    
    return all_available

def create_minimal_config():
    """Create a minimal configuration file."""
    config = {
        "experiment": {
            "name": "mve-test",
            "device": "cpu",  # Use CPU for simplicity
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
        },
        "immune": {
            "max_js_divergence": 0.15,
            "max_ppl_spike": 1.5,
            "max_cosine_drift": 0.05,
        },
        "logging": {
            "verbose": True,
            "save_dir": "mve_logs",
        },
    }
    
    config_path = Path("config_mve.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nCreated minimal config at: {config_path}")
    return config_path

def test_basic_components():
    """Test basic components without full engine."""
    print("\n" + "=" * 60)
    print("Testing Basic Components")
    print("=" * 60)
    
    try:
        # Test 1: Can we import the modules?
        print("\n1. Testing module imports...")
        from hera.utils.logger import HeraLogger
        print("[OK] Can import HeraLogger")
        
        # Test 2: Can we create a simple config?
        print("\n2. Testing configuration...")
        test_config = {
            "experiment": {"device": "cpu"},
            "logging": {"verbose": False}
        }
        logger = HeraLogger(test_config)
        print("[OK] Can create logger with config")
        
        # Test 3: Test metrics functions
        print("\n3. Testing metrics calculations...")
        from hera.utils.metrics import js_divergence
        
        # Create dummy logits
        logits1 = torch.randn(1, 5, 50257)
        logits2 = logits1 + torch.randn_like(logits1) * 0.1
        
        js = js_divergence(logits1, logits2)
        print(f"[OK] JS divergence calculation: {js:.6f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_minimal_evolution():
    """Run a minimal evolution example."""
    print("\n" + "=" * 60)
    print("Running Minimal Evolution")
    print("=" * 60)
    
    config_path = create_minimal_config()
    
    try:
        print("\n1. Importing HeraEngine...")
        from hera.core.engine import HeraEngine
        
        print("2. Initializing engine...")
        engine = HeraEngine(str(config_path))
        
        print("3. Engine status:")
        status = engine.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        print("\n4. Running single evolution step...")
        test_prompt = "Artificial intelligence"
        print(f"   Prompt: '{test_prompt}'")
        
        success = engine.evolve(test_prompt)
        
        if success:
            print("   [SUCCESS] Evolution accepted!")
        else:
            print("   [REJECTED] Evolution rejected by immune system")
        
        print("\n5. Checking metrics...")
        evo_metrics = engine.evo_layer.get_metrics()
        print(f"   Evolution updates: {evo_metrics.get('total_updates', 0)}")
        print(f"   Success rate: {evo_metrics.get('success_rate', 0):.1%}")
        
        immune_stats = engine.immune.get_stats()
        print(f"   Safety checks: {immune_stats.get('total_checks', 0)}")
        print(f"   Rejection rate: {immune_stats.get('rejection_rate', 0):.1%}")
        
        print("\n6. Testing reset functionality...")
        engine.reset()
        print("   [OK] Engine reset successfully")
        
        print("\n7. Clean shutdown...")
        engine.shutdown()
        print("   [OK] Shutdown complete")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        print("\nThis might be because:")
        print("1. Dependencies are not installed")
        print("2. Python path is not set correctly")
        print("3. There's a circular import issue")
        return False
        
    except Exception as e:
        print(f"[FAIL] Evolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up config file
        if config_path.exists():
            config_path.unlink()
        
        # Clean up logs
        log_dir = Path("mve_logs")
        if log_dir.exists():
            import shutil
            shutil.rmtree(log_dir)

def main():
    """Run the minimal working example."""
    print("H.E.R.A.-R Minimal Working Example")
    print("=" * 60)
    
    print("\nThis example demonstrates the absolute minimum required")
    print("to get H.E.R.A.-R working end-to-end.")
    print("\nSteps:")
    print("1. Check dependencies")
    print("2. Test basic components")
    print("3. Run minimal evolution")
    
    # Check dependencies
    if not check_dependencies():
        print("\n[WARN] Some dependencies missing.")
        print("You may need to install them:")
        print("  pip install torch transformer-lens sae-lens pyyaml")
        print("\nContinue anyway? Some tests may fail.")
        response = input("Continue? (y/n): ").lower().strip()
        if response != 'y':
            print("Exiting.")
            return False
    
    # Test basic components
    if not test_basic_components():
        print("\n[WARN] Basic component tests failed.")
        print("There may be issues with the code structure.")
        response = input("Continue to evolution test? (y/n): ").lower().strip()
        if response != 'y':
            print("Exiting.")
            return False
    
    # Run minimal evolution
    print("\n" + "=" * 60)
    print("Starting Minimal Evolution Test")
    print("=" * 60)
    
    success = run_minimal_evolution()
    
    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] Minimal Working Example PASSED!")
        print("\nH.E.R.A.-R is working correctly.")
        print("The core functionality has been verified.")
    else:
        print("[FAIL] Minimal Working Example FAILED.")
        print("\nThere are issues that need to be addressed.")
        print("Check the error messages above for details.")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[STOP] Interrupted by user.")
        exit(1)