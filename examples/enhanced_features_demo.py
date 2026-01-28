#!/usr/bin/env python3
"""
Enhanced Features Demo for H.E.R.A.-R

This script demonstrates the enhanced features and improved code quality
in the latest version of H.E.R.A.-R.
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from hera.core.engine import HeraEngine
from hera.modules.evolutionary_layer_enhanced import EvolutionStrategy


def demonstrate_type_hints_and_error_handling():
    """Demonstrate improved type hints and error handling."""
    print("=" * 60)
    print("Demonstrating Type Hints and Error Handling")
    print("=" * 60)
    
    # Create a minimal config for demonstration
    config = {
        "experiment": {
            "name": "demo-experiment",
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
            "strategy": "hebbian",  # New: evolution strategy
        },
        "immune": {
            "max_js_divergence": 0.15,
            "max_ppl_spike": 1.5,
            "max_cosine_drift": 0.05,
        },
        "logging": {
            "verbose": True,
            "save_dir": "logs/demo",
        },
    }
    
    # Save config
    config_path = Path("demo_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Demonstrate initialization with error handling
        print("\n1. Initializing HeraEngine with validation...")
        engine = HeraEngine(str(config_path))
        print("   ✅ Engine initialized successfully")
        
        # Demonstrate type hints in action
        print("\n2. Type hints demonstration:")
        print(f"   Engine type: {type(engine).__name__}")
        print(f"   Model type: {type(engine.model).__name__}")
        print(f"   Config type: {type(engine.cfg).__name__}")
        
        # Demonstrate error handling
        print("\n3. Error handling demonstration:")
        try:
            # Try to evolve with empty text (should raise ValueError)
            engine.evolve("")
            print("   ❌ Should have raised ValueError for empty text")
        except ValueError as e:
            print(f"   ✅ Correctly raised ValueError: {e}")
        
        # Demonstrate enhanced features
        print("\n4. Enhanced features demonstration:")
        
        # Get engine status
        status = engine.get_status()
        print(f"   Engine status: {status}")
        
        # Demonstrate evolution with different strategies
        print("\n5. Evolution strategies demonstration:")
        
        test_prompts = [
            "The future of artificial intelligence",
            "Machine learning algorithms can",
            "Deep neural networks are",
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n   Prompt {i+1}: '{prompt}'")
            try:
                success = engine.evolve(prompt)
                print(f"   Result: {'✅ Committed' if success else '🛑 Rejected'}")
                
                # Get evolution metrics
                metrics = engine.evo_layer.get_metrics()
                print(f"   Evolution metrics: {metrics.get('success_rate', 0):.1%} success rate")
                
            except Exception as e:
                print(f"   Error: {e}")
        
        # Demonstrate reset functionality
        print("\n6. Reset functionality demonstration:")
        engine.reset()
        print("   ✅ Engine reset successfully")
        
        # Clean shutdown
        print("\n7. Clean shutdown demonstration:")
        engine.shutdown()
        print("   ✅ Engine shutdown successfully")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if config_path.exists():
            config_path.unlink()
        
        # Clean up logs directory
        log_dir = Path("logs/demo")
        if log_dir.exists():
            import shutil
            shutil.rmtree(log_dir)


def demonstrate_evolution_strategies():
    """Demonstrate different evolution strategies."""
    print("\n" + "=" * 60)
    print("Demonstrating Evolution Strategies")
    print("=" * 60)
    
    # Create config with different strategies
    strategies = [
        ("Hebbian", EvolutionStrategy.HEBBIAN),
        ("Random", EvolutionStrategy.RANDOM),
        ("Mixed", EvolutionStrategy.MIXED),
    ]
    
    for strategy_name, strategy in strategies:
        print(f"\nStrategy: {strategy_name}")
        
        config = {
            "experiment": {"name": f"strategy-{strategy_name.lower()}", "device": "cpu", "seed": 42},
            "model": {"name": "gpt2-small"},
            "sae": {"release": "gpt2-small-res-jb", "id": "blocks.8.hook_resid_pre", "threshold": 2.0},
            "evolution": {
                "target_layers": [8],
                "learning_rate": 0.01,
                "top_k_features": 3,
                "strategy": strategy.value,
            },
            "immune": {
                "max_js_divergence": 0.15,
                "max_ppl_spike": 1.5,
                "max_cosine_drift": 0.05,
            },
            "logging": {"verbose": False, "save_dir": f"logs/strategy-{strategy_name.lower()}"},
        }
        
        config_path = Path(f"config_{strategy_name.lower()}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            engine = HeraEngine(str(config_path))
            
            # Run a few evolution steps
            prompts = ["AI research focuses on", "Neural networks learn by", "Language models understand"]
            results = []
            
            for prompt in prompts:
                success = engine.evolve(prompt)
                results.append(success)
            
            success_rate = sum(results) / len(results) if results else 0
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Evolution metrics: {engine.evo_layer.get_metrics()}")
            
            engine.shutdown()
            
        except Exception as e:
            print(f"  Error: {e}")
        
        finally:
            if config_path.exists():
                config_path.unlink()


def demonstrate_code_quality_features():
    """Demonstrate code quality improvements."""
    print("\n" + "=" * 60)
    print("Demonstrating Code Quality Features")
    print("=" * 60)
    
    print("\n1. Type Hints:")
    print("   - All functions have explicit type annotations")
    print("   - Classes use dataclasses for structured data")
    print("   - Enums for categorical values (EvolutionStrategy)")
    
    print("\n2. Error Handling:")
    print("   - Comprehensive try-except blocks")
    print("   - Specific exception types (ValueError, RuntimeError)")
    print("   - Clean error messages with context")
    
    print("\n3. Documentation:")
    print("   - Google-style docstrings")
    print("   - Args/Returns/Raises sections")
    print("   - Examples and usage notes")
    
    print("\n4. Configuration Validation:")
    print("   - Automatic config validation on initialization")
    print("   - Required parameter checking")
    print("   - Type and range validation")
    
    print("\n5. Metrics and Monitoring:")
    print("   - Evolution metrics tracking")
    print("   - Success/failure statistics")
    print("   - History of updates")


def main():
    """Main demonstration function."""
    print("H.E.R.A.-R Enhanced Features Demonstration")
    print("=" * 60)
    
    try:
        # Check if dependencies are available
        import transformer_lens
        import sae_lens
        
        # Run demonstrations
        demonstrate_type_hints_and_error_handling()
        demonstrate_evolution_strategies()
        demonstrate_code_quality_features()
        
        print("\n" + "=" * 60)
        print("Demonstration Complete!")
        print("=" * 60)
        print("\nSummary of enhanced features:")
        print("1. ✅ Type hints throughout codebase")
        print("2. ✅ Comprehensive error handling")
        print("3. ✅ Multiple evolution strategies")
        print("4. ✅ Enhanced metrics and monitoring")
        print("5. ✅ Better configuration validation")
        print("6. ✅ Improved documentation")
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("  pip install transformer-lens sae-lens")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()