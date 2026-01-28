#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Core Functionality Test for H.E.R.A.-R

This script tests the absolute core functionality:
1. Can we load a model?
2. Can we load SAE?
3. Can we extract features?
4. Can we propose weight updates?
5. Can we verify safety?
"""

import sys
import os

# Fix encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """Test if we can load a transformer model."""
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    try:
        from transformer_lens import HookedTransformer
        
        print("1. Loading GPT-2 small...")
        model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
        model.eval()
        
        print(f"   [OK] Model loaded: {model.cfg.model_name}")
        print(f"   [OK] Device: {model.cfg.device}")
        print(f"   [OK] Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test inference
        print("\n2. Testing inference...")
        text = "Hello world"
        tokens = model.to_tokens(text)
        logits = model(tokens)
        
        print(f"   [OK] Tokenization: {tokens.shape}")
        print(f"   [OK] Inference: {logits.shape}")
        
        return model
        
    except Exception as e:
        print(f"   [FAIL] Failed to load model: {e}")
        return None

def test_sae_loading():
    """Test if we can load SAE."""
    print("\n" + "=" * 60)
    print("Testing SAE Loading")
    print("=" * 60)
    
    try:
        from sae_lens import SAE
        
        print("1. Loading SAE...")
        sae, cfg, sparsity = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.8.hook_resid_pre",
            device="cpu"
        )
        sae.eval()
        
        print(f"   [OK] SAE loaded: {cfg['hook_name']}")
        print(f"   [OK] Device: {sae.cfg.device}")
        print(f"   [OK] Features: {sae.cfg.d_sae}")
        
        # Test encoding
        print("\n2. Testing SAE encoding...")
        test_input = torch.randn(1, 10, 768)  # (batch, seq, d_model)
        with torch.no_grad():
            features = sae.encode(test_input)
        
        print(f"   [OK] Input shape: {test_input.shape}")
        print(f"   [OK] Features shape: {features.shape}")
        
        return sae
        
    except Exception as e:
        print(f"   [FAIL] Failed to load SAE: {e}")
        return None

def test_feature_extraction(model, sae):
    """Test feature extraction from model activations."""
    print("\n" + "=" * 60)
    print("Testing Feature Extraction")
    print("=" * 60)
    
    try:
        print("1. Extracting activations...")
        
        # Create hook to capture activations
        cache = {}
        def hook_fn(act, hook):
            cache["act"] = act.detach().clone()
        
        # Hook at layer 8 (as per config)
        hook_point = "blocks.8.hook_resid_pre"
        
        text = "The future of artificial intelligence"
        tokens = model.to_tokens(text)
        
        with model.hooks(fwd_hooks=[(hook_point, hook_fn)]):
            logits = model(tokens)
        
        if "act" not in cache:
            print("   [FAIL] Failed to capture activations")
            return False
        
        activations = cache["act"]
        print(f"   [OK] Activations captured: {activations.shape}")
        
        # Encode with SAE
        print("\n2. Encoding with SAE...")
        with torch.no_grad():
            features = sae.encode(activations)
        
        print(f"   [OK] Features extracted: {features.shape}")
        
        # Analyze features
        last_token_features = features[0, -1, :]  # Last token features
        top_k = 5
        top_vals, top_indices = torch.topk(last_token_features, k=top_k)
        
        print(f"\n3. Top {top_k} features (last token):")
        for i, (val, idx) in enumerate(zip(top_vals, top_indices)):
            print(f"   Feature {idx.item():4d}: {val.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Feature extraction failed: {e}")
        return False

def test_weight_update_proposal(sae):
    """Test weight update proposal based on features."""
    print("\n" + "=" * 60)
    print("Testing Weight Update Proposal")
    print("=" * 60)
    
    try:
        print("1. Simulating feature selection...")
        
        # Simulate some active features
        feature_indices = torch.tensor([10, 25, 50, 75, 100])
        
        print(f"   Selected features: {feature_indices.tolist()}")
        
        # Get feature directions
        print("\n2. Getting feature directions...")
        directions = []
        for idx in feature_indices:
            direction = sae.W_dec[idx].detach()
            directions.append(direction)
            print(f"   Feature {idx}: direction norm = {direction.norm().item():.4f}")
        
        # Calculate Hebbian update
        print("\n3. Calculating Hebbian update...")
        d_model = sae.cfg.d_in  # Should be 768 for GPT-2 small
        delta_w = torch.zeros(d_model, d_model)
        
        for direction in directions:
            update = torch.outer(direction, direction)
            delta_w += update
        
        # Normalize
        delta_w = delta_w / len(directions)
        
        print(f"   [OK] Update matrix shape: {delta_w.shape}")
        print(f"   [OK] Update norm: {delta_w.norm().item():.6f}")
        
        # Apply learning rate
        learning_rate = 0.01
        delta_w = delta_w * learning_rate
        print(f"   [OK] After learning rate ({learning_rate}): {delta_w.norm().item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Weight update proposal failed: {e}")
        return False

def test_safety_metrics():
    """Test safety metric calculations."""
    print("\n" + "=" * 60)
    print("Testing Safety Metrics")
    print("=" * 60)
    
    try:
        from hera.utils.metrics import js_divergence, cosine_drift
        
        print("1. Testing Jensen-Shannon divergence...")
        
        # Create test distributions
        baseline_logits = torch.randn(1, 10, 50257)  # GPT-2 vocab size
        mutated_logits = baseline_logits + torch.randn_like(baseline_logits) * 0.1
        
        js = js_divergence(baseline_logits, mutated_logits)
        print(f"   [OK] JS divergence: {js:.6f}")
        
        print("\n2. Testing cosine drift...")
        
        # Create test activations
        baseline_acts = torch.randn(1, 10, 768)
        mutated_acts = baseline_acts + torch.randn_like(baseline_acts) * 0.05
        
        drift = cosine_drift(baseline_acts, mutated_acts)
        print(f"   [OK] Cosine drift: {drift:.6f}")
        
        print("\n3. Testing PPL calculation...")
        
        # Simple PPL test
        tokens = torch.randint(0, 50257, (1, 10))
        logits = torch.randn(1, 10, 50257)
        
        # Calculate loss (simplified)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 50257),
            tokens.view(-1)
        )
        ppl = torch.exp(loss).item()
        
        print(f"   [OK] PPL: {ppl:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Safety metrics failed: {e}")
        return False

def test_full_pipeline():
    """Test the full H.E.R.A.-R pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline")
    print("=" * 60)
    
    try:
        print("1. Testing with minimal configuration...")
        
        # Create minimal config
        config = {
            "experiment": {"name": "test", "device": "cpu", "seed": 42},
            "model": {"name": "gpt2-small"},
            "sae": {
                "release": "gpt2-small-res-jb",
                "id": "blocks.8.hook_resid_pre",
                "threshold": 2.0
            },
            "evolution": {
                "target_layers": [8],
                "learning_rate": 0.01,
                "top_k_features": 3
            },
            "immune": {
                "max_js_divergence": 0.15,
                "max_ppl_spike": 1.5,
                "max_cosine_drift": 0.05
            },
            "logging": {"verbose": False, "save_dir": "test_logs"}
        }
        
        import yaml
        import tempfile
        from pathlib import Path
        
        # Save config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        print("2. Loading HeraEngine...")
        from hera.core.engine import HeraEngine
        
        engine = HeraEngine(config_path)
        print(f"   [OK] Engine loaded: {engine.get_status()}")
        
        print("\n3. Testing single evolution step...")
        test_text = "Artificial intelligence is"
        
        success = engine.evolve(test_text)
        print(f"   [OK] Evolution {'accepted' if success else 'rejected'}")
        
        print("\n4. Checking metrics...")
        evo_metrics = engine.evo_layer.get_metrics()
        immune_stats = engine.immune.get_stats()
        
        print(f"   Evolution metrics: {evo_metrics}")
        print(f"   Immune stats: {immune_stats}")
        
        print("\n5. Testing reset...")
        engine.reset()
        print(f"   [OK] Engine reset")
        
        print("\n6. Clean shutdown...")
        engine.shutdown()
        print(f"   [OK] Shutdown complete")
        
        # Clean up
        Path(config_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all core functionality tests."""
    print("H.E.R.A.-R Core Functionality Verification")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Model loading
    print("\n[Phase 1] Testing dependencies...")
    model = test_model_loading()
    results["model_loading"] = model is not None
    
    # Test 2: SAE loading
    sae = test_sae_loading()
    results["sae_loading"] = sae is not None
    
    if model and sae:
        # Test 3: Feature extraction
        results["feature_extraction"] = test_feature_extraction(model, sae)
        
        # Test 4: Weight update
        results["weight_update"] = test_weight_update_proposal(sae)
    
    # Test 5: Safety metrics
    results["safety_metrics"] = test_safety_metrics()
    
    # Test 6: Full pipeline (if dependencies are available)
    print("\n[Phase 2] Testing integrated pipeline...")
    results["full_pipeline"] = test_full_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("Core Functionality Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL CORE FUNCTIONS ARE WORKING!")
        print("\nH.E.R.A.-R can:")
        print("1. [OK] Load transformer models")
        print("2. [OK] Load SAE feature extractors")
        print("3. [OK] Extract features from activations")
        print("4. [OK] Propose weight updates based on features")
        print("5. [OK] Calculate safety metrics")
        print("6. [OK] Run full evolution pipeline")
    else:
        print("[WARN] Some core functions are not working.")
        print("\nIssues detected:")
        for test_name, passed in results.items():
            if not passed:
                print(f"  • {test_name}")
        
        print("\nRecommendations:")
        if not results.get("model_loading"):
            print("  • Check transformer-lens installation")
        if not results.get("sae_loading"):
            print("  • Check sae-lens installation")
        if not results.get("full_pipeline"):
            print("  • Check all dependencies and config")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)