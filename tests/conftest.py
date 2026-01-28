# -*- coding: utf-8 -*-
"""Pytest configuration for H.E.R.A.-R tests."""

import pytest
import torch
import os
import sys
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "experiment": {
            "name": "test-hera",
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
        },
        "immune": {
            "max_js_divergence": 0.15,
            "max_ppl_spike": 1.5,
            "max_cosine_drift": 0.05,
        },
        "logging": {
            "verbose": False,
            "save_dir": "test_logs",
        },
    }

@pytest.fixture
def cleanup_test_dirs():
    """Clean up test directories after tests."""
    test_dirs = ["test_logs", "test_checkpoints"]
    
    # Yield to test
    yield
    
    # Cleanup after test
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield