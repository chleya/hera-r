#!/usr/bin/env python3
"""Development environment setup script for H.E.R.A.-R."""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command and print output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True

def setup_environment():
    """Set up the development environment."""
    print("=" * 60)
    print("Setting up H.E.R.A.-R development environment")
    print("=" * 60)
    
    # Create necessary directories
    directories = [
        "logs",
        "checkpoints",
        "data/cache",
        "data/processed",
        "data/raw",
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Copy .env.example to .env if it doesn't exist
    if not Path(".env").exists() and Path(".env.example").exists():
        shutil.copy(".env.example", ".env")
        print("Created .env file from .env.example")
    
    # Create default config if missing
    config_dir = Path("configs")
    if not config_dir.exists():
        config_dir.mkdir()
    
    default_config = config_dir / "default.yaml"
    if not default_config.exists():
        default_config.write_text("""experiment:
  name: "hera-r-gpt2-sae"
  device: "cuda" # or "cpu"
  seed: 42

model:
  name: "gpt2-small"

sae:
  release: "gpt2-small-res-jb"
  id: "blocks.8.hook_resid_pre" 
  threshold: 2.0 

evolution:
  target_layers: [8]
  learning_rate: 0.01
  max_rank: 1 
  top_k_features: 3

immune:
  max_js_divergence: 0.15 
  max_ppl_spike: 1.5 
  max_cosine_drift: 0.05 

logging:
  verbose: true
  save_dir: "logs/"
""")
        print("Created default config file")
    
    print("\nEnvironment setup complete!")

def install_dependencies():
    """Install development dependencies."""
    print("\n" + "=" * 60)
    print("Installing development dependencies")
    print("=" * 60)
    
    # Install the package in development mode
    if not run_command("pip install -e .[dev]"):
        print("Failed to install package in development mode")
        return False
    
    # Install pre-commit hooks
    if run_command("pre-commit install"):
        print("Pre-commit hooks installed")
    
    print("\nDependencies installed successfully!")
    return True

def run_checks():
    """Run initial code quality checks."""
    print("\n" + "=" * 60)
    print("Running initial code quality checks")
    print("=" * 60)
    
    checks = [
        ("black --check .", "Code formatting check"),
        ("isort --check-only .", "Import sorting check"),
        ("mypy hera", "Type checking"),
    ]
    
    all_passed = True
    for cmd, description in checks:
        print(f"\n{description}:")
        if not run_command(cmd):
            print(f"⚠️  {description} failed")
            all_passed = False
        else:
            print(f"✅ {description} passed")
    
    return all_passed

def main():
    """Main setup function."""
    print("H.E.R.A.-R Development Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required")
        sys.exit(1)
    
    # Setup steps
    setup_environment()
    
    if not install_dependencies():
        print("Warning: Dependency installation had issues")
    
    if not run_checks():
        print("Warning: Some code quality checks failed")
    
    print("\n" + "=" * 60)
    print("Setup complete! Next steps:")
    print("1. Review and edit .env file if needed")
    print("2. Run tests: pytest")
    print("3. Start development!")
    print("=" * 60)

if __name__ == "__main__":
    main()