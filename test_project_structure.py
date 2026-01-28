#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Structure Test

This test verifies that the project structure is correct
and all files are in place, without running any actual code.
"""

import os
import sys
from pathlib import Path

def check_project_structure():
    """Check if the project has all required files."""
    print("=" * 60)
    print("Project Structure Verification")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    
    # Required directories
    required_dirs = [
        "hera",
        "hera/core",
        "hera/modules", 
        "hera/utils",
        "configs",
        "examples",
        "docs",
        "tests",
    ]
    
    # Required files
    required_files = [
        "main.py",
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "configs/default.yaml",
        "hera/__init__.py",
        "hera/core/__init__.py",
        "hera/core/engine.py",
        "hera/modules/__init__.py",
        "hera/modules/evolutionary_layer.py",
        "hera/modules/immune_system.py",
        "hera/modules/sae_interface.py",
        "hera/utils/__init__.py",
        "hera/utils/logger.py",
        "hera/utils/metrics.py",
    ]
    
    # Enhanced files (new additions)
    enhanced_files = [
        "hera/modules/evolutionary_layer_enhanced.py",
        "hera/modules/immune_system_enhanced.py",
        "examples/enhanced_features_demo.py",
        "examples/research_experiment.py",
        "test_enhanced_features.py",
        "test_core_functionality.py",
        "minimal_working_example.py",
        "docs/research_paper.md",
        "configs/research_config.yaml",
    ]
    
    print("\n1. Checking required directories...")
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
        else:
            print(f"   [OK] {dir_path}/")
    
    if missing_dirs:
        print(f"   [FAIL] Missing directories: {missing_dirs}")
    else:
        print("   [OK] All directories present")
    
    print("\n2. Checking required files...")
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"   [OK] {file_path}")
    
    if missing_files:
        print(f"   [FAIL] Missing files: {missing_files}")
    else:
        print("   [OK] All required files present")
    
    print("\n3. Checking enhanced files...")
    enhanced_missing = []
    for file_path in enhanced_files:
        full_path = project_root / file_path
        if not full_path.exists():
            enhanced_missing.append(file_path)
        else:
            print(f"   [OK] {file_path}")
    
    if enhanced_missing:
        print(f"   [WARN] Missing enhanced files: {enhanced_missing}")
    else:
        print("   [OK] All enhanced files present")
    
    print("\n4. Checking Python module imports...")
    try:
        # Test importing core modules (without dependencies)
        sys.path.insert(0, str(project_root))
        
        print("   Testing hera.utils.logger...")
        from hera.utils.logger import HeraLogger
        print("   [OK] HeraLogger imports successfully")
        
        print("   Testing hera.utils.metrics...")
        from hera.utils.metrics import js_divergence, cosine_drift
        print("   [OK] Metrics functions import successfully")
        
        print("   Testing hera.core.patch_registry...")
        from hera.core.patch_registry import PatchRegistry
        print("   [OK] PatchRegistry imports successfully")
        
    except ImportError as e:
        print(f"   [FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"   [FAIL] Other error: {e}")
        return False
    
    print("\n5. Checking configuration files...")
    config_files = list((project_root / "configs").glob("*.yaml"))
    if config_files:
        print(f"   Found {len(config_files)} config files:")
        for cfg in config_files:
            print(f"   - {cfg.name}")
    else:
        print("   [WARN] No config files found in configs/")
    
    print("\n6. Checking example files...")
    example_files = list((project_root / "examples").glob("*.py"))
    if example_files:
        print(f"   Found {len(example_files)} example files:")
        for ex in example_files:
            print(f"   - {ex.name}")
    else:
        print("   [WARN] No example files found in examples/")
    
    print("\n7. Summary:")
    total_checks = len(required_dirs) + len(required_files) + len(enhanced_files)
    passed_checks = (len(required_dirs) - len(missing_dirs)) + \
                   (len(required_files) - len(missing_files)) + \
                   (len(enhanced_files) - len(enhanced_missing))
    
    print(f"   Total checks: {total_checks}")
    print(f"   Passed: {passed_checks}")
    print(f"   Failed: {total_checks - passed_checks}")
    
    if not missing_dirs and not missing_files:
        print("\n" + "=" * 60)
        print("[SUCCESS] Project structure is COMPLETE!")
        print("=" * 60)
        print("\nThe project has all required files and directories.")
        print("Import structure is correct.")
        return True
    else:
        print("\n" + "=" * 60)
        print("[FAIL] Project structure is INCOMPLETE")
        print("=" * 60)
        print("\nMissing components need to be addressed.")
        return False

def check_dependency_issues():
    """Check for common dependency issues."""
    print("\n" + "=" * 60)
    print("Dependency Issue Analysis")
    print("=" * 60)
    
    issues = []
    
    # Check transformers version issue
    try:
        import transformers
        print(f"1. Transformers version: {transformers.__version__}")
        
        # Check for known compatibility issues
        if hasattr(transformers, 'BertForPreTraining'):
            print("   [OK] BertForPreTraining is available")
        else:
            issues.append("transformers.BertForPreTraining not found")
            print("   [WARN] BertForPreTraining not found (may cause issues)")
            
    except ImportError:
        issues.append("transformers not installed")
        print("1. [FAIL] transformers not installed")
    
    # Check torch
    try:
        import torch
        print(f"2. PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        issues.append("torch not installed")
        print("2. [FAIL] torch not installed")
    
    # Check transformer-lens
    try:
        import transformer_lens
        print(f"3. TransformerLens version: {getattr(transformer_lens, '__version__', 'unknown')}")
    except ImportError as e:
        issues.append(f"transformer_lens import error: {e}")
        print(f"3. [FAIL] transformer_lens import error: {e}")
    
    # Check sae-lens
    try:
        import sae_lens
        print(f"4. SAE Lens version: {getattr(sae_lens, '__version__', 'unknown')}")
    except ImportError as e:
        issues.append(f"sae_lens import error: {e}")
        print(f"4. [FAIL] sae_lens import error: {e}")
    
    if issues:
        print("\n" + "=" * 60)
        print("DEPENDENCY ISSUES DETECTED")
        print("=" * 60)
        print("\nIssues found:")
        for issue in issues:
            print(f"  • {issue}")
        
        print("\nRecommended fixes:")
        print("1. Update transformers to a compatible version:")
        print("   pip install transformers==4.36.0")
        print("\n2. Install transformer-lens and sae-lens:")
        print("   pip install transformer-lens sae-lens")
        print("\n3. Or install all dependencies at once:")
        print("   pip install -r requirements.txt")
        
        return False
    else:
        print("\n" + "=" * 60)
        print("[OK] All dependencies available")
        print("=" * 60)
        return True

def main():
    """Run project structure verification."""
    print("H.E.R.A.-R Project Structure Verification")
    print("=" * 60)
    
    # Check project structure
    structure_ok = check_project_structure()
    
    # Check dependencies
    print("\n" + "=" * 60)
    print("Note: Dependency checks may fail due to version issues.")
    print("This is expected if you haven't installed all dependencies.")
    print("=" * 60)
    
    response = input("\nRun dependency check anyway? (y/n): ").lower().strip()
    if response == 'y':
        dependencies_ok = check_dependency_issues()
    else:
        dependencies_ok = False
        print("\nSkipping dependency check.")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    
    if structure_ok:
        print("\n[SUCCESS] Project structure is correct.")
        print("All required files and directories are in place.")
        
        if not dependencies_ok:
            print("\n[WARN] Dependencies need to be installed/updated.")
            print("See recommendations above.")
    else:
        print("\n[FAIL] Project structure issues found.")
        print("Please fix missing files/directories.")
    
    return structure_ok

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[STOP] Interrupted by user.")
        exit(1)