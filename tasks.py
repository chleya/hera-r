#!/usr/bin/env python3
"""
Task runner for H.E.R.A.-R development.
Usage: python tasks.py [command]
"""

import sys
import subprocess
import os
from pathlib import Path

def run(cmd, capture=False):
    """Run a shell command."""
    print(f"$ {cmd}")
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    else:
        return subprocess.run(cmd, shell=True).returncode

def task_help():
    """Show available tasks."""
    print("Available tasks:")
    print("  setup     - Set up development environment")
    print("  install   - Install package in development mode")
    print("  test      - Run tests")
    print("  lint      - Run code quality checks")
    print("  format    - Format code")
    print("  typecheck - Run type checking")
    print("  docs      - Build documentation")
    print("  clean     - Clean build artifacts")
    print("  run-cli   - Run CLI interface")
    print("  run-web   - Run web interface")

def task_setup():
    """Set up development environment."""
    print("Setting up development environment...")
    return run("python scripts/setup_dev.py")

def task_install():
    """Install package in development mode."""
    print("Installing package in development mode...")
    return run("pip install -e .[dev]")

def task_test():
    """Run tests."""
    print("Running tests...")
    return run("pytest tests/ -v")

def task_lint():
    """Run code quality checks."""
    print("Running code quality checks...")
    commands = [
        "black --check .",
        "isort --check-only .",
        "flake8 hera tests",
    ]
    for cmd in commands:
        if run(cmd) != 0:
            return 1
    return 0

def task_format():
    """Format code."""
    print("Formatting code...")
    commands = [
        "black .",
        "isort .",
    ]
    for cmd in commands:
        if run(cmd) != 0:
            return 1
    return 0

def task_typecheck():
    """Run type checking."""
    print("Running type checking...")
    return run("mypy hera")

def task_docs():
    """Build documentation."""
    print("Building documentation...")
    return run("mkdocs build")

def task_clean():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    to_clean = [
        "build/",
        "dist/",
        "*.egg-info/",
        "__pycache__/",
        "**/__pycache__/",
        "*.pyc",
        "*.pyo",
        ".coverage",
        "htmlcov/",
        ".pytest_cache/",
        ".mypy_cache/",
    ]
    for pattern in to_clean:
        run(f"rm -rf {pattern}")
    return 0

def task_run_cli():
    """Run CLI interface."""
    print("Running CLI interface...")
    return run("python main.py")

def task_run_web():
    """Run web interface."""
    print("Running web interface...")
    return run("python app.py")

def main():
    """Main task runner."""
    if len(sys.argv) < 2:
        task_help()
        return 1
    
    task = sys.argv[1]
    tasks = {
        "help": task_help,
        "setup": task_setup,
        "install": task_install,
        "test": task_test,
        "lint": task_lint,
        "format": task_format,
        "typecheck": task_typecheck,
        "docs": task_docs,
        "clean": task_clean,
        "run-cli": task_run_cli,
        "run-web": task_run_web,
    }
    
    if task in tasks:
        return tasks[task]()
    else:
        print(f"Unknown task: {task}")
        task_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())