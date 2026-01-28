#!/usr/bin/env python3
"""Setup script for H.E.R.A.-R package."""

from setuptools import setup, find_packages

# Read requirements from pyproject.toml or requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hera-r",
    version="0.1.0",
    description="Hyper-Evolving Robust Architecture - Research Edition: Online self-evolving LLM framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chen Leiyang",
    author_email="chleya@example.com",
    url="https://github.com/chleya/hera-r",
    packages=find_packages(include=["hera", "hera.*"]),
    package_data={
        "hera": ["py.typed"],
    },
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings[python]>=0.22.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["llm", "evolution", "online-learning", "transformer", "research"],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "hera-cli=hera.cli:main",
            "hera-web=hera.web:main",
        ],
    },
)