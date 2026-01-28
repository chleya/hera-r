# H.E.R.A.-R (Research Edition) 🧬

**Hyper-Evolving Robust Architecture - Research Edition**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

H.E.R.A.-R is a research framework for **online, self-evolving Large Language Models (LLMs)**. Unlike traditional fine-tuning, H.E.R.A.-R enables real-time weight updates based on input stimuli, guarded by a multi-metric "Digital Immune System" that prevents catastrophic forgetting and concept drift.

## ✨ Key Features

- **🧠 Online Neuro-Evolution**: Modifies model weights in real-time using Hebbian-like learning rules
- **🛡️ Digital Immune System**: Multi-metric safety checks with automatic rollback capability
- **🔬 Interpretable Features**: Sparse Autoencoder (SAE) integration for feature-level analysis
- **🖥️ Dual Interface**: Command-line interface for research + Gradio Web UI for experimentation
- **📊 Comprehensive Monitoring**: Real-time metrics, logging, and visualization
- **🔧 Extensible Architecture**: Modular design for easy experimentation and extension

## 🎯 Research Goals

H.E.R.A.-R explores fundamental questions in LLM adaptation:
- Can LLMs evolve in real-time without catastrophic forgetting?
- How can we make online learning safe and controllable?
- What neural mechanisms enable stable continuous adaptation?

## 🚀 Quick Start

### Installation

#### From Source (Recommended for Development)
```bash
# Clone the repository
git clone https://github.com/chleya/hera-r.git
cd hera-r

# Install in development mode
pip install -e .[dev]

# Set up development environment
python scripts/setup_dev.py
```

#### Basic Installation
```bash
pip install hera-r
```

### Basic Usage

#### Command Line Interface
```bash
# Run the CLI demo
python main.py

# Or use the installed script
hera-cli
```

#### Web Interface
```bash
# Launch the Gradio web UI
python app.py

# Or use the installed script
hera-web
```

## 📖 Documentation

- [User Guide](docs/user_guide.md) - Complete usage instructions
- [API Reference](docs/api.md) - Detailed API documentation
- [Research Background](docs/research.md) - Theoretical foundations
- [Examples](examples/) - Practical usage examples

## 🏗️ Architecture Overview

```
Input Text
    │
    ▼
HeraEngine (Orchestrator)
    ├── Model Wrapper (transformer-lens)
    ├── SAE Interface (sae-lens)
    ├── NeuroEvolutionaryLayer
    └── ImmuneSystem
        ├── JS Divergence Monitor
        ├── Perplexity Spike Detector
        └── Cosine Drift Analyzer
    │
    ▼
Safe Update or Rollback
```

### Core Components

1. **HeraEngine**: Main orchestrator that coordinates all components
2. **NeuroEvolutionaryLayer**: Implements Hebbian-like weight updates based on SAE features
3. **ImmuneSystem**: Safety monitor with configurable thresholds
4. **SAEInterface**: Bridge to sparse autoencoder models for interpretable features
5. **PatchRegistry**: Tracks all evolutionary changes for analysis and rollback

## 🔧 Configuration

H.E.R.A.-R uses YAML configuration files. See `configs/default.yaml` for the default configuration:

```yaml
experiment:
  name: "hera-r-gpt2-sae"
  device: "cuda"  # or "cpu"
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
```

## 🧪 Examples

### Basic Evolution Session
```python
from hera.core.engine import HeraEngine

# Initialize engine
engine = HeraEngine("configs/default.yaml")

# Evolve with a prompt
success = engine.evolve("The capital of France is")
if success:
    print("Evolution committed successfully!")
else:
    print("Evolution rejected by immune system")
```

### Custom Configuration
```python
import yaml

# Create custom config
config = {
    "experiment": {"device": "cpu"},
    "evolution": {"learning_rate": 0.005},
    "immune": {"max_js_divergence": 0.1}
}

# Save and use
with open("custom_config.yaml", "w") as f:
    yaml.dump(config, f)

engine = HeraEngine("custom_config.yaml")
```

## 📊 Monitoring and Analysis

H.E.R.A.-R provides comprehensive monitoring:

```python
# Access evolution history
history = engine.registry.history
for entry in history:
    print(f"Step {entry['step']}: {entry['status']}")
    print(f"Metrics: {entry['metrics']}")
```

## 🔬 Research Applications

### 1. **Online Adaptation Studies**
- Study how LLMs adapt to new information in real-time
- Measure adaptation speed and stability

### 2. **Safety Mechanism Research**
- Test different immune system configurations
- Develop new safety metrics for online learning

### 3. **Feature Analysis**
- Correlate SAE features with semantic concepts
- Study feature activation patterns during evolution

### 4. **Comparative Studies**
- Compare different evolution strategies
- Benchmark against traditional fine-tuning

## 🛠️ Development

### Setting Up Development Environment
```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
python tasks.py format

# Run all checks
python tasks.py lint
```

### Project Structure
```
hera-r/
├── hera/                    # Main package
│   ├── core/               # Core engine and orchestration
│   ├── modules/            # Functional modules
│   └── utils/              # Utilities and helpers
├── configs/                # Configuration files
├── data/                   # Data and reference probes
├── tests/                  # Test suite
├── examples/               # Usage examples
├── docs/                   # Documentation
├── scripts/                # Development scripts
└── tasks.py               # Development task runner
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📚 Citation

If you use H.E.R.A.-R in your research, please cite:

```bibtex
@software{hera_r_2026,
  author = {Chen Leiyang},
  title = {H.E.R.A.-R: Hyper-Evolving Robust Architecture - Research Edition},
  year = {2026},
  url = {https://github.com/chleya/hera-r}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/chleya/hera-r/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chleya/hera-r/discussions)
- **Email**: chleya@example.com

## 🙏 Acknowledgments

- Built on top of [`transformer-lens`](https://github.com/neelnanda-io/TransformerLens)
- Uses [`sae-lens`](https://github.com/jbloomAus/SAELens) for sparse autoencoders
- Inspired by research on online learning and neural plasticity

---

**H.E.R.A.-R**: Exploring the boundaries of adaptive intelligence, one evolution at a time. 🧬