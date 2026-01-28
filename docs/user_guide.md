# User Guide

Welcome to H.E.R.A.-R! This guide will help you get started with using the framework for your research and experiments.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Core Concepts](#core-concepts)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- (Optional) CUDA-capable GPU for faster inference

### Installation Methods

#### Method 1: From PyPI (Coming Soon)
```bash
pip install hera-r
```

#### Method 2: From Source (Recommended)
```bash
# Clone the repository
git clone https://github.com/chleya/hera-r.git
cd hera-r

# Install in development mode
pip install -e .

# Install optional dependencies for development
pip install -e .[dev]
```

#### Method 3: Using Conda
```bash
conda create -n hera-r python=3.9
conda activate hera-r
pip install -e .
```

### Verification
```bash
# Check installation
python -c "import hera; print(hera.__version__)"

# Test basic functionality
python -m pytest tests/test_basic.py -v
```

## Quick Start

### Running the Demo

#### Command Line Interface
```bash
# Run the pre-configured demo
python main.py
```

This will:
1. Load a GPT-2 model with pre-trained SAE
2. Process a series of test prompts
3. Show evolution decisions and metrics

#### Web Interface
```bash
# Launch the Gradio web UI
python app.py
```

Then open your browser to `http://localhost:7860`

### Your First Evolution Session

```python
import yaml
from hera.core.engine import HeraEngine

# Create a simple configuration
config = {
    "experiment": {
        "name": "my-first-evolution",
        "device": "cpu",  # Use "cuda" if you have GPU
        "seed": 42
    },
    "model": {
        "name": "gpt2-small"
    },
    "sae": {
        "release": "gpt2-small-res-jb",
        "id": "blocks.8.hook_resid_pre",
        "threshold": 2.0
    },
    "evolution": {
        "target_layers": [8],
        "learning_rate": 0.01,
        "max_rank": 1,
        "top_k_features": 3
    },
    "immune": {
        "max_js_divergence": 0.15,
        "max_ppl_spike": 1.5,
        "max_cosine_drift": 0.05
    }
}

# Save configuration
with open("my_config.yaml", "w") as f:
    yaml.dump(config, f)

# Initialize engine
engine = HeraEngine("my_config.yaml")

# Run evolution on some prompts
prompts = [
    "The capital of France is",
    "Python is a programming language for",
    "Machine learning is"
]

for prompt in prompts:
    print(f"\nProcessing: {prompt}")
    success = engine.evolve(prompt)
    if success:
        print("✅ Evolution committed")
    else:
        print("🛑 Evolution rejected by immune system")

# Save results
engine.registry.save_history()
```

## Configuration

### Configuration Files

H.E.R.A.-R uses YAML configuration files. The main configuration sections are:

#### Experiment Settings
```yaml
experiment:
  name: "experiment-name"  # Unique identifier
  device: "cuda"  # "cuda" or "cpu"
  seed: 42  # Random seed for reproducibility
```

#### Model Configuration
```yaml
model:
  name: "gpt2-small"  # Currently supports GPT-2 variants
  # Future: Will support more models
```

#### SAE Configuration
```yaml
sae:
  release: "gpt2-small-res-jb"  # SAE release name
  id: "blocks.8.hook_resid_pre"  # Hook point in transformer
  threshold: 2.0  # Activation threshold for features
```

#### Evolution Parameters
```yaml
evolution:
  target_layers: [8]  # Which layers to evolve
  learning_rate: 0.01  # How aggressive updates are
  max_rank: 1  # Rank of weight updates
  top_k_features: 3  # Number of top features to consider
```

#### Immune System Settings
```yaml
immune:
  max_js_divergence: 0.15  # Maximum Jensen-Shannon divergence
  max_ppl_spike: 1.5  # Maximum perplexity increase
  max_cosine_drift: 0.05  # Maximum activation drift
```

### Environment Variables

You can also use environment variables for configuration:

```bash
# Set in your shell or .env file
export HERA_DEVICE="cuda"
export HERA_MODEL_NAME="gpt2-small"
export HERA_LEARNING_RATE="0.01"
```

## Core Concepts

### 1. Online Evolution

Unlike traditional fine-tuning that requires a full training cycle, H.E.R.A.-R performs **online evolution**:
- Processes one prompt at a time
- Makes immediate weight updates
- Each update is evaluated by the immune system

### 2. Sparse Autoencoders (SAEs)

SAEs are used to:
- Decompose neural activations into interpretable features
- Identify which features are activated by specific inputs
- Guide weight updates toward meaningful directions

### 3. Digital Immune System

The immune system monitors three key metrics:

1. **JS Divergence**: Measures output distribution changes
2. **Perplexity Spike**: Detects degradation in language modeling
3. **Cosine Drift**: Tracks internal representation changes

### 4. Patch Registry

All evolutionary changes are recorded in the patch registry:
- Tracks successful and rejected updates
- Stores metrics and context for each evolution
- Enables analysis and rollback if needed

## Advanced Usage

### Custom Evolution Strategies

You can implement custom evolution strategies by extending the `NeuroEvolutionaryLayer` class:

```python
from hera.modules.evolutionary_layer import NeuroEvolutionaryLayer

class CustomEvolutionLayer(NeuroEvolutionaryLayer):
    def propose(self, activations):
        # Your custom evolution logic here
        feature_acts = self.sae.encode(activations)
        
        # Example: Use different feature selection
        mean_acts = feature_acts.mean(dim=[0, 1])
        top_indices = torch.topk(mean_acts, k=5).indices
        
        # Create custom weight update
        delta_w = torch.zeros_like(self.target_module.W.data)
        for idx in top_indices:
            direction = self.sae.decode_direction(idx)
            # Custom update rule
            update = torch.outer(direction, direction) * 0.5
            delta_w += update
        
        return delta_w * self.cfg["evolution"]["learning_rate"]
```

### Custom Immune Metrics

Add custom safety metrics by extending the `ImmuneSystem` class:

```python
from hera.modules.immune_system import ImmuneSystem

class EnhancedImmuneSystem(ImmuneSystem):
    def verify(self, baseline, mutated, input_text):
        # Run standard checks
        is_safe, reason, metrics = super().verify(baseline, mutated, input_text)
        
        if not is_safe:
            return False, reason, metrics
        
        # Add custom check: vocabulary distribution stability
        vocab_change = self._check_vocabulary_distribution(baseline, mutated)
        metrics["vocab_change"] = vocab_change
        
        if vocab_change > 0.1:  # Custom threshold
            return False, f"Vocabulary shift too large ({vocab_change:.3f})", metrics
        
        return True, "Stable", metrics
    
    def _check_vocabulary_distribution(self, baseline, mutated):
        # Implement custom metric
        base_probs = torch.softmax(baseline["logits"][0, -1], dim=-1)
        mut_probs = torch.softmax(mutated["logits"][0, -1], dim=-1)
        
        # KL divergence between distributions
        kl = torch.sum(base_probs * torch.log(base_probs / mut_probs))
        return kl.item()
```

### Batch Processing

For research experiments, you might want to process batches:

```python
def batch_evolution(engine, prompts, batch_size=4):
    """Process prompts in batches."""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_results = []
        
        for prompt in batch:
            success = engine.evolve(prompt)
            batch_results.append({
                "prompt": prompt,
                "success": success,
                "metrics": engine.registry.history[-1]["metrics"] if engine.registry.history else {}
            })
        
        results.extend(batch_results)
        
        # Optional: Save checkpoint
        if (i // batch_size) % 10 == 0:
            engine.registry.save_history()
            print(f"Checkpoint saved at batch {i//batch_size}")
    
    return results
```

## Troubleshooting

### Common Issues

#### Issue: "CUDA out of memory"
**Solution:**
- Reduce batch size or sequence length
- Use `device: "cpu"` in configuration
- Use a smaller model

#### Issue: "SAE model not found"
**Solution:**
- Check internet connection
- Verify SAE release name is correct
- Try downloading manually: https://github.com/jbloomAus/SAELens

#### Issue: "Evolution always rejected"
**Solution:**
- Increase immune system thresholds
- Reduce learning rate
- Check if prompts are too diverse

#### Issue: "Web UI won't start"
**Solution:**
- Check if port 7860 is available
- Try `python app.py --share` for temporary public link
- Check Gradio installation

### Getting Help

- Check the [FAQ](faq.md)
- Search [GitHub Issues](https://github.com/chleya/hera-r/issues)
- Join the [Discussions](https://github.com/chleya/hera-r/discussions)

## Next Steps

Now that you're familiar with the basics, you might want to:

1. **Explore Examples**: Check the `examples/` directory
2. **Read API Documentation**: See `docs/api.md`
3. **Contribute**: Read `CONTRIBUTING.md`
4. **Experiment**: Try your own research questions!

Happy evolving! 🧬