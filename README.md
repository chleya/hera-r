# H.E.R.A.-R (Research Edition) 🧬

**Hyper-Evolving Robust Architecture**

H.E.R.A.-R is a research framework for **online, self-evolving Large Language Models (LLMs)**. Unlike traditional fine-tuning, H.E.R.A.-R uses Sparse Autoencoders (SAEs) to decompose internal representations and applies Hebbian-like weight updates in real-time, guarded by a multi-metric "Immune System."

## ✨ Features

- **🧠 Online Neuro-Evolution**: Modifies weights on-the-fly based on input stimuli.
- **🛡️ Digital Immune System**: Rolls back changes that cause concept drift or perplexity spikes.
- **🔬 SAE Integration**: Powered by `sae-lens` and `transformer-lens` for interpretable features.
- **🖥️ Dual Interface**: Includes a CLI (Terminal) and a Web UI (Gradio).

## 🚀 Quick Start

### 1. Installation
```bash
git clone [https://github.com/YOUR_USERNAME/hera-r.git](https://github.com/YOUR_USERNAME/hera-r.git)
cd hera-r
pip install -r requirements.txt