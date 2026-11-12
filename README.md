# LLM Inference Lab

Experiments and reference implementations for understanding and improving
autoregressive language-model inference.

## Overview

This repository currently explores:

- **Speculative decoding**: speed up generation by pairing a small draft
  model with a larger target model.
- **Attention sinks**: detect and visualize tokens that keep attracting
  attention during long-context inference.
- **Entropy tracking**: inspect token-level uncertainty and predictability
  while a model generates text.

Related research papers are collected in [`references/`](references/).

## Requirements

- Python 3.10 or newer
- A PyTorch-supported device. CPU works; CUDA and Apple Silicon (MPS) are
  used when available.
- Access to the Hugging Face models used by each experiment

## Installation

Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Apple Silicon, check whether the Metal Performance Shaders (MPS) device is
available:

```bash
python -m utils.check_mps
```

## Project Layout

```text
.
├── attention_sink_detector/   # Attention-sink analysis and rendering
├── entropy_tracker/           # Token entropy tracking and rendering
├── speculative_decoding/      # Draft/target generation and benchmarks
├── utils/                     # Model-loading and device utilities
├── cli/                       # Shared command-line rendering and runner code
└── references/                # Papers related to the experiments
```

## Running Experiments

Run the experiment runners from the repository root:

```bash
# Speculative decoding
python -m speculative_decoding.runner

# Attention-sink detection
python -m attention_sink_detector.runner

# Entropy tracking
python -m entropy_tracker.runner
```

> **Note:** The runners may download model weights the first time they run.

To check that models load correctly:

```bash
python -m utils.test_model_load
```

The `speculative_decoding` package also has separate modules for generation,
sampling, FLOP accounting and benchmarking.

## Development

Run the syntax check from the project root:

```bash
python -m compileall .
```

Keep local virtual environments, model caches and generated experiment output
out of version control. The repository's `.gitignore` covers the project
defaults.

## References

The experiments draw on the papers in [`references/`](references/), including
work on:

- Efficient streaming language models and attention sinks
- Predictability and surprise in language models
- Speculative decoding

## License

See [`LICENSE`](LICENSE).
