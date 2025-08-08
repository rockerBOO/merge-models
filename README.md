# Model Merger

Initially only DARE merger implemented

Implementation of DARE (Drop And REscale) method from the paper "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch".

## Overview

DARE allows merging multiple fine-tuned language models without retraining or GPUs by:

1. Computing delta parameters (differences between fine-tuned and base models)
2. Randomly dropping 90%+ of delta parameters
3. Rescaling remaining parameters to preserve model performance
4. Memory-efficient key-by-key processing

## Usage

### Command Line Interface

#### Using TOML Configuration (Recommended)

Create a configuration file (e.g., `merge_config.toml`):

```toml
[dare]
drop_rate = 0.9
delta_threshold = 0.002
verbose = true

[models]
base_file = "models/base_model.safetensors"
merge_files = [
    "models/math_tuned_model.safetensors",
    "models/code_tuned_model.safetensors"
]
output_file = "merged_model.safetensors"
```

Run the merger:

```bash
uv run python dare.py --config merge_config.toml
```

#### Using Command Line Arguments

```bash
# Basic merge
uv run python dare.py \
  --base models/base_model.safetensors \
  --merge models/math_model.safetensors models/code_model.safetensors \
  --output merged_model.safetensors

# With custom parameters
uv run python dare.py \
  --base base.safetensors \
  --merge math.safetensors code.safetensors \
  --drop-rate 0.95 \
  --delta-threshold 0.001 \
  --verbose \
  --output super_merged.safetensors
```

#### Override TOML with CLI Arguments

```bash
# Use config file but override specific settings
uv run python dare.py \
  --config merge_config.toml \
  --drop-rate 0.99 \
  --quiet
```

#### CLI Options

```
--config, -c          Path to TOML configuration file
--base, -b            Path to base model SafeTensors file
--merge, -m           Paths to model SafeTensors files to merge
--output, -o          Output path for merged model
--drop-rate           Proportion of delta parameters to drop (0.0-1.0)
--delta-threshold     Warning threshold for large delta parameters
--verbose, -v         Enable verbose output
--quiet, -q           Disable verbose output
--help, -h            Show help message
```

### Python API

#### Basic Example

```python
from dare import DareMerger, DareConfig

# Configure DARE parameters
config = DareConfig(
    drop_rate=0.9,          # Drop 90% of delta parameters
    delta_threshold=0.002,  # Warning threshold for large deltas
    verbose=True            # Enable progress logging
)

# Create merger
merger = DareMerger(
    base_file='base_model.safetensors',
    merge_files=['math_model.safetensors', 'code_model.safetensors'],
    config=config
)

# Execute merge
merger.merge('merged_model.safetensors')
```

#### Advanced Configuration

```python
# Custom configuration
config = DareConfig(
    drop_rate=0.95,         # More aggressive dropping
    delta_threshold=0.001,  # Stricter delta warnings
    verbose=False           # Silent operation
)
```

## Requirements

- Python 3.12+
- PyTorch 2.8.0+
- SafeTensors 0.6.1+
- NumPy

## Key Features

- **Memory Efficient**: Processes weights key-by-key, minimal RAM usage
- **Flexible**: Works with any SafeTensors model weights
- **Fast**: No model loading, direct tensor operations
- **Configurable**: Adjustable drop rates and thresholds

## Limitations

- Works best with small delta parameter ranges (< 0.002)
- Requires models fine-tuned from same base model
- Performance depends on quality of source models

## Testing

Run the test suite:

```bash
uv run python test_dare.py
```

This creates dummy models and verifies the merger functionality.

## Paper Reference

```
@article{yu2024language,
  title={Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch},
  author={Yu, Le and Yu, Bowen and Yu, Haiyang and Huang, Fei and Li, Yongbin},
  journal={arXiv preprint arXiv:2311.03099},
  year={2024}
}
```
