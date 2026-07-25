---
language:
- en
- fr
license: mit
library_name: pytorch
tags:
- cognet
- language-model
- aicl
- custom-architecture
- non-transformer
---

# CogNet-1B

A ~1.06B parameter **non-transformer** language model with a novel cognitive architecture featuring working, episodic, and semantic memory systems. CogNet uses cognitive routing with vectorized channel processing and hierarchical memory tiers, achieving O(n) per-layer complexity instead of O(n^2) for transformers.

## Architecture

| Parameter | Value |
|-----------|-------|
| **Hidden dim** | 2048 |
| **Blocks** | 16 (8 channels each) |
| **Channel dim** | 384 |
| **FF dim** | 8192 (Fused SwiGLU) |
| **Working memory slots** | 128 |
| **Episodic memory slots** | 256 |
| **Semantic memory slots** | 512 |
| **Tokenizer** | CharTokenizer (136 vocab) |
| **Normalization** | RMSNorm |
| **Positional encoding** | RoPE |

### Key Differences from Transformers

- **Cognitive routing**: Input is routed through parallel channels instead of attention heads
- **Hierarchical memory**: 3-tier memory system (working/episodic/semantic) with SDPA reads
- **O(n) per-layer complexity**: Channel processing is linear in sequence length (vs O(n^2) attention)
- **Vectorized channels**: All 8 channels processed in a single batched operation (no for-loops)
- **Fused SwiGLU**: Gate and up projections combined into a single matmul

## Optimized Training Pipeline

The `train_ultra.py` script includes the complete training pipeline with all optimizations:

### Data Pipeline (A-B-C-D-E)

| Part | Source | Description |
|------|--------|-------------|
| **A** | HuggingFace datasets | wikitext-103, codeparrot-clean, fineweb, oscar-fr, the-stack-smol, alpaca-cleaned, c4-en |
| **B** | CogNet HF repo data | Pre-tokenized .pt files from this repository |
| **C** | AICL repo | JSONL datasets, .aicl examples, source code, spec, tests (10x repeated) |
| **D** | HF scripts | Python/JSON/MD scripts from this repo (3x weight) |
| **E** | Synthetic data | Code templates + English + French sentences (~50M chars) |

All parts are merged, shuffled, and saved as a single `train_merged.pt` file.

### Optimizations

| # | Optimization | Benefit |
|---|-------------|---------|
| 1 | BF16 mixed precision | 2x throughput vs FP32 |
| 2 | RMSNorm + RoPE | No learned positional table |
| 3 | Vectorized channel processing | No Python for-loops over channels |
| 4 | SDPA/Flash Attention for memory tiers | Fused attention for memory reads |
| 5 | Fused SwiGLU | Single matmul for gate+up |
| 6 | Gradient checkpointing | ~3x memory savings |
| 7 | torch.compile() | Kernel fusion, reduced overhead |
| 8 | FSDP multi-GPU | Near-linear multi-GPU scaling |
| 9 | Fused AdamW | Faster optimizer step |
| 10 | CUDA prefetch pipeline | Overlaps data transfer with compute |
| 11 | Async checkpointing | Saves in background, no training pause |
| 12 | Sequence length warmup | 128 -> target over warmup period |
| 13 | 8-bit optimizer (optional) | 50% less VRAM for optimizer states |

### Real Benchmark

**No fabricated performance claims.** The training script runs a real benchmark at startup:

1. **3 warmup steps** to heat up compile caches and CUDA allocations
2. **10 measured steps** (forward + backward + optimizer) with `cuda.synchronize()`
3. Reports real **steps/sec** and **tokens/sec** on your hardware
4. Calculates **ETA** based on measured speed
5. Saves results to `benchmark_results.json`

Every log line shows `ETA: Xh` calculated from the measured speed.

## Files

### Optimized (V2) — Recommended

| File | Description |
|------|-------------|
| `cognet_1b_optimized.py` | **Optimized model architecture** (RMSNorm, RoPE, vectorized, SDPA, FusedSwiGLU) |
| `train_ultra.py` | **Main training script** (complete A-B-C-D-E pipeline + benchmark + all optimizations) |
| `run.py` | **Python launcher** (auto-detects GPUs, installs deps, launches torchrun) |
| `infer_optimized.py` | Inference with optimized model (generate, analyze, benchmark) |
| `benchmark.py` | Standalone benchmark (original vs optimized, scalability test) |
| `convert_checkpoint.py` | Convert original checkpoint to optimized format |
| `requirements.txt` | Python dependencies |
| `setup.sh` | Quick start setup script |

### Original — Legacy

| File | Description |
|------|-------------|
| `cognet_1b.py` | Original model architecture |
| `runpod_train_1b.py` | Original RunPod training script |
| `train_1b_final.py` | Previous training script |
| `train_1b_v2.py` | Previous training script v2 |
| `train_1b_v3.py` | Previous training script v3 |
| `train_bg.py` | Background training script |
| `train_pipeline.py` | Pipeline training script |
| `infer.py` | Original inference script |
| `chat_infer.py` | Chat-style inference |
| `gen_data_1b.py` | Synthetic data generation |
| `cognet_data_prep.py` | Standalone data prep |
| `config.json` | Model config |
| `tokenizer_v3.json` | CharTokenizer vocabulary |
| `data/` | AICL datasets and examples |

## Quick Start

```bash
# 1. Clone
git clone https://huggingface.co/thefinalboss/CogNet-1B
cd CogNet-1B

# 2. Install deps
pip install torch datasets huggingface_hub tokenizers

# 3. Set HF token (for data download)
export HF_TOKEN=your_token_here

# 4. Train — everything is automatic
python run.py
```

### Training Options

```bash
# Single GPU with all optimizations
python train_ultra.py --max-steps 100000 --compile --cuda-prefetch --seq-warmup --async-ckpt

# Multi-GPU with FSDP
torchrun --nproc_per_node=4 train_ultra.py --use-fsdp --max-steps 100000

# Use the Python launcher (auto-detects GPUs, installs deps)
python run.py --max-steps 100000 --hf-token hf_xxx

# Just prepare data (no training)
python run.py --prep-only

# Resume from checkpoint
python run.py --resume ./checkpoints_1b/cognet_1b_latest.pt

# 350M model (faster for testing)
python run.py --model-size 350m

# 8-bit optimizer (less VRAM)
python run.py --8bit
```

## Inference

```python
from cognet_1b_optimized import create_cognet_1b_optimized
import torch

# Create model
model = create_cognet_1b_optimized(vocab_size=136, max_seq_len=512)

# Load checkpoint
ckpt = torch.load('checkpoints/cognet_best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Generate
prompt = torch.tensor([[2]])  # BOS token
output = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=50)

# Decode (CharTokenizer)
vocab = {0: '', 1: '', 2: '', 3: ''}
for i in range(4, 136):
    vocab[i] = chr([*range(32,127), *[
        192,193,194,195,196,197,199,200,201,202,203,204,205,206,207,
        210,211,212,213,214,217,218,219,220,224,225,226,227,228,229,
        231,232,233,234,235,236,237,238,239,242,243,244,245,246,249,
        250,251,252,253,255
    ]][i-4])

text = ''.join(vocab.get(t, '') for t in output[0].tolist() if t not in (0,1,2,3))
print(text)
```

Or use the inference script:

```bash
python infer_optimized.py generate --prompt "The future of AI is" --max-tokens 100
python infer_optimized.py benchmark
```

## Benchmark Your Hardware

```bash
# Full benchmark: original vs optimized + scalability test
python benchmark.py

# Quick benchmark during training (automatic)
python train_ultra.py --max-steps 20
# The first 13 steps are: 3 warmup + 10 benchmark = real speed measurement
```

## Config Files

YAML configs are available in `configs/`:

| Config | Description |
|--------|-------------|
| `1b_single_gpu.yaml` | 1B model, single GPU |
| `1b_fsdp.yaml` | 1B model, multi-GPU FSDP |
| `350m_fast.yaml` | 350M model, fast iteration |
