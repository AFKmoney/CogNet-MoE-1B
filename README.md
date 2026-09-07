# CogNet-MoE-1B

> **Mixture-of-Experts adaptation of [CogNet-1B](https://github.com/AFKmoney/CogNet-1B), strictly CogNet-native.**
> 100% non-transformer. No self-attention. No O(n²). No FlashAttention/GQA/KV-cache.
> The 8 cognitive channels = the 8 MoE experts. Routing is the O(n) coherence router.

[![Architecture](https://img.shields.io/badge/architecture-CogNet--native-blue)]()
[![MoE](https://img.shields.io/badge/MoE-8%20experts%20top--2-purple)]()
[![Complexity](https://img.shields.io/badge/complexity-O(n)-green)]()
[![License](https://img.shields.io/badge/license-MIT-yellow)]()

## What is this?

CogNet-MoE-1B adapts the original CogNet-1B (a non-transformer LLM with cognitive routing + 3-tier hierarchical memory) into a Mixture-of-Experts variant. The key architectural decision: **the 8 channels of the CognitiveRouter become the 8 MoE experts**, and the O(n) coherence router does the routing. No separate transformer-style gate is introduced.

This repo also includes:
- **EDT pipeline** (Expert Decoupled Training) adapted to CogNet-native MoE — 4 phases, ~35× token reduction vs full Chinchilla
- **Chinchilla scaling analysis** with active-param interpretation (47.66B tokens optimal → 1.36B tokens via EDT Scénario C)
- **Proprietary BPE tokenizer** (vocab 16,384, FR+EN+code, ByteLevel) replacing the original CharTokenizer (vocab 136)
- **Training time estimate** for the "real 1B" — ~10 days on RTX 3090, ~6 hours on H100

## ⚠ Architectural constraint

CogNet is **100% non-transformer**. The following patterns are **INAPPLICABLE**:

| Pattern transformer | Statut | Alternative CogNet-native |
|---|---|---|
| Self-attention, multi-head attention | ❌ INAPPLICABLE | Cognitive routing O(n) |
| FlashAttention / FlashAttention-2 | ❌ INAPPLICABLE | No inter-token attention to optimize |
| GQA / MQA | ❌ INAPPLICABLE | No multi-head |
| KV-cache | ❌ INAPPLICABLE | 3-tier memory with fixed SDPA slots |
| Sliding-window attention | ❌ INAPPLICABLE | No attention at all |
| Mixtral-style MoE gate (Linear D→N) | ❌ INAPPLICABLE | CoherenceRouter O(n) does the routing |
| O(n²) complexity | ❌ INAPPLICABLE | Strict O(n) per layer |

The only SDPA authorized is on the **fixed memory slots** (128+256+512 = 896 slots), which is O(1) per token in seq_len.

## Architecture

```
TokenEncoder (RoPE + RMSNorm, separable for EDT Phase 2b)
  → 16× CogNetMoEBlock
      ├─ CognitiveExpertRouter              ← Router + MoE UNIFIED
      │     ├─ CoherenceRouter O(n)         (query × mean_key, softmax over 8 channels)
      │     │     → produces routing_weights (B,T,8) — NO separate gate
      │     ├─ to_channels                  Linear(D → 8×D)
      │     ├─ Top-2 sparse on routing_weights (noisy top-k, Shazeer 2017)
      │     ├─ 8 experts = 8 channels FusedSwiGLU(D=2048, ff=8192)
      │     ├─ aux_loss : N · Σ f_i · P_i   (Switch Transformer load-balancing)
      │     ├─ z_loss   : mean(router_logits²)  (ST-MoE)
      │     └─ clamp(max=10)                (anti-explosion, EDT)
      ├─ ParallelHierarchicalMemory         (3-tier Working/Episodic/Semantic, SDPA reads)
      │     ├─ Working  : 128 slots (short-term)
      │     ├─ Episodic : 256 slots (mid-term)
      │     └─ Semantic : 512 slots (long-term)
      │     → O(1) per token in seq_len (fixed slots, bounded SDPA)
      └─ CompositionalReasoner              (hyperdim role-filler binding)
  → final RMSNorm
  → output_proj (weight-tied with token_emb)
```

**Complexity: STRICTLY O(n) per layer.** No inter-token attention. No transformer-style gate. Cognitive routing and MoE are unified — one mechanism decides both cognitive activation AND expert selection.

## Key numbers

| Metric | Value |
|---|---|
| Architecture | 100% non-transformer (cognitive routing + 3-tier memory) |
| Total params (MoE) | 7.21B |
| Active params / token (top-2/8) | 2.38B |
| Capacity multiplier vs dense | 3.03× |
| Tokenizer | BPE proprietary 16,384 (ByteLevel, FR+EN+code) |
| Chinchilla optimal (reference) | 47.66B BPE-tokens |
| EDT Scénario C (recommended) | 1.36B BPE-tokens (~35× reduction) |
| Training time (RTX 3090, MFU 18%, PGSU) | ~10 days |
| Training time (H100 SXM, MFU 48%) | ~6 hours |
| Complexity per layer | O(n) strict (no O(n²) operation) |

## Files

| File | Role |
|---|---|
| `cognet_tokenizer.py` | BPE tokenizer (vocab 16k, FR+EN+code, ByteLevel) |
| `cognet_tokenizer.json` | Trained tokenizer (HuggingFace format) |
| `cognet_moe.py` | `CognitiveExpertRouter` (CogNet-native MoE) + `CogNetMoE1B` (full model) |
| `edt_pipeline.py` | EDT 4-phase pipeline + PGSU + prerequisites check + aux-loss clamping |
| `chinchilla_scaling.py` | Params breakdown + Chinchilla + CharTokenizer vs BPE 16k comparison |
| `run_cognet_moe.py` | CLI orchestrator with `--self-test` flag |
| `fast_train.py` | Fast training stack (memmap dataset, fast router, FastTrainer) |
| `phase_routed_moe.py` | Phase-Routed MoE (on-the-fly experts, infinite training) |
| `run_infinite.py` | Lifelong multi-phase training orchestrator |
| `FAST_TRAINING.md` | Fast + infinite training guide (French) |
| `training_time_estimate.json` | Training time estimates (RTX 3090/4090, A100, H100, H200) |
| `CogNet-MoE-1B_Whitepaper.pdf` | Technical whitepaper (French, 18 pages) |
| `source/` | Original CogNet-1B code (cloned from GitHub) |

## Quickstart

```bash
# 1. Install dependencies
pip install torch tokenizers bitsandbytes

# 2. Full self-test (tokenizer + MoE + EDT + Chinchilla + CogNet-native audit)
python3 run_cognet_moe.py --self-test

# 3. Chinchilla report (CharTokenizer vs BPE 16k comparison)
python3 chinchilla_scaling.py

# 4. Train BPE tokenizer on a real corpus
python3 -c "from cognet_tokenizer import train_cognet_tokenizer; \
  train_cognet_tokenizer(corpus='/path/to/corpus.txt', vocab_size=16384)"

# 5. Full training (requires GPU 3090+ and text dataset)
python3 run_cognet_moe.py \
    --dataset-path /path/to/text_corpus.txt \
    --max-seq-len 512 \
    --n-experts 8 \
    --top-k 2 \
    --phase3-tokens 1361634450 \
    --aux-loss-weight 0.01 \
    --pgsu-n-active 4 \
    --use-bf16 \
    --use-8bit-optimizer
```

## Self-test

The self-test in `cognet_moe.py` verifies at runtime:
1. ✅ Forward + backward pass functional
2. ✅ All 16 experts (8 channels × 2 blocks in test) receive non-zero gradient
3. ✅ Aux loss and z loss finite (no NaN/Inf)
4. ✅ **No transformer-style gate detected** (module name audit)
5. ✅ Routing = coherence O(n) only (architectural verification)
6. ✅ Memory SDPA on fixed slots (128+256+512 = 896) → O(1)/token in seq_len

```bash
python3 cognet_moe.py
# → "All self-tests passed! CogNet-native MoE is correct."
```

## EDT pipeline (4 phases)

| Phase | Trains | Tokens / steps | Time (3090) |
|---|---|---|---|
| **1** | 128 independent experts (16×8) — MSE identity | 2000 steps/expert | ~1.2h |
| **2a** | CoherenceRouter + Memory + Composer (1 symmetry break step) | 1 step × 16 blocks | < 6 min |
| **2b** | TokenEncoder (separable, no MoE blocks) | 125M BPE tokens | ~20 min |
| **3** | Joint fine-tune with PGSU (n_active=4) + aux-loss clamping | 1.36B BPE tokens (Scénario C) | ~175h |
| **Total** | | | **~10 days (RTX 3090, MFU 18%)** |

## CogNet-native optimizations (no flash-attn)

**INAPPLICABLE** (transformer-centric, destroy CogNet's architecture):
- ❌ FlashAttention / FlashAttention-2 (no inter-token attention)
- ❌ GQA / MQA (no multi-head)
- ❌ KV-cache optimizations (slots SDPA, not KV-cache)
- ❌ Sliding-window attention (no attention at all)

**APPLICABLE** (CogNet-native):
1. Fused channel matmul (coherence + to_channels)
2. Memory tier batching (already done in source)
3. Working memory SRAM caching (H100/A100 only)
4. Triton kernel for CompositionalReasoner
5. Sparse dispatch optimized for MoE
6. Gradient checkpointing (already done)
7. PGSU (n_active=4/16)
8. bf16 + bitsandbytes 8-bit Adam
9. torch.compile (mode=reduce-overhead)
10. Batch packing

## Fast & infinite training

- **Fast stack** (`fast_train.py`): pre-tokenized memmap dataset, single-pass
  fast router (numerically identical), `torch.compile`, fused/8-bit optimizers,
  PGSU, seq-len curriculum, resumable checkpoints, DDP-ready.
  Projected: 1.36B tokens in ~2.5–4 days on RTX 3090 (was ~10 days), ~2–3h on H100.
- **Phase-Routed MoE** (`phase_routed_moe.py` + `run_infinite.py`): phase-conditioned
  CogNet-native routing, on-the-fly expert creation (clone-busiest + noise),
  frozen old experts (no catastrophic forgetting), infinite `.pt` checkpoints.
  Each new billion tokens costs ~15–25% of a full train.

See **[FAST_TRAINING.md](FAST_TRAINING.md)** (French) for the full guide.

```bash
python3 fast_train.py --build-bin --txt corpus.txt --tokenizer cognet_tokenizer.json --out data/p0
python3 run_infinite.py --bins data/p0.bin --tokens-per-phase 1000000000 --compile reduce-overhead
```

## License

Original CogNet-1B code: see `source/` (original license preserved).
MoE + EDT + Chinchilla + BPE modifications: MIT.
