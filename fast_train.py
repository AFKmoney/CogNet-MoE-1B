#!/usr/bin/env python3
"""
fast_train.py — Training rapide CogNet-MoE : milliards de tokens sans les jours de GPU
======================================================================================

Pourquoi l'entraînement actuel prend des jours (goulots identifiés)
--------------------------------------------------------------------
  1. Dataset : `CognetDataset.make_iter_fn` fait open()+seek()+read()+BPE-encode
     PAR SAMPLE. Des millions d'open/encode Python → CPU-bound, GPU idle.
     → Pré-tokenisation UNE fois en `.bin` memmap + fenêtres contiguës (100% tokens utiles).
  2. Router legacy : cohérence calculée 2× (softmax + re-calcul logits).
     → Single-pass (weights + logits en 1 passage).
  3. Dispatch MoE : boucle naïve avec passes redondantes (one_hot, masques multiples).
     → 1 scatter_add + boucle sur actifs, torch.compile-friendly.
  4. Micro-batch minuscule (B=4, T=512 = 2k tokens) → GPU sous-alimenté.
     → Gros batch + grad accum + packing implicite (fenêtres contiguës, 0 padding).
  5. Pas de torch.compile, pas de fused optimizer, pas de TF32/SDP flash.
     → enable_fast_mode() + compile + AdamW fused / 8-bit.
  6. Phase 1 EDT séquentielle (128 experts × 2000 steps, 1 par 1).
     → fast_phase1_parallel() : 8 experts d'un bloc en batch partagé.
  7. Pas de curriculum, pas de resume exact, pas de DDP.
     → Curriculum seq-len, checkpoints resumables, DDP torchrun-ready.

Gains attendus (honnêtes, à iso-tokens Phase 3 = 1.36B, Scénario C) :
  - RTX 3090 : ~10 jours → ~2.5-4 jours (dataloader + compile + fused + PGSU).
  - + stacking progressif (4→8→16 blocs) : → ~1.5-2 jours.
  - RTX 4090 : ~3.2 jours → < 1 jour.  H100 : ~6h → ~2-3h.
  - Le VRAI déblocage « milliards de tokens » vient du combo avec
    phase_routed_moe.py : chaque nouveau milliard (nouvelle phase) ne coûte
    que ~20% d'un full-train (anciens experts gelés). Voir FAST_TRAINING.md.

Contenu :
  - build_bin_dataset() : .txt → .bin memmap pré-tokenisé (+ .meta.json).
  - MMapTokens : lecture memmap + échantillonnage de fenêtres (B, T).
  - MixedPhaseLoader : batch courant + replay d'anciennes phases (lifelong).
  - FastCognitiveExpertRouter : forward optimisé, poids-identique au legacy.
  - convert_to_fast() : swap in-place (checkpoints interchangeables).
  - FastTrainerConfig / FastTrainer : boucle Phase-3 rapide (compile, fused,
    PGSU, curriculum, resume, DDP).
  - fast_phase1_parallel() : Phase 1 EDT parallélisée par bloc.
  - benchmark() : mesure tok/s forward+backward.

Usage :
    # 1. Pré-tokeniser UNE fois :
    python3 fast_train.py --build-bin --txt corpus.txt --tokenizer cognet_tokenizer.json --out data/train
    # 2. Entraîner vite :
    python3 fast_train.py --train --bin data/train.bin --tokens 1361634450 --compile --batch-size 16
    # 3. Mesurer :
    python3 fast_train.py --benchmark
    python3 fast_train.py --self-test   # tiny CPU
"""

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))

from cognet_moe import CogNetMoE1B, CognitiveExpertRouter, create_cognet_moe_1b  # noqa: E402

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ═══════════════════════════════════════════════════════════════════════
#  Mode rapide global (flags torch, une fois au démarrage)
# ═══════════════════════════════════════════════════════════════════════

def enable_fast_mode():
    """Active les flags matmul/alloc rapides (CogNet-native : pas de flash-attn à activer,
    la mémoire utilise déjà SDPA sur slots fixes)."""
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("[Fast] fast mode ON (tf32, bf16-reduction, expandable_segments)")


# ═══════════════════════════════════════════════════════════════════════
#  Dataset memmap pré-tokenisé (le fix #1 : tue le goulot CPU)
# ═══════════════════════════════════════════════════════════════════════

def _get_tokenizer(tokenizer) -> Tuple[Callable[[str], List[int]], int, int]:
    """
    Retourne (encode_fn, vocab_size, eos_id) depuis :
      - un path vers cognet_tokenizer.json (CognetTokenizer lazy),
      - un objet avec .encode/.vocab_size/.eos_id,
      - None → fallback byte-level (tests uniquement).
    """
    if tokenizer is None:
        def _byte_encode(s: str) -> List[int]:
            return [2] + [ord(c) % 255 + 4 for c in s[:100000]] + [3]
        return _byte_encode, 512, 3
    if isinstance(tokenizer, (str, Path)):
        from cognet_tokenizer import CognetTokenizer
        tok = CognetTokenizer(tokenizer_path=str(tokenizer))
        return (lambda s: tok.encode(s, add_special_tokens=False)), tok.vocab_size, tok.eos_id
    return ((lambda s: tokenizer.encode(s, add_special_tokens=False))
            if hasattr(tokenizer, "encode") else tokenizer), \
        getattr(tokenizer, "vocab_size", 16384), getattr(tokenizer, "eos_id", 2)


def build_bin_dataset(
    txt_path: str,
    out_prefix: str,
    tokenizer=None,
    chunk_chars: int = 1_000_000,
    add_eos_between_chunks: bool = True,
    dtype: Optional[str] = None,
) -> Dict:
    """
    Pré-tokenise un corpus .txt en `.bin` (streaming, RAM O(1)).

    - Une seule passe, écriture incrémentale (pas de 2× pic mémoire).
    - uint16 si vocab < 65536 (BPE 16k ✓ → 2 octets/token : 1.36B tokens = 2.7 GB).
    - Écrit out_prefix.bin + out_prefix.meta.json {n_tokens, dtype, vocab_size}.
    """
    if not _HAS_NUMPY:
        raise ImportError("numpy requis pour build_bin_dataset")
    encode, vocab_size, eos_id = _get_tokenizer(tokenizer)
    if dtype is None:
        dtype = "uint16" if vocab_size < 65536 else "uint32"
    np_dtype = np.dtype(dtype)
    assert np.iinfo(np_dtype).max >= vocab_size, f"{dtype} trop petit pour vocab {vocab_size}"

    bin_path = out_prefix + ".bin"
    meta_path = out_prefix + ".meta.json"
    os.makedirs(os.path.dirname(os.path.abspath(bin_path)) or ".", exist_ok=True)

    n_tokens = 0
    t0 = time.time()
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f_in, \
            open(bin_path, "wb") as f_out:
        first = True
        while True:
            chunk = f_in.read(chunk_chars)
            if not chunk:
                break
            ids = encode(chunk)
            if add_eos_between_chunks and not first:
                ids = [eos_id] + ids
            first = False
            arr = np.asarray(ids, dtype=np_dtype)
            arr.tofile(f_out)
            n_tokens += len(ids)
            if n_tokens % 5_000_000 < len(ids):
                dt = time.time() - t0
                print(f"[Bin] {n_tokens/1e6:.1f}M tokens ({n_tokens/max(1,dt):.0f} tok/s)")
    meta = {"n_tokens": n_tokens, "dtype": dtype, "vocab_size": vocab_size,
            "source": os.path.basename(txt_path), "eos_id": eos_id}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    size_gb = os.path.getsize(bin_path) / 1e9
    print(f"[Bin] ✓ {bin_path} : {n_tokens:,} tokens, {size_gb:.2f} GB ({dtype}) "
          f"en {time.time()-t0:.1f}s")
    return {"bin_path": bin_path, "meta_path": meta_path, **meta}


class MMapTokens:
    """Corpus pré-tokenisé en lecture memmap + échantillonnage de fenêtres contiguës."""

    def __init__(self, bin_path: str):
        if not _HAS_NUMPY:
            raise ImportError("numpy requis pour MMapTokens")
        meta_path = os.path.splitext(bin_path)[0] + ".meta.json"
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.n_tokens = int(self.meta["n_tokens"])
        self.vocab_size = int(self.meta["vocab_size"])
        self.arr = np.memmap(bin_path, mode="r", dtype=np.dtype(self.meta["dtype"]),
                             shape=(self.n_tokens,))

    def sample_batch(self, batch_size: int, seq_len: int,
                     rng: "np.random.Generator") -> torch.Tensor:
        """(B, T) fenêtres contiguës aléatoires — 0 padding, 100% tokens utiles."""
        assert self.n_tokens > seq_len + 1
        starts = rng.integers(0, self.n_tokens - seq_len, size=batch_size)
        offs = starts[:, None] + np.arange(seq_len)[None, :]
        batch = self.arr[offs]  # copie (B, T) — le seul coût CPU, ~µs
        return torch.from_numpy(np.ascontiguousarray(batch)).long()

    def sequential_batch(self, start_token: int, batch_size: int, seq_len: int) -> torch.Tensor:
        n = batch_size * seq_len
        sl = self.arr[start_token:start_token + n]
        if sl.shape[0] < n:  # wrap-around (streaming infini)
            extra = self.arr[:n - sl.shape[0]]
            sl = np.concatenate([sl, extra])
        return torch.from_numpy(np.ascontiguousarray(sl).reshape(batch_size, seq_len)).long()


class MixedPhaseLoader:
    """
    Batches pour lifelong learning : (1 - replay_ratio) du bin courant (phase p)
    + replay_ratio de bins antérieurs (ancrage anti-oubli du router).
    Retourne (input_ids, phase_ids) avec phase_ids (B,) par-sample.
    """

    def __init__(self, bin_paths: List[str], current_idx: int = 0,
                 replay_ratio: float = 0.0, seed: int = 42):
        assert _HAS_NUMPY
        self.bins = [MMapTokens(p) for p in bin_paths]
        self.current_idx = current_idx
        self.replay_ratio = replay_ratio
        self.rng = np.random.default_rng(seed)

    def set_current(self, idx: int):
        self.current_idx = idx

    def next_batch(self, batch_size: int, seq_len: int, device: str = "cpu",
                   pin_memory: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        B = batch_size
        if self.replay_ratio > 0 and self.current_idx > 0:
            n_replay = int(B * self.replay_ratio)
            n_cur = B - n_replay
        else:
            n_replay, n_cur = 0, B
        parts, phases = [], []
        if n_cur > 0:
            parts.append(self.bins[self.current_idx].sample_batch(n_cur, seq_len, self.rng))
            phases.append(torch.full((n_cur,), self.current_idx, dtype=torch.long))
        for _ in range(n_replay):
            b = int(self.rng.integers(0, self.current_idx))
            parts.append(self.bins[b].sample_batch(1, seq_len, self.rng))
            phases.append(torch.tensor([b], dtype=torch.long))
        ids = torch.cat(parts, dim=0)
        pids = torch.cat(phases, dim=0)
        # Mélange intra-batch (évite un bloc replay contigu).
        perm = torch.randperm(B, generator=torch.Generator().manual_seed(int(self.rng.integers(0, 2**31))))
        ids, pids = ids[perm], pids[perm]
        if pin_memory and device.startswith("cuda"):
            ids = ids.pin_memory()
        return ids.to(device, non_blocking=True), pids.to(device, non_blocking=True)


# ═══════════════════════════════════════════════════════════════════════
#  FastCognitiveExpertRouter — même maths, forward optimisé
# ═══════════════════════════════════════════════════════════════════════

class FastCognitiveExpertRouter(CognitiveExpertRouter):
    """
    Sous-classe 100% poids-compatible du CognitiveExpertRouter legacy :
    AUCUN nouveau paramètre → state_dict interchangeable dans les 2 sens.

    Optimisations (mathématiques identiques, à l'arrondi fp près) :
      1. Single-pass coherence : q, k, mean calculés UNE fois (le legacy
         appelait coherence_router(x) PUIS recalculait q/k/mean).
      2. Logits fp32 + softmax fp32 (stabilité), cast à la fin.
      3. Dispatch : 1 scatter_add_ (N, C) au lieu de one_hot (N, K, C) + sum,
         boucle sur C avec indexation booléenne directe (torch.compile fuse).
      4. 0 passe redondante sur (B, T, C).
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape
        C, K = self.num_channels, self.top_k
        n_tokens = B * T

        # ── 1. Cohérence O(n) single-pass ──
        q = self.coherence_router.query(x)       # (B, T, C)
        k = self.coherence_router.key(x)         # (B, T, C)
        mean_key = k.mean(dim=1, keepdim=True)   # (B, 1, C)
        router_logits = (q.float() * mean_key.float())
        routing_weights = F.softmax(router_logits, dim=-1).to(x.dtype)

        # ── 2. Noisy top-k ──
        if self.training and self.noise_std > 0:
            noisy = router_logits + torch.randn_like(router_logits) * self.noise_std
        else:
            noisy = router_logits
        topk_weights, topk_indices = torch.topk(noisy, K, dim=-1)
        topk_weights = F.softmax(topk_weights.float(), dim=-1).to(x.dtype)

        # ── 3. Projection (1 GEMM) ──
        channel_input = self.to_channels(x).view(B, T, C, D)

        # ── 4. Dispatch sparse : 1 scatter_add + boucle actifs ──
        flat_idx = topk_indices.reshape(n_tokens, K)
        flat_w = topk_weights.reshape(n_tokens, K)
        expert_w = torch.zeros(n_tokens, C, device=x.device, dtype=x.dtype)
        expert_w.scatter_add_(1, flat_idx, flat_w)
        f = (expert_w > 0).float().mean(dim=0)
        P = routing_weights.reshape(n_tokens, C).float().mean(dim=0)

        chan_flat = channel_input.reshape(n_tokens, C, D)
        combined = torch.zeros(n_tokens, D, device=x.device, dtype=x.dtype)
        for i in range(C):
            w_i = expert_w[:, i]
            if not bool(torch.any(w_i > 0).item()):
                continue
            tok = w_i > 0
            out_i = self.experts[i](chan_flat[tok, i])
            combined[tok] += w_i[tok].unsqueeze(-1).to(out_i.dtype) * out_i

        # ── 5. Norm + résiduel ──
        out = self.norm(combined.view(B, T, D))
        out = x + out

        # ── 6. Aux losses (formules legacy inchangées) ──
        aux_loss = C * (f * P.to(f.dtype)).sum()
        z_loss = router_logits.square().mean()

        with torch.no_grad():
            moe_max_load = f.max()
            moe_min_load = f.min()
            Pp = P.clamp_min(1e-8)
            moe_routing_entropy = -(Pp * Pp.log()).sum()
            coherence_entropy = -(
                routing_weights.float() * (routing_weights.float() + 1e-8).log()
            ).sum(-1).mean()

        stats = {
            "moe_aux_loss": aux_loss,
            "moe_z_loss": z_loss,
            "moe_max_load": moe_max_load.detach(),
            "moe_min_load": moe_min_load.detach(),
            "moe_routing_entropy": moe_routing_entropy.detach(),
            "moe_expert_usage": f.detach(),
            "routing_entropy": coherence_entropy.detach(),
        }
        return out, stats


def convert_to_fast(model: CogNetMoE1B) -> CogNetMoE1B:
    """Swap in-place chaque CognitiveExpertRouter → Fast (poids copiés, ckpt compatibles)."""
    for block in model.blocks:
        legacy = block.cognitive_expert_router
        if isinstance(legacy, FastCognitiveExpertRouter):
            continue
        ff_dim = legacy.experts[0].w_down.weight.shape[1]  # w_down: (hidden, ff)
        fast = FastCognitiveExpertRouter(
            hidden_dim=legacy.hidden_dim, num_channels=legacy.num_channels,
            ff_dim=ff_dim, top_k=legacy.top_k,
            dropout=legacy.experts[0].dropout.p,
            aux_loss_weight=legacy.aux_loss_weight,
            z_loss_weight=legacy.z_loss_weight,
            noise_std=legacy.noise_std,
        )
        fast.load_state_dict(legacy.state_dict())
        block.cognitive_expert_router = fast
    return model


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1 EDT parallélisée par bloc (8 experts en batch partagé)
# ═══════════════════════════════════════════════════════════════════════

def fast_phase1_parallel(
    model: CogNetMoE1B,
    hidden_fn: Callable[[int, int], torch.Tensor],
    steps_per_expert: int = 2000,
    batch_size: int = 64,
    seq_len: int = 512,
    lr: float = 3e-4,
    target_loss: float = 0.05,
    perturbation_scale: float = 0.02,
    device: str = "cuda",
    log_every: int = 500,
) -> Dict:
    """
    Phase 1 EDT accélérée : au lieu d'entraîner 128 experts SÉQUENTIELLEMENT
    (1 optimizer + 1 boucle par expert), on entraîne les 8 experts d'un bloc
    EN PARALLÈLE sur le MÊME batch de hidden states (8 forwards groupés,
    1 seul optimizer par bloc, perturbations vectorisées).

    Speedup attendu : ~4-6× vs phase1_experts() séquentielle.
    """
    from edt_pipeline import _get_expert_perturbation  # lazy (léger)
    dev = device if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    model.eval()
    t0 = time.time()
    stats = {"blocks": []}

    for b in range(model.num_blocks):
        router = model.blocks[b].cognitive_expert_router
        experts = router.experts if hasattr(router, "experts") else router.experts
        C = len(experts)
        for e in experts:
            e.train()
        for p in model.parameters():
            p.requires_grad = False
        for e in experts:
            for p in e.parameters():
                p.requires_grad = True
        params = [p for e in experts for p in e.parameters()]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

        # Perturbations uniques par expert (Fix EDT #1), pré-calculées.
        perts = []
        for e in range(C):
            mult, add = _get_expert_perturbation(
                b, e, model.hidden_dim, torch.device(dev),
                perturbation_scale, "both", 42)
            perts.append((mult, add))

        losses = []
        for step in range(steps_per_expert):
            with torch.no_grad():
                h_in = hidden_fn(batch_size, seq_len).to(dev).float()
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for e_idx, e in enumerate(experts):
                h_out = e(h_in)
                mult, add = perts[e_idx]
                tgt = h_in * (1.0 + mult) + add
                loss = F.mse_loss(h_out, tgt) / C
                loss.backward()
                total += float(loss.item()) * C / C
            opt.step()
            losses.append(total / 1.0)
            if (step + 1) % log_every == 0:
                print(f"[FastP1] bloc {b} step {step+1}/{steps_per_expert} loss={losses[-1]:.4f}")
            if losses[-1] < target_loss and step > 100:
                break
        for e in experts:
            for p in e.parameters():
                p.requires_grad = False
        stats["blocks"].append({"block": b, "final_loss": losses[-1], "steps": len(losses)})
        print(f"[FastP1] bloc {b}/{model.num_blocks-1} ✓ loss={losses[-1]:.4f} ({len(losses)} steps)")

    for p in model.parameters():
        p.requires_grad = True
    stats["time_s"] = time.time() - t0
    print(f"[FastP1] terminé en {stats['time_s']:.1f}s")
    return stats


# ═══════════════════════════════════════════════════════════════════════
#  FastTrainer — boucle Phase-3 rapide (curriculum, PGSU, compile, resume)
# ═══════════════════════════════════════════════════════════════════════

_PEAK_FLOPS = {  # bf16 dense, TFLOPS
    "3090": 142e12, "4090": 330e12, "a100": 312e12, "h100": 989e12, "h200": 989e12,
    "a6000": 155e12, "l40": 181e12, "4070": 194e12, "4080": 330e12,
}


def _peak_flops_current() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0).lower().replace("nvidia ", "").replace("geforce ", "")
    for k, v in _PEAK_FLOPS.items():
        if k in name:
            return v
    return None


@dataclass
class FastTrainerConfig:
    batch_size: int = 16
    grad_accum: int = 4
    lr: float = 1e-4
    lr_min_ratio: float = 0.1
    warmup_steps: int = 500
    aux_loss_weight: float = 0.05
    z_loss_weight: float = 1e-3
    aux_clamp: float = 10.0
    z_clamp: float = 10.0
    max_grad_norm: float = 1.0
    use_bf16: bool = True
    compile_mode: Optional[str] = "reduce-overhead"  # None pour désactiver
    optimizer: str = "adamw-fused"                   # adamw | adamw-fused | adamw8bit
    pgsu_n_active: int = 0          # 0 = désactivé (full), sinon rotation PGSU
    seq_stages: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 512)])
    log_every: int = 50
    ckpt_every_steps: int = 1000
    save_optimizer: bool = False    # True = resume exact (lourd : +2× params)
    seed: int = 42


class FastTrainer:
    """
    Boucle d'entraînement joint rapide (remplace phase3_joint pour scaler).

    - batch_fn(batch_size, seq_len) -> Tensor(B, T) fourni par le caller
      (MMapTokens.sample_batch typiquement).
    - Curriculum seq-len via seq_stages=[(tokens_start, T), ...].
    - PGSU optionnel (rotation de n_active blocs, PGSU inline).
    - torch.compile (recommandé : 'reduce-overhead' ; 'max-autotune' sur H100).
    - LR cosine manuel (robuste), checkpoints resumables, DDP-ready.
    """

    def __init__(self, model: CogNetMoE1B, cfg: FastTrainerConfig,
                 device: str = "cuda", ckpt_dir: str = "./fast_ckpts"):
        self.cfg = cfg
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.raw_model = model.to(self.device)
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        # DDP (torchrun) — opt-in via env.
        self.rank, self.world = 0, 1
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ and torch.cuda.is_available():
            import torch.distributed as dist
            dist.init_process_group("nccl")
            self.rank = dist.get_rank()
            self.world = dist.get_world_size()
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
            print(f"[Fast] DDP rank {self.rank}/{self.world}")

        self.step = 0
        self.tokens_seen = 0
        self.loss_ema: Optional[float] = None
        self.pgsu_step = 0

        if cfg.compile_mode and self.device.startswith("cuda"):
            try:
                self.model = torch.compile(self.raw_model, mode=cfg.compile_mode)
                print(f"[Fast] torch.compile({cfg.compile_mode})")
            except Exception as e:
                print(f"[Fast] compile KO ({e}) → eager")
                self.model = self.raw_model
        else:
            self.model = self.raw_model

        if self.world > 1:
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, find_unused_parameters=cfg.pgsu_n_active > 0)
        self.opt = self._make_optimizer()

    def _make_optimizer(self):
        params = [p for p in self.raw_model.parameters() if p.requires_grad]
        name = self.cfg.optimizer.lower()
        if name == "adamw8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(params, lr=self.cfg.lr, weight_decay=0.01)
            except ImportError:
                print("[Fast] bitsandbytes absent → AdamW")
        if name == "adamw-fused" and self.device.startswith("cuda"):
            try:
                return torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=0.01, fused=True)
            except Exception:
                pass
        return torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=0.01)

    def seq_len_at(self, tokens: int) -> int:
        T = self.cfg.seq_stages[0][1]
        for start, t in sorted(self.cfg.seq_stages):
            if tokens >= start:
                T = t
        return T

    def lr_at(self, total_steps: int) -> float:
        s, w = self.step, max(1, self.cfg.warmup_steps)
        if s < w:
            return self.cfg.lr * (s + 1) / w
        prog = min(max((s - w) / max(1, total_steps - w), 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return self.cfg.lr * (self.cfg.lr_min_ratio + (1 - self.cfg.lr_min_ratio) * cos)

    def _apply_pgsu(self):
        """Rotation PGSU inline (n_active blocs + encoder/final_norm toujours actifs)."""
        n = self.cfg.pgsu_n_active
        m = self.raw_model
        if not n or n >= m.num_blocks:
            for p in m.parameters():
                p.requires_grad = True
            return list(range(m.num_blocks))
        start = self.pgsu_step % m.num_blocks
        active = sorted((start + i) % m.num_blocks for i in range(n))
        for b in range(m.num_blocks):
            on = b in active
            for p in m.blocks[b].parameters():
                p.requires_grad = on
        for p in m.encoder.parameters():
            p.requires_grad = True
        for p in m.final_norm.parameters():
            p.requires_grad = True
        self.pgsu_step += 1
        return active

    def train(
        self,
        batch_fn: Callable[[int, int], torch.Tensor],
        total_tokens: int,
        total_steps: Optional[int] = None,
    ) -> Dict:
        cfg = self.cfg
        # Steps estimés au T max (le curriculum ajuste T à la volée).
        T_max = max(t for _, t in cfg.seq_stages)
        est_steps = total_steps or max(1, total_tokens // (cfg.batch_size * T_max))
        peak = _peak_flops_current()
        active_params = self.raw_model.count_parameters()["active_per_token"]
        t0 = time.time()
        is_cuda = self.device.startswith("cuda")

        if self.rank == 0:
            print(f"[Fast] objectif {total_tokens/1e9:.2f}B tokens, ~{est_steps} steps, "
                  f"B={cfg.batch_size} accum={cfg.grad_accum} T_stages={cfg.seq_stages}")

        use_amp = cfg.use_bf16 and is_cuda
        while self.tokens_seen < total_tokens:
            T = self.seq_len_at(self.tokens_seen)
            active = self._apply_pgsu()
            lr = self.lr_at(est_steps)
            for g in self.opt.param_groups:
                g["lr"] = lr
            self.opt.zero_grad(set_to_none=True)

            acc_lm, acc_aux, acc_max = 0.0, 0.0, 0.0
            for _ in range(cfg.grad_accum):
                ids = batch_fn(cfg.batch_size, T)
                if not torch.is_tensor(ids):
                    ids = torch.as_tensor(ids, dtype=torch.long)
                ids = ids.to(self.device, non_blocking=True)
                sync_ctx = self.model.no_sync() if (self.world > 1 and _ != cfg.grad_accum - 1
                                                    and hasattr(self.model, "no_sync")) \
                    else _nullcontext()
                with sync_ctx:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _nullcontext():
                        out = self.model(ids, return_stats=False)
                        logits = out["logits"]
                        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                             ids[:, 1:].reshape(-1))
                        aux = out["moe_aux_loss"].clamp(max=cfg.aux_clamp)
                        z = out["moe_z_loss"].clamp(max=cfg.z_clamp)
                        loss = (lm + cfg.aux_loss_weight * aux + cfg.z_loss_weight * z) / cfg.grad_accum
                    loss.backward()
                acc_lm += float(lm.detach().item())
                acc_aux += float(aux.detach().item())
                self.tokens_seen += ids.numel()
                if self.tokens_seen >= total_tokens:
                    break

            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), cfg.max_grad_norm)
            self.opt.step()
            self.step += 1
            lm_v, aux_v = acc_lm / cfg.grad_accum, acc_aux / cfg.grad_accum
            self.loss_ema = lm_v if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * lm_v

            if self.rank == 0 and (self.step % cfg.log_every == 0 or self.step == 1):
                dt = time.time() - t0
                tps = self.tokens_seen / max(1e-6, dt)
                eta = (total_tokens - self.tokens_seen) / max(1, tps)
                mfu = f" MFU={100*tps*6*active_params/peak:.1f}%" if peak else ""
                print(f"[Fast] step {self.step}/{est_steps} tok={self.tokens_seen/1e9:.3f}B "
                      f"T={T} loss={lm_v:.4f} aux={aux_v:.4f} lr={lr:.2e} "
                      f"tps={tps:.0f}{mfu} ETA={eta/3600:.1f}h active={active if cfg.pgsu_n_active else 'all'}")
            if self.rank == 0 and self.step % cfg.ckpt_every_steps == 0:
                self.save(os.path.join(self.ckpt_dir, f"step{self.step}.pt"))

        if self.rank == 0:
            self.save(os.path.join(self.ckpt_dir, "final.pt"))
        dt = time.time() - t0
        return {"steps": self.step, "tokens": self.tokens_seen, "time_s": dt,
                "tok_s": self.tokens_seen / max(1e-6, dt), "loss_ema": self.loss_ema}

    def save(self, path: str):
        payload = {"kind": "cognet-fast-v1",
                   "model_state_dict": self.raw_model.state_dict(),
                   "step": self.step, "tokens_seen": self.tokens_seen,
                   "loss_ema": self.loss_ema}
        if self.cfg.save_optimizer:
            payload["optimizer_state_dict"] = self.opt.state_dict()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(payload, path)
        print(f"[Fast] sauvé : {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ckpt["model_state_dict"])
        self.step = ckpt.get("step", 0)
        self.tokens_seen = ckpt.get("tokens_seen", 0)
        self.loss_ema = ckpt.get("loss_ema")
        if "optimizer_state_dict" in ckpt:
            try:
                self.opt.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[Fast] optimizer non restauré ({e}) — fresh")
        print(f"[Fast] repris : {path} (step={self.step}, tok={self.tokens_seen})")


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark débit (tok/s)
# ═══════════════════════════════════════════════════════════════════════

def benchmark(batch_size: int = 8, seq_len: int = 512, steps: int = 20,
              hidden_dim: int = 1024, num_blocks: int = 4, vocab_size: int = 16384,
              use_fast: bool = True, compile_mode: Optional[str] = None,
              device: Optional[str] = None) -> Dict:
    """Mesure tok/s forward+backward (modèle synthétique configurable)."""
    enable_fast_mode()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CogNetMoE1B(vocab_size=vocab_size, hidden_dim=hidden_dim, num_blocks=num_blocks,
                        num_channels=8, channel_dim=384, ff_dim=hidden_dim * 4,
                        max_seq_len=seq_len, working_slots=32, episodic_slots=64,
                        semantic_slots=128, key_dim=128, n_experts=8, top_k=2,
                        use_gradient_checkpointing=False).to(dev)
    if use_fast:
        model = convert_to_fast(model)
    if compile_mode and dev.startswith("cuda"):
        model = torch.compile(model, mode=compile_mode)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    use_amp = dev.startswith("cuda")
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
    # Warmup (compile...).
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _nullcontext():
            out = model(ids)
            loss = out["logits"].float().mean() + out["moe_aux_loss"]
        loss.backward()
        opt.step()
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _nullcontext():
            out = model(ids)
            loss = out["logits"].float().mean() + out["moe_aux_loss"]
        loss.backward()
        opt.step()
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    toks = batch_size * seq_len * steps
    tps = toks / dt
    active = model.count_parameters()["active_per_token"] if hasattr(model, "count_parameters") \
        else sum(p.numel() for p in model.parameters())
    peak = _peak_flops_current()
    print(f"[Bench] tok/s={tps:.0f} ({toks/1e6:.1f}M tokens en {dt:.1f}s) "
          f"fast={use_fast} compile={compile_mode} B={batch_size} T={seq_len}")
    if peak:
        print(f"[Bench] MFU≈{100*tps*6*active/peak:.1f}% (6N/token, N_act={active/1e9:.2f}B)")
    return {"tok_s": tps, "tokens": toks, "time_s": dt}


# ═══════════════════════════════════════════════════════════════════════
#  Self-test (tiny, CPU)
# ═══════════════════════════════════════════════════════════════════════

def self_test():
    print("=" * 70)
    print("FastTrain — Self-test (tiny, CPU)")
    print("=" * 70)
    torch.manual_seed(0)
    import tempfile

    # 1. Équivalence legacy ↔ fast.
    print("\n[1/5] Équivalence numérique legacy ↔ Fast...")
    legacy = CognitiveExpertRouter(hidden_dim=64, num_channels=4, ff_dim=128, top_k=2)
    legacy.noise_std = 0.0
    legacy.eval()
    ff = legacy.experts[0].w_down.weight.shape[1]  # w_down: (hidden, ff)
    fast = FastCognitiveExpertRouter(hidden_dim=64, num_channels=4, ff_dim=ff,
                                     top_k=2, noise_std=0.0)
    fast.load_state_dict(legacy.state_dict())
    fast.eval()
    x = torch.randn(2, 16, 64)
    yl, sl = legacy(x)
    yf, sf = fast(x)
    d = (yl - yf).abs().max().item()
    assert d < 1e-4, f"divergence: {d}"
    assert abs(sl["moe_aux_loss"].item() - sf["moe_aux_loss"].item()) < 1e-4
    print(f"  ✓ écart max : {d:.2e}, aux legacy={sl['moe_aux_loss'].item():.4f} "
          f"fast={sf['moe_aux_loss'].item():.4f}")

    # 2. convert_to_fast + backward.
    print("\n[2/5] convert_to_fast + backward...")
    model = CogNetMoE1B(vocab_size=512, hidden_dim=64, num_blocks=2, num_channels=4,
                        channel_dim=32, ff_dim=128, max_seq_len=64, working_slots=4,
                        episodic_slots=8, semantic_slots=16, key_dim=32,
                        n_experts=4, top_k=2, use_gradient_checkpointing=False)
    model = convert_to_fast(model)
    ids = torch.randint(0, 512, (2, 32))
    out = model(ids, return_stats=True)
    (out["logits"].sum() + out["moe_aux_loss"]).backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    assert n_grad > 0
    print(f"  ✓ backward OK ({n_grad} params avec grad)")

    # 3. build_bin + MMapTokens + MixedPhaseLoader.
    print("\n[3/5] Dataset binaire memmap...")
    tmp = tempfile.mkdtemp()
    txt = os.path.join(tmp, "corpus.txt")
    with open(txt, "w") as f:
        f.write(("Bonjour le monde, ceci est un test CogNet. " * 200 + "\n") * 20)
    r = build_bin_dataset(txt, os.path.join(tmp, "train"), tokenizer=None)
    assert r["n_tokens"] > 1000
    ds = MMapTokens(r["bin_path"])
    rng = np.random.default_rng(0)
    b = ds.sample_batch(4, 64, rng)
    assert b.shape == (4, 64) and b.min() >= 0
    loader = MixedPhaseLoader([r["bin_path"], r["bin_path"]], current_idx=1,
                              replay_ratio=0.25, seed=0)
    ids2, pids = loader.next_batch(8, 64, device="cpu")
    assert ids2.shape == (8, 64) and pids.shape == (8,)
    assert set(pids.tolist()) <= {0, 1}
    print(f"  ✓ {r['n_tokens']:,} tokens, batch={tuple(ids2.shape)}, phases={sorted(set(pids.tolist()))}")

    # 4. FastTrainer tiny (2 steps + save/resume).
    print("\n[4/5] FastTrainer (tiny)...")
    model2 = CogNetMoE1B(vocab_size=512, hidden_dim=32, num_blocks=2, num_channels=4,
                         channel_dim=16, ff_dim=64, max_seq_len=32, working_slots=2,
                         episodic_slots=4, semantic_slots=8, key_dim=16,
                         n_experts=4, top_k=2, use_gradient_checkpointing=False)
    model2 = convert_to_fast(model2)
    cfg = FastTrainerConfig(batch_size=4, grad_accum=1, lr=3e-4, use_bf16=False,
                            compile_mode=None, log_every=1, ckpt_every_steps=100,
                            seq_stages=[(0, 16), (100, 32)])
    tr = FastTrainer(model2, cfg, device="cpu", ckpt_dir=os.path.join(tmp, "ckpts"))
    rng2 = np.random.default_rng(1)
    ds2 = MMapTokens(r["bin_path"])
    stats = tr.train(lambda B, T: ds2.sample_batch(B, T, rng2), total_tokens=512)
    assert stats["tokens"] >= 512 and stats["loss_ema"] is not None
    assert os.path.exists(os.path.join(tmp, "ckpts", "final.pt"))
    # Resume.
    tr2 = FastTrainer(model2, cfg, device="cpu", ckpt_dir=os.path.join(tmp, "ckpts"))
    tr2.load(os.path.join(tmp, "ckpts", "final.pt"))
    assert tr2.tokens_seen == tr.tokens_seen
    print(f"  ✓ {stats['tokens']} tokens, loss_ema={stats['loss_ema']:.4f}, resume OK")

    # 5. Curriculum seq_len.
    print("\n[5/5] Curriculum...")
    assert tr.seq_len_at(0) == 16 and tr.seq_len_at(10_000) == 32
    print("  ✓ T(0)=16 → T(100+)=32")

    print("\n" + "=" * 70)
    print("✓ Self-test FastTrain passé.")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="CogNet-MoE training rapide (milliards de tokens)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--build-bin", action="store_true")
    ap.add_argument("--txt", type=str, help="corpus .txt source")
    ap.add_argument("--tokenizer", type=str, default=None, help="cognet_tokenizer.json (None=byte fallback)")
    ap.add_argument("--out", type=str, help="préfixe de sortie (sans extension)")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--bin", type=str, help="dataset .bin pré-tokenisé")
    ap.add_argument("--tokens", type=float, default=1_000_000_000, help="budget tokens (défaut 1B)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pgsu", type=int, default=0, help="PGSU n_active (0=désactivé)")
    ap.add_argument("--compile", type=str, default=None, help="None|reduce-overhead|max-autotune")
    ap.add_argument("--optimizer", type=str, default="adamw-fused")
    ap.add_argument("--ckpt-dir", type=str, default="./fast_ckpts")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--tiny", action="store_true", help="modèle tiny (smoke test GPU/CPU)")
    ap.add_argument("--vocab-size", type=int, default=16384)
    args = ap.parse_args()

    if args.self_test or (not args.benchmark and not args.build_bin and not args.train):
        self_test()
        return
    if args.benchmark:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            benchmark(batch_size=2, seq_len=64, steps=5, hidden_dim=128,
                      num_blocks=2, vocab_size=1024, device="cpu")
        else:
            benchmark(compile_mode=args.compile)
        return
    if args.build_bin:
        assert args.txt and args.out, "--txt et --out requis"
        build_bin_dataset(args.txt, args.out, tokenizer=args.tokenizer)
        return
    if args.train:
        assert args.bin, "--bin requis"
        enable_fast_mode()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ds = MMapTokens(args.bin)
        vocab = ds.vocab_size
        print(f"[Fast] dataset {args.bin} : {ds.n_tokens:,} tokens, vocab={vocab}")
        if args.tiny:
            model = CogNetMoE1B(vocab_size=vocab, hidden_dim=256, num_blocks=4,
                                num_channels=8, channel_dim=64, ff_dim=1024,
                                max_seq_len=args.seq_len, working_slots=16,
                                episodic_slots=32, semantic_slots=64, key_dim=64,
                                n_experts=8, top_k=2, use_gradient_checkpointing=True)
        else:
            model = create_cognet_moe_1b(vocab_size=vocab, max_seq_len=args.seq_len)
        model = convert_to_fast(model)
        p = model.count_parameters()
        print(f"[Fast] params total={p['total']:,} actif/token={p['active_per_token']:,}")
        cfg = FastTrainerConfig(batch_size=args.batch_size, grad_accum=args.grad_accum,
                                lr=args.lr, compile_mode=args.compile,
                                optimizer=args.optimizer, pgsu_n_active=args.pgsu,
                                seq_stages=[(0, args.seq_len)])
        tr = FastTrainer(model, cfg, device=device, ckpt_dir=args.ckpt_dir)
        if args.resume:
            tr.load(args.resume)
        rng = np.random.default_rng(cfg.seed + tr.rank * 10_000)
        tr.train(lambda B, T: ds.sample_batch(B, T, rng), total_tokens=int(args.tokens))


if __name__ == "__main__":
    main()
