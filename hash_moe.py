#!/usr/bin/env python3
"""
hash_moe.py — Routing par HASHAGE (sans gradient) pour CogNet-MoE
==================================================================

Idée : remplacer le router appris (cohérence + gradients + aux-loss + collapse)
par une fonction de hashage DÉTERMINISTE qui assigne chaque token à ses experts.

Pourquoi c'est intéressant (et prouvé : « Hash Layers », Roller et al., 2021) :
  1. ZÉRO paramètre de routing, ZÉRO gradient de routing, ZÉRO aux/z-loss,
     ZÉRO routing collapse (la balance est structurelle, pas apprise).
  2. L'assignation est connue AVANT tout calcul → on peut PRÉ-DÉCOUPER les
     données par hash et entraîner chaque expert sur un CPU/worker séparé
     avec ZÉRO communication (impossible avec un router appris).
  3. Reste CogNet-native : pas d'attention, pas de gate Linear(D→N), O(n).

Deux modes (sans gradient dans les deux cas) :
  - 'token' : expert = hash(token_id, salt) mod C — balance parfaite, mais
    nécessite les token_ids (le trainer les fournit via set_batch_token_ids).
  - 'lsh'   : LSH sur le hidden state (random projection fixe + signes → code,
    code → experts). Content-dépendant (états similaires → mêmes experts),
    aucun plumbing (pur fonction de x). Mode par défaut.

Les experts EUX restent des FusedSwiGLU entraînés normalement (le hash ne
supprime pas les matmuls des experts — il supprime le router et permet le
sharding sans communication). La variante 100% sans gradient (random features
+ moindres carrés) est testée dans hash_experiment.py (Exp B).

Usage :
    python3 hash_moe.py --self-test     # équivalence d'interface + balance (CPU)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
sys.path.insert(0, str(HERE))

from cognet_1b_optimized import RMSNorm, FusedSwiGLU  # noqa: E402
from cognet_moe import CogNetMoE1B  # noqa: E402


# ───────────────────────────────────────────────────────────────────
# Hashage : token-hash (parfait) et LSH (content-dépendant)
# ───────────────────────────────────────────────────────────────────

def token_hash_experts(token_ids: torch.Tensor, n_experts: int, top_k: int,
                       salt: int = 0x9E3779B9) -> torch.Tensor:
    """
    Assigne k experts DISTINCTS par token via hash entier (splitmix64-esque).
    Balance parfaite en espérance. (B, T) -> (B, T, K).
    """
    C, K = n_experts, top_k
    x = token_ids.long() + salt
    # Avalanche entière (constantes 32-bit, arithmétique int64 wraparound).
    x = x ^ (x >> 16)
    x = x * torch.tensor(0x7FEB352D, dtype=torch.int64)
    x = x ^ (x >> 15)
    x = x * torch.tensor(0x846CA68B, dtype=torch.int64)
    x = x ^ (x >> 16)
    out = torch.empty(token_ids.shape + (K,), dtype=torch.long)
    for j in range(K):
        e = ((x >> (j * 11)) % C).to(torch.long)
        if j > 0:
            # dédup : si collision avec un expert déjà pris, décale (déterministe).
            for p in range(j):
                dup = e == out[..., p]
                e = torch.where(dup, (e + 1 + p) % C, e)
        out[..., j] = e
    return out


class LSHHasher(nn.Module):
    """
    LSH : code binaire du hidden state via projection aléatoire FIXE (buffer gelé).
    États similaires → codes proches → souvent mêmes experts (localité).
    """

    def __init__(self, hidden_dim: int, n_bits: int = 24, seed: int = 1234):
        super().__init__()
        assert n_bits <= 60
        g = torch.Generator().manual_seed(seed)
        R = torch.randn(hidden_dim, n_bits, generator=g)
        R = R / R.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.register_buffer("R", R)
        self.n_bits = n_bits

    def codes(self, x: torch.Tensor) -> torch.Tensor:
        bits = (x.float() @ self.R.to(x.device).float()) > 0  # (..., B)
        packed = torch.zeros(x.shape[:-1], dtype=torch.int64, device=x.device)
        for i in range(self.n_bits):
            packed |= bits[..., i].long() << i
        return packed

    def experts(self, x: torch.Tensor, n_experts: int, top_k: int) -> torch.Tensor:
        code = self.codes(x)  # (B, T)
        C, K = n_experts, top_k
        out = torch.empty(code.shape + (K,), dtype=torch.long, device=x.device)
        for j in range(top_k):
            e = ((code >> (j * 7)) % C).to(torch.long)
            if j > 0:
                for p in range(j):
                    dup = e == out[..., p]
                    e = torch.where(dup, (e + 1 + p) % C, e)
            out[..., j] = e
        return out


# ───────────────────────────────────────────────────────────────────
# HashExpertRouter — drop-in CogNet-native sans router appris
# ───────────────────────────────────────────────────────────────────

class HashExpertRouter(nn.Module):
    """
    Remplacement du CognitiveExpertRouter : to_channels + experts + norm +
    résiduel identiques, mais routing = hash fixe (pas de coherence_router,
    pas de gate, pas d'aux/z-loss à optimiser).

    Poids-compatibles avec le legacy pour to_channels/experts/norm
    (load_state_dict partiel). stats compatibles (aux/z = 0).
    """

    def __init__(self, hidden_dim: int, ff_dim: int, n_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.0,
                 mode: str = "lsh", n_bits: int = 24, seed: int = 1234):
        super().__init__()
        assert mode in ("lsh", "token")
        assert top_k <= n_experts
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.n_experts = n_experts
        self.num_channels = n_experts  # alias compat
        self.top_k = top_k
        self.mode = mode
        self.aux_loss_weight = 0.0  # pas d'aux-loss : balance structurelle
        self.z_loss_weight = 0.0

        self.to_channels = nn.Linear(hidden_dim, n_experts * hidden_dim, bias=False)
        self.experts = nn.ModuleList([FusedSwiGLU(hidden_dim, ff_dim, dropout)
                                      for _ in range(n_experts)])
        self.norm = RMSNorm(hidden_dim)
        self.lsh = LSHHasher(hidden_dim, n_bits, seed) if mode == "lsh" else None
        self._batch_token_ids: Optional[torch.Tensor] = None
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.to_channels.weight, mean=0.0, std=0.02)

    def get_expert(self, i: int) -> FusedSwiGLU:
        return self.experts[i]

    def set_batch_token_ids(self, ids: Optional[torch.Tensor]):
        self._batch_token_ids = ids

    def forward(self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape
        C, K = self.n_experts, self.top_k
        N = B * T

        # 1. Assignation par hash (PAS DE GRADIENT — pur routage).
        with torch.no_grad():
            if self.mode == "token":
                ids = self._batch_token_ids if token_ids is None else token_ids
                assert ids is not None, "mode token : fournir token_ids"
                assign = token_hash_experts(ids.to(x.device), C, K)  # (B,T,K)
            else:
                assign = self.lsh.experts(x.detach(), C, K)          # (B,T,K)
            w = torch.full(assign.shape, 1.0 / K, device=x.device, dtype=x.dtype)

        # 2. Projection + dispatch (identique au fast router).
        chan = self.to_channels(x).view(B, T, C, D)
        flat_idx, flat_w = assign.reshape(N, K), w.reshape(N, K)
        expert_w = torch.zeros(N, C, device=x.device, dtype=x.dtype)
        expert_w.scatter_add_(1, flat_idx, flat_w)
        f = (expert_w > 0).float().mean(0)
        chan_flat = chan.reshape(N, C, D)
        combined = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        for i in range(C):
            w_i = expert_w[:, i]
            if not bool(torch.any(w_i > 0).item()):
                continue
            tok = w_i > 0
            combined[tok] += w_i[tok].unsqueeze(-1).to(chan.dtype) * self.experts[i](chan_flat[tok, i])

        out = self.norm(combined.view(B, T, D))
        out = x + out

        with torch.no_grad():
            Pp = f.clamp_min(1e-8) / f.sum().clamp_min(1e-8)
            ent = -(Pp * Pp.log()).sum()
        stats = {
            "moe_aux_loss": torch.tensor(0.0, device=x.device, dtype=x.dtype),
            "moe_z_loss": torch.tensor(0.0, device=x.device, dtype=x.dtype),
            "moe_max_load": f.max().detach(),
            "moe_min_load": f.min().detach(),
            "moe_routing_entropy": ent.detach(),
            "moe_expert_usage": f.detach(),
            "routing_entropy": ent.detach(),
        }
        return out, stats


def convert_to_hash(model: CogNetMoE1B, mode: str = "lsh", n_bits: int = 24,
                    seed: int = 1234) -> CogNetMoE1B:
    """Swap chaque router appris → hash (to_channels/experts/norm copiés)."""
    for blk in model.blocks:
        legacy = blk.cognitive_expert_router
        if isinstance(legacy, HashExpertRouter):
            continue
        C = len(legacy.experts)
        ff = legacy.experts[0].w_down.weight.shape[1]
        h = HashExpertRouter(legacy.hidden_dim, ff, C, legacy.top_k,
                             legacy.experts[0].dropout.p, mode, n_bits, seed + 1)
        with torch.no_grad():
            h.to_channels.weight.copy_(legacy.to_channels.weight)
            for i in range(C):
                h.experts[i].load_state_dict(legacy.experts[i].state_dict())
            h.norm.load_state_dict(legacy.norm.state_dict())
        blk.cognitive_expert_router = h
    return model


def shard_by_hash(token_ids: torch.Tensor, n_experts: int, top_k: int = 1) -> Dict[int, torch.Tensor]:
    """
    PRÉ-DÉCOUPAGE sans modèle : positions de chaque expert (top-1, salt 0).
    Chaque worker CPU entraîne ses experts sur son shard, ZÉRO communication.
    """
    a = token_hash_experts(token_ids, n_experts, top_k)
    return {e: (a[..., 0] == e).nonzero(as_tuple=False) for e in range(n_experts)}


# ───────────────────────────────────────────────────────────────────
# Self-test
# ───────────────────────────────────────────────────────────────────
def self_test():
    print("=" * 70)
    print("HashMoE — Self-test (CPU)")
    print("=" * 70)
    torch.manual_seed(0)
    # 1. Balance token-hash.
    ids = torch.randint(0, 512, (8, 64))
    a = token_hash_experts(ids, 8, 2)
    assert a.shape == (8, 64, 2) and (a[..., 0] != a[..., 1]).all()
    cnt = torch.bincount(a.reshape(-1), minlength=8).float() / a.numel()
    print(f"  [1] token-hash : usage={cnt.tolist()} (attendu ~0.125 chacun)")
    assert cnt.min() > 0.09 and cnt.max() < 0.16, "hash biaisé!"
    # 2. LSH : balance + localité (même input → même assignation).
    lsh = LSHHasher(64, 24, 1)
    x = torch.randn(4, 32, 64)
    e1, e2 = lsh.experts(x, 8, 2), lsh.experts(x, 8, 2)
    assert torch.equal(e1, e2), "LSH non déterministe!"
    cnt = torch.bincount(e1.reshape(-1), minlength=8).float() / e1.numel()
    print(f"  [2] LSH déterministe ✓, usage={cnt.tolist()}")
    xn = x + torch.randn_like(x) * 0.01
    same = (lsh.experts(xn, 8, 2)[..., 0] == e1[..., 0]).float().mean().item()
    print(f"  [2b] localité : {same:.1%} mêmes experts après micro-bruit (attendu élevé)")
    assert same > 0.8
    # 3. Router : forward/backward, pas de grad routing (pas de params routing).
    r = HashExpertRouter(64, 128, 4, 2, mode="lsh")
    n_params = sum(p.numel() for p in r.parameters())
    rparams = sum(p.numel() for n, p in r.named_parameters() if "lsh" in n)
    assert rparams == 0, "le LSH ne doit avoir AUCUN paramètre entraînable"
    y, s = r(x)
    assert y.shape == x.shape
    (y.sum()).backward()
    n_grad = sum(1 for p in r.parameters() if p.grad is not None)
    print(f"  [3] forward/backward OK, {n_params:,} params (0 routing), {n_grad} avec grad")
    # 4. convert + sharding.
    m = CogNetMoE1B(vocab_size=512, hidden_dim=64, num_blocks=2, num_channels=4,
                    channel_dim=16, ff_dim=128, max_seq_len=64, working_slots=4,
                    episodic_slots=8, semantic_slots=16, key_dim=16,
                    n_experts=4, top_k=2, use_gradient_checkpointing=False)
    m = convert_to_hash(m, mode="lsh")
    out = m(torch.randint(0, 512, (2, 32)))
    assert out["logits"].shape == (2, 32, 512)
    assert float(out["moe_aux_loss"].item()) == 0.0
    shards = shard_by_hash(torch.arange(0, 10000).reshape(100, 100), 4)
    sizes = [len(v) for v in shards.values()]
    print(f"  [4] convert OK (aux=0 ✓), pré-sharding 10k tokens → {sizes} (attendu ~2500 chacun)")
    assert min(sizes) > 2000 and max(sizes) < 3000
    print("\n✓ Self-test HashMoE passé.")
    print("=" * 70)


if __name__ == "__main__":
    self_test()
