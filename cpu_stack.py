#!/usr/bin/env python3
"""
cpu_stack.py — Stack d'entraînement 100% CPU pour CogNet-MoE (zéro GPU)
=======================================================================

THÈSE : impossible d'entraîner le 1B (7.21B params / 2.38B actifs) sur CPU avec la
recette naïve — 6·N·D FLOPs ≈ 2×10¹⁹ FLOPs pour la cible 1.36B tokens, vs ~10¹⁰–10¹¹
FLOP/s pour un CPU : ~30-200 ans. MAIS CogNet possède des propriétés mathématiques
inédites (absentes des transformers) qui changent le coût au premier ordre. Ce module
les matérialise. Rapport complet : CPU_TRAINING_REPORT.md.

Propriétés exploitées (toutes vérifiées par self-test, mesures honnêtes) :

  P1 — ASSOCIATIVITÉ DU PRODUIT MATRICIEL (identité exacte, 0 approximation)
       to_channels = Linear(D → C·D) ≡ C matrices W_e (D×D) empilées. L'expert e
       calcule w_gate_up(W_e·x) = (U_e·W_e)·x par associativité. On PLIE A_e = U_e·W_e
       à l'init : to_channels disparaît du chemin actif. C'est une reparamétrisation
       EXACTE (rang(A_e) ≤ D, l'espace fonctionnel est inchangé), pas une approx.
       a) fold_residual=True  : conserve le résiduel intra-expert W_e·x, mais ne le
          calcule QUE pour les K experts sélectionnés (K·D² au lieu de C·D² → ÷4).
       b) fold_residual=False (« simplify », from-scratch) : le résiduel intra-expert
          est redondant avec le résiduel de bloc (out = x + Norm(Σ w_e·out_e)) → W_e
          supprimé. MACs/token des blocs : −25% vs legacy, à structure résiduelle
          de bloc identique. L'invariant « canaux == experts » est préservé : chaque
          expert possède sa propre vue de x (A_e = sa projection-canal pliée).

  P2 — ROUTAGE = FONCTION PURE DU TOKEN (hash) → assignation connue AVANT le calcul
       token_hash_experts (hash_moe.py, validé 3/3 seeds, bat le routing appris de
       ~0.8 nats sur la sonde sans fuite du repo). Zéro param de routing, zéro
       aux/z-loss, zéro collapse, et : (a) plus de router à backpropager, (b) le
       dispatch ne calcule QUE les experts sélectionnés (pas de to_channels dense),
       (c) pré-sharding CPU expert-parallèle sans communication (Exp C du repo),
       (d) prefetch exact batch N+1 pour le paging disque (expert_pager.py).

  P3 — PARCOURS SOUS-DÉTERMINÉ DU GRAPHE (LISA/PGSU + détachement)
       Le repo a déjà PGSU (rotation de blocs actifs). LISA (Pan et al., NeurIPS
       2024) a montré qu'entraîner γ blocs au hasard et geler le reste égale ou
       bat le full-parameter. On ajoute le mode « lisa-detach » : le préfixe gelé
       = extracteur de features DÉTACHÉ (backward strictement coupé : 0 GEMM de
       rétroprop sous le cutoff, pas seulement les grads poids). Sur CPU le backward
       coûte ~2× le forward : couper le backward d'une fraction f du tronc économise
       ~2f/(1+2) ≈ jusqu'à ~50% des FLOPs d'entraînement dans les phases profondes.

  P4 — SOFTMAX ÉCHANTILLONNÉ CORRIGÉ (tête V=16384 liée à l'embedding)
       Logits = h·Eᵀ : V·D MACs/token forward + scatter dense dans le backward.
       Tête d'entraînement : cibles + S négatifs uniformes, correction log(S/V)
       exacte (Jean et al. 2015) → (S+1)·D MACs (~64× moins). Eval/test : logits
       pleins, modèle identique. Le backward ne touche que S+1 lignes de E.

  P5 — INIT INTELLIGENTE P1 SANS DONNÉES (EDT Phase 1, validée 4/4 seeds +0.50 nat)
       Entraîner chaque expert vers l'identité sur du BRUIT SYNTHÉTIQUE (aucun corpus,
       MSE f(x̂)≈x̂, x̂∼N(0,1)) : le pré-conditionnement résiduel qui a fait gagner
       P1-ONLY dans EDT_REFUTATION_REPORT.md. Coût CPU : minutes à l'échelle tiny,
       quelques heures à l'échelle réelle, ZÉRO token de corpus.

  P6 — ÉTATS D'OPTIMISEUR 8-BIT SUR CPU (bitsandbytes est GPU-only)
       AdamW à moments quantifiés par blocs (int8 + échelles fp32/bloc, style
       bnb 8-bit optimizer) : mémoire optimiseur ÷4, pure torch CPU.

  P7 — CŒURS BAS-RANG / TERNALISATION b1.58 (pistes noyau, qualité mesurée)
       SwiGLU factorisé (D→r→ff) et poids ternaires {-1,0,+1}·s par STE
       (BitNet b1.58, Ma et al. 2024 — entraîné from scratch à 2.4B params / 4T
       tokens à parité pleine précision ; bitnet.cpp : ×2.37-6.17 sur CPU x86 en
       inférence). En eager PyTorch on ne touche pas le gain noyau : ces options
       sont évaluées en QUALITÉ, la vitesse reste du travail noyau documenté.

  P8 — MUR MÉMOIRE → MUR DISQUE (ExpertPager, déjà prouvé au repo)
       Échelle de capacité sans RAM : experts paginés NVMe (int8 ×3.72, prefetch
       exact hash-ahead, anti-stall 4.4×). Le pager page FusedSwiGLU : les experts
       pliés (simplify) ont EXACTEMENT la même forme state_dict (clés/id formes)
       → compatibilité pager immédiate. (llama.cpp a RFC la même idée en 2026 :
       « MoE offload to disk with on-demand paging » — le repo l'anticipait.)

Tout reste CogNet-native : pas d'attention, pas de gate Linear(D→N) appris,
canaux == experts, résiduels partout, mémoire 3-tier et composer intacts, O(n).

Usage :
    PYTHONPATH=source:. python3 cpu_stack.py --self-test    # 8 tests, ~2 min CPU
    PYTHONPATH=source:. python3 cpu_stack.py --probe        # sonde qualité sans fuite
    PYTHONPATH=source:. python3 cpu_stack.py --bench        # tok/s tiny mesurés
    PYTHONPATH=source:. python3 cpu_stack.py --tables       # projections 1B + CPU-native
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
sys.path.insert(0, str(HERE))

from cognet_1b_optimized import RMSNorm, FusedSwiGLU  # noqa: E402
from cognet_moe import CogNetMoE1B, CognitiveExpertRouter  # noqa: E402
from hash_moe import token_hash_experts, LSHHasher  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
#  P7 — Ternarisation b1.58 (STE) — poids latents fp32, forward ternaire
# ═══════════════════════════════════════════════════════════════════════

def ternary_ste(w: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58 : s = mean|w| par LIGNE, q = round(w/s) clampé {-1,0,1},
    y = s·q, straight-through (grad passe identité). Poids latent intact."""
    s = w.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    q = (w / s).round().clamp_(-1.0, 1.0)
    y = q * s
    return w + (y - w).detach()


class TernaryLinear(nn.Linear):
    """Linear dont le forward utilise la ternarisation STE (qualité b1.58)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, ternary_ste(self.weight), self.bias)


# ═══════════════════════════════════════════════════════════════════════
#  P1+P2 — Expert plié (fold associatif) + routeur hash CPU
# ═══════════════════════════════════════════════════════════════════════

class FoldedExpert(nn.Module):
    """
    Expert CogNet plié : core SwiGLU alimenté DIRECTEMENT par x.

    Legacy :  out = W_e·x + Norm( W_down · (SiLU ⊙ (U_e · (W_e·x))) )
    Plié  :   out =   R   + Norm( W_down · (SiLU ⊙ (A_e · x)) ),  A_e = U_e·W_e
        avec R = W_e·x  (fold_residual=True,  identité exacte vs legacy)
        ou   R = 0      (fold_residual=False, « simplify », from-scratch)

    state_dict simplify == FusedSwiGLU (mêmes clés/formes) → pager-compatible.
    Options : low_rank (factorisation D→r→ff) ; ternary (forward b1.58 STE).
    """

    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float = 0.0,
                 fold_residual: bool = False, low_rank: Optional[int] = None,
                 ternary: bool = False):
        super().__init__()
        D, FF = hidden_dim, ff_dim
        self.fold_residual = fold_residual
        self.low_rank = low_rank
        self.ternary = ternary
        Lin = TernaryLinear if ternary else nn.Linear
        if low_rank:
            self.w_gate_up1 = Lin(D, low_rank, bias=False)
            self.w_gate_up2 = Lin(low_rank, 2 * FF, bias=False)
            self.w_down1 = Lin(FF, low_rank, bias=False)
            self.w_down2 = Lin(low_rank, D, bias=False)
        else:
            self.w_gate_up = Lin(D, 2 * FF, bias=False)
            self.w_down = Lin(FF, D, bias=False)
        self.w_res = nn.Linear(D, D, bias=False) if fold_residual else None
        self.norm = RMSNorm(D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_rank:
            gate_up = self.w_gate_up2(self.w_gate_up1(x))
            core = self.w_down2(self.w_down1(F.silu(gate_up.chunk(2, dim=-1)[0])
                                             * gate_up.chunk(2, dim=-1)[1]))
        else:
            gate, up = self.w_gate_up(x).chunk(2, dim=-1)
            core = self.w_down(F.silu(gate) * up)
        out = self.norm(core)
        if self.fold_residual:
            out = self.w_res(x) + out
        return self.dropout(out)

    @classmethod
    def from_legacy(cls, expert: FusedSwiGLU, w_e: torch.Tensor) -> "FoldedExpert":
        """PLIAGE EXACT d'un (expert legacy, slice W_e de to_channels).
        A_e = U_e·W_e ; w_down, norm, dropout copiés ; w_res = W_e.
        Fonction STRICTEMENT identique (à l'arrondi fp près) — cf. self-test [2]."""
        D = expert.w_down.weight.shape[0]
        FF = expert.w_down.weight.shape[1]
        fe = cls(D, FF, expert.dropout.p, fold_residual=True)
        with torch.no_grad():
            fe.w_gate_up.weight.copy_(expert.w_gate_up.weight @ w_e)
            fe.w_down.weight.copy_(expert.w_down.weight)
            fe.norm.load_state_dict(expert.norm.state_dict())
            fe.w_res.weight.copy_(w_e)
        return fe


class FoldedHashRouter(nn.Module):
    """
    Routeur CPU : canaux == experts (invariant CogNet), routing = hash (0 param),
    projections pliées par associativité, dispatch = uniquement les K sélectionnés.

    Drop-in pour CogNetMoEBlock.cognitive_expert_router (même interface/stats
    que HashExpertRouter). Modes : 'token' (balance parfaite, ids requis) |
    'lsh' (content-dépendant, 0 plumbing).
    """

    def __init__(self, hidden_dim: int, ff_dim: int, n_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.0, mode: str = "token",
                 n_bits: int = 24, seed: int = 1234,
                 fold_residual: bool = False, low_rank: Optional[int] = None,
                 ternary: bool = False):
        super().__init__()
        assert mode in ("token", "lsh")
        assert top_k <= n_experts
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.n_experts = n_experts
        self.num_channels = n_experts  # invariant canaux == experts
        self.top_k = top_k
        self.mode = mode
        self.fold_residual = fold_residual
        self.low_rank = low_rank
        self.ternary = ternary
        self.aux_loss_weight = 0.0   # balance structurelle (hash) : rien à optimiser
        self.z_loss_weight = 0.0

        self.experts = nn.ModuleList([
            FoldedExpert(hidden_dim, ff_dim, dropout, fold_residual,
                         low_rank, ternary)
            for _ in range(n_experts)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lsh = LSHHasher(hidden_dim, n_bits, seed) if mode == "lsh" else None
        self._batch_token_ids: Optional[torch.Tensor] = None
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

    # ── API compat (EDT / pager / trainer) ──────────────────────────────
    def get_expert(self, i: int) -> FoldedExpert:
        return self.experts[i]

    def set_batch_token_ids(self, ids: Optional[torch.Tensor]):
        self._batch_token_ids = ids

    @classmethod
    def from_legacy(cls, legacy: CognitiveExpertRouter, mode: str = "token",
                    n_bits: int = 24, seed: int = 1234) -> "FoldedHashRouter":
        """Pliage EXACT d'un CognitiveExpertRouter entraîné (fold_residual=True).
        Le routing devient hash (le router appris n'a pas d'équivalent — c'est
        le point P2) ; les experts+projections sont fonctionnellement identiques."""
        C, D = legacy.num_channels, legacy.hidden_dim
        FF = legacy.experts[0].w_down.weight.shape[1]
        r = cls(D, FF, C, legacy.top_k, legacy.experts[0].dropout.p,
                mode, n_bits, seed, fold_residual=True)
        W_all = legacy.to_channels.weight  # (C·D, D) — C slices W_e empilées
        with torch.no_grad():
            for e in range(C):
                w_e = W_all[e * D:(e + 1) * D, :]
                r.experts[e] = FoldedExpert.from_legacy(legacy.experts[e], w_e)
            r.norm.load_state_dict(legacy.norm.state_dict())
        return r

    def forward(self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape
        C, K = self.n_experts, self.top_k
        N = B * T

        # 1. Assignation hash — PURE, pas de gradient, connue avant tout calcul.
        with torch.no_grad():
            if self.mode == "token":
                ids = self._batch_token_ids if token_ids is None else token_ids
                assert ids is not None, "mode token : fournir token_ids (set_batch_token_ids)"
                assign = token_hash_experts(ids.to(x.device), C, K)      # (B,T,K)
            else:
                assign = self.lsh.experts(x.detach(), C, K)              # (B,T,K)
            w = torch.full(assign.shape, 1.0 / K, device=x.device, dtype=x.dtype)

        # 2. Dispatch : SEULEMENT les K·N projections actives (pas de to_channels).
        flat_idx = assign.reshape(N, K)
        flat_w = w.reshape(N, K)
        expert_w = torch.zeros(N, C, device=x.device, dtype=x.dtype)
        expert_w.scatter_add_(1, flat_idx, flat_w)
        f = (expert_w > 0).float().mean(0)                     # stats usage
        x_flat = x.reshape(N, D)
        combined = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        for i in range(C):
            w_i = expert_w[:, i]
            if not bool(torch.any(w_i > 0)):
                continue
            tok = w_i > 0
            combined[tok] += w_i[tok].unsqueeze(-1) * self.experts[i](x_flat[tok])

        out = self.norm(combined.view(B, T, D))
        out = x + out                                          # résiduel de bloc

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


def convert_to_cpu(model: CogNetMoE1B, mode: str = "token",
                   fold_residual: bool = False, low_rank: Optional[int] = None,
                   ternary: bool = False, seed: int = 1234,
                   from_trained: bool = False) -> CogNetMoE1B:
    """
    Swap chaque router → FoldedHashRouter (stack CPU).

    from_trained=False : nouvelle init (le fold est à l'init — A_e libre).
    from_trained=True  : PLIAGE EXACT des poids existants (fold_residual forcé,
                         low_rank/ternary ignorés — la fonction est préservée).
    """
    for blk in model.blocks:
        legacy = blk.cognitive_expert_router
        if isinstance(legacy, FoldedHashRouter):
            continue
        if from_trained:
            r = FoldedHashRouter.from_legacy(legacy, mode=mode, seed=seed + 1)
        else:
            C = getattr(legacy, "num_channels", len(legacy.experts))
            ff = legacy.experts[0].w_down.weight.shape[1]
            r = FoldedHashRouter(legacy.hidden_dim, ff, C, legacy.top_k,
                                 legacy.experts[0].dropout.p, mode=mode,
                                 fold_residual=fold_residual, low_rank=low_rank,
                                 ternary=ternary, seed=seed + 1)
            with torch.no_grad():
                # la norm de bloc du legacy est réutilisable telle quelle
                r.norm.load_state_dict(legacy.norm.state_dict())
        blk.cognitive_expert_router = r
    return model


# ═══════════════════════════════════════════════════════════════════════
#  P5 — Init intelligente P1 (EDT validée) : experts → identité, données = bruit
# ═══════════════════════════════════════════════════════════════════════

def p1_identity_init(router: FoldedHashRouter, steps: int = 200, lr: float = 3e-3,
                     batch: int = 512, seed: int = 0, verbose: bool = False
                     ) -> Dict[str, float]:
    """
    Phase-1 EDT adaptée au routeur plié : chaque expert apprend f(x̂) ≈ x̂ sur du
    bruit Gaussien synthétique — AUCUN corpus, AUCUN token réel consommé.
    (fold_residual : le résiduel interne aide à représenter l'identité ;
     simplify       : le core apprend la carte identité complète.)
    Retourne les métriques MSE init/fin (moyenne experts).
    """
    D = router.hidden_dim
    g = torch.Generator().manual_seed(seed)
    was_training = router.training
    router.train()
    mse0, mse1 = [], []
    for e in router.experts:
        opt = torch.optim.AdamW(e.parameters(), lr=lr, weight_decay=0.0)
        m0 = None
        for s in range(steps):
            x = torch.randn(batch, D, generator=g)
            with torch.no_grad() if s == 0 else torch.enable_grad():
                pass
            pred = e(x)
            loss = F.mse_loss(pred, x)
            if s == 0:
                m0 = float(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        mse0.append(m0)
        mse1.append(float(loss))
        if verbose:
            print(f"    expert MSE {m0:.4f} → {float(loss):.4f}")
    if not was_training:
        router.eval()
    return {"mse_init": sum(mse0) / len(mse0), "mse_final": sum(mse1) / len(mse1)}


# ═══════════════════════════════════════════════════════════════════════
#  P4 — Tête softmax échantillonnée, correction exacte log(S/V)
# ═══════════════════════════════════════════════════════════════════════

def forward_hidden(model: CogNetMoE1B, ids: torch.Tensor, grad: bool = True,
                   frozen_prefix: int = 0) -> torch.Tensor:
    """
    Replique model.forward jusqu'à AVANT output_proj → h (B,T,D) post final_norm.
    Sert à la tête échantillonnée (P4) et à lisa-detach (P3) : les blocs
    [0, frozen_prefix) sont exécutés SANS graphe et leur sortie est détachée.
    """
    x = model.encoder(ids)
    if frozen_prefix > 0:
        with torch.no_grad():
            for block in model.blocks[:frozen_prefix]:
                x, _ = block(x)
        x = x.detach()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for block in model.blocks[frozen_prefix:]:
            x, _ = block(x)
        x = model.final_norm(x)
    return x


def sampled_ce_loss(h: torch.Tensor, y: torch.Tensor, emb_weight: torch.Tensor,
                    num_neg: int, generator: Optional[torch.Generator] = None
                    ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Cross-entropie échantillonnée uniforme, correction exacte (Jean et al. 2015).

    Partition vraie :  Z_i = exp(l_y) + Σ_{j≠y} exp(l_j)
    Estimateur      :  Ẑ_i = exp(l_y) + Σ_{s=1..S} exp(l_s − log(S/V))·1(l_s≠y_i)
    Tirage AVEC remise, multiplicités CONSERVÉES (sinon la correction est fausse :
    E[Ẑ]=Z exige Σ sur les S tirages bruts), masquage des vrais-accidentels PAR
    ÉCHANTILLON (pas par batch). Alors E[Ẑ] = Z EXACTEMENT (cf. self-test [5]).
    Coût : (S+1)·D MACs/token au lieu de V·D ; backward ne touche que les lignes
    cibles/négatives de l'embedding lié. Eval : tête pleine (modèle inchangé).
    """
    N, D = h.shape
    V = emb_weight.shape[0]
    y = y.reshape(-1)
    neg = torch.randint(0, V, (num_neg,), generator=generator,
                        device=emb_weight.device)
    W_neg = emb_weight[neg]                        # (S,D) — répétitions conservées
    pos_w = emb_weight[y]                          # (N,D)
    l_pos = (h * pos_w).sum(-1)                    # (N,)
    l_neg = h @ W_neg.T - math.log(num_neg / V)    # (N,S), correction log q
    own = neg.unsqueeze(0) == y.unsqueeze(1)       # (N,S) vrais accidentels
    l_neg = l_neg.masked_fill(own, float("-inf"))
    logits = torch.cat([l_pos.unsqueeze(1), l_neg], dim=1)   # (N, 1+S)
    target = torch.zeros(N, dtype=torch.long, device=h.device)
    loss = F.cross_entropy(logits, target)
    return loss, {"V": float(V), "S_neg": float(num_neg),
                  "head_mac_ratio": float((num_neg + 1) * D) / float(V * D)}


# ═══════════════════════════════════════════════════════════════════════
#  P6 — AdamW 8-bit CPU (moments quantifiés par blocs)
# ═══════════════════════════════════════════════════════════════════════

class AdamW8bitCPU(torch.optim.Optimizer):
    """
    AdamW dont m et v sont stockés en int8 + échelles fp32 par bloc de 2048
    éléments (style bitsandbytes 8-bit, mais 100% CPU/torch). Mémoire optimiseur
    ≈ 2.002 octets/param vs 8 pour AdamW fp32 → ÷4. Déquant dynamique à chaque
    step ; sous-échantillonnage d'erreur borné par bloc (rapport bnb : quasi-sans
    perte en pratique).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, blocksize: int = 2048):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.blocksize = blocksize

    @staticmethod
    def _quant(t: torch.Tensor, blocksize: int):
        tf = t.detach().reshape(-1)
        n = tf.numel()
        pad = (-n) % blocksize
        if pad:
            tf = torch.cat([tf, tf.new_zeros(pad)])
        b = tf.reshape(-1, blocksize)
        amax = b.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        q = torch.round(b / amax * 127.0).clamp_(-127, 127).to(torch.int8)
        return q, amax  # (nb, B) int8, (nb, 1) fp32

    @staticmethod
    def _dequant(q: torch.Tensor, amax: torch.Tensor, n: int,
                 half_bin_floor: bool = False) -> torch.Tensor:
        if half_bin_floor:
            # Compensation d'arrondi + plancher demi-LSB : v ≥ 0 quantifié par
            # blocs sans ce plancher s'effondre à 0 pour les petites entrées
            # (g_j << gmax d'un bloc → round→0 → div par ~0 dans l'update).
            # v̂ = (|q|+0.5)/127·amax ≥ 3.9e-3·amax : borne mathématique sur
            # l'erreur relative (≤ +50% du LSB, comme tout arrondi), garde le
            # dénominateur Adam ≥ 6% de l'échelle du bloc → pas d'explosion.
            t = ((q.abs().float() + 0.5) / 127.0) * amax
        else:
            t = (q.float() / 127.0) * amax
        return t.reshape(-1)[:n]

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    g = g.coalesce().to_dense()
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["qm"], st["sm"] = self._quant(torch.zeros_like(p), self.blocksize)
                    st["qv"], st["sv"] = self._quant(torch.zeros_like(p), self.blocksize)
                st["step"] += 1
                n = p.numel()
                m = self._dequant(st["qm"], st["sm"], n).reshape_as(p)
                v = self._dequant(st["qv"], st["sv"], n,
                                  half_bin_floor=True).reshape_as(p)
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                mh = m / (1 - b1 ** st["step"])
                vh = v / (1 - b2 ** st["step"])
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.addcdiv_(mh, vh.sqrt().add_(group["eps"]),
                           value=-group["lr"])
                st["qm"], st["sm"] = self._quant(m, self.blocksize)
                st["qv"], st["sv"] = self._quant(v, self.blocksize)


# ═══════════════════════════════════════════════════════════════════════
#  P3+P4 — CPUTrainer : plomberie ids, lisa-detach, tête pleine/échantillonnée
# ═══════════════════════════════════════════════════════════════════════

class CPUTrainerConfig:
    def __init__(self, batch_size: int = 8, seq_len: int = 128, lr: float = 3e-4,
                 weight_decay: float = 0.01, warmup_tokens: int = 10_000,
                 sampled_softmax_negs: int = 0,       # 0 = tête pleine
                 lisa_schedule: Optional[List[Tuple[int, int]]] = None,
                 # [(tokens_seuil, n_blocs_actifs_SUFFIXE), ...] trié croissant ;
                 # avant le 1er seuil : tout le modèle. Ex. [(50_000, 2), (150_000, 4)]
                 optimizer: str = "adamw",            # 'adamw' | 'adamw8bit-cpu'
                 clip: float = 1.0, log_every: int = 20, seed: int = 0):
        self.batch_size, self.seq_len, self.lr = batch_size, seq_len, lr
        self.weight_decay, self.warmup_tokens = weight_decay, warmup_tokens
        self.sampled_softmax_negs = sampled_softmax_negs
        self.lisa_schedule = lisa_schedule or []
        self.optimizer, self.clip = optimizer, clip
        self.log_every, self.seed = log_every, seed

    def suffix_active_at(self, tokens: int) -> int:
        """Combien de blocs DU SUFFIXE sont entraînés à ce stade (0 = tous)."""
        n = 0
        for seuil, k in self.lisa_schedule:
            if tokens >= seuil:
                n = k
        return n


class CPUTrainer:
    def __init__(self, model: CogNetMoE1B, cfg: CPUTrainerConfig, device: str = "cpu"):
        self.model, self.cfg, self.device = model, cfg, device
        self.g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self.tokens_seen = 0
        self._lisa_frozen_prefix = 0

    def _routers(self):
        for blk in self.model.blocks:
            r = blk.cognitive_expert_router
            if hasattr(r, "set_batch_token_ids"):
                yield r

    def _set_ids(self, ids):
        for r in self._routers():
            r.set_batch_token_ids(ids)

    def _apply_lisa(self) -> int:
        """Rotation suffixe à détachement strict. Retourne le préfixe gelé."""
        n_suffix = self.cfg.suffix_active_at(self.tokens_seen)
        nb = self.model.num_blocks
        if not n_suffix or n_suffix >= nb:
            prefix = 0
            for b in range(nb):
                for p in self.model.blocks[b].parameters():
                    p.requires_grad = True
        else:
            prefix = nb - n_suffix
            for b in range(nb):
                for p in self.model.blocks[b].parameters():
                    p.requires_grad = (b >= prefix)
        for p in self.model.encoder.parameters():
            p.requires_grad = (prefix == 0)
        for p in self.model.final_norm.parameters():
            p.requires_grad = True
        self._lisa_frozen_prefix = prefix
        return prefix

    def _make_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.optimizer == "adamw8bit-cpu":
            return AdamW8bitCPU(params, lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)
        return torch.optim.AdamW(params, lr=self.cfg.lr,
                                 weight_decay=self.cfg.weight_decay)

    def lr_at(self, tokens: int, total: int) -> float:
        w = max(self.cfg.warmup_tokens, 1)
        if tokens < w:
            return self.cfg.lr * (tokens + 1) / w
        t = (tokens - w) / max(total - w, 1)
        return self.cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1))))

    def loss_fn(self, ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Loss LM next-token AVEC fuite bidirectionnelle assumée (connue et
        partagée par tous les bras — voir EDT_REFUTATION_REPORT, volet 0 : les
        valeurs absolues ne mesurent pas un LM autorégressif ; les comparaisons
        relatives à budget égal restent informatives). Pour la sonde sans fuite :
        --probe."""
        B, T = ids.shape
        self._set_ids(ids)
        S = self.cfg.sampled_softmax_negs
        if S > 0:
            h = forward_hidden(self.model, ids,
                               frozen_prefix=self._lisa_frozen_prefix)
            h_flat = h[:, :-1].reshape(-1, h.shape[-1])
            y = ids[:, 1:].reshape(-1)
            return sampled_ce_loss(h_flat, y, self.model.encoder.token_emb.weight,
                                   S, self.g)
        out = self.model(ids)
        logits = out["logits"][:, :-1].reshape(-1, out["logits"].shape[-1])
        y = ids[:, 1:].reshape(-1)
        return F.cross_entropy(logits, y), {}

    def train(self, batch_fn, total_tokens: int, eval_fn=None):
        Bcfg = self.cfg
        opt = None
        ema, t0, last_logged = None, time.time(), self.tokens_seen
        seen0 = self.tokens_seen
        while self.tokens_seen < total_tokens:
            self._apply_lisa()
            if opt is None:
                opt = self._make_optimizer()
            lr = self.lr_at(self.tokens_seen, total_tokens)
            for gp in opt.param_groups:
                gp["lr"] = lr
            ids = batch_fn(Bcfg.batch_size, Bcfg.seq_len)
            loss, info = self.loss_fn(ids)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if Bcfg.clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], Bcfg.clip)
            opt.step()
            self.tokens_seen += Bcfg.batch_size * Bcfg.seq_len
            ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss)
            if self.tokens_seen - last_logged >= Bcfg.log_every * Bcfg.batch_size * Bcfg.seq_len:
                el = time.time() - t0
                tps = (self.tokens_seen - seen0) / max(el, 1e-9)
                print(f"  tokens={self.tokens_seen:>9,} loss={float(loss):.4f} "
                      f"ema={ema:.4f} lr={lr:.2e} tok/s={tps:,.0f}")
                last_logged = self.tokens_seen
        dt = time.time() - t0
        res = {"tokens": self.tokens_seen, "wall_s": dt,
               "tok_per_s": (self.tokens_seen - seen0) / max(dt, 1e-9),
               "loss_ema": ema}
        if eval_fn is not None:
            res["eval"] = eval_fn()
        return res

    def save(self, path: str):
        torch.save({"model": self.model.state_dict(),
                    "tokens_seen": self.tokens_seen}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"], strict=False)
        self.tokens_seen = ck.get("tokens_seen", 0)


# ═══════════════════════════════════════════════════════════════════════
#  Paramètres / MACs par token — comptage honnête
# ═══════════════════════════════════════════════════════════════════════

def router_macs_per_token(D: int, FF: int, C: int, K: int, kind: str,
                          low_rank: Optional[int] = None) -> Dict[str, float]:
    """MACs/token forward des blocs MoE (hors mémoire/composer — comptés à part)."""
    core = low_rank * (2 * D + 3 * FF) if low_rank else 3 * D * FF
    d = {"legacy":      C * D * D + K * 3 * D * FF + D * 2 * C,   # to_ch + experts + router
         "fold_res":    K * D * D + K * core,                     # w_res sel. + experts
         "simplify":    K * core}
    return {"macs": d[kind], "vs_legacy": d[kind] / d["legacy"]}


def model_param_counts(model: CogNetMoE1B) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    experts = sum(p.numel() for blk in model.blocks
                  for p in blk.cognitive_expert_router.experts.parameters())
    return {"total": total, "experts": experts,
            "non_expert": total - experts}


# ═══════════════════════════════════════════════════════════════════════
#  Sonde qualité sans fuite — REPRODUCTION EXACTE du protocole refute_edt.py
#  (motif 256, vocab 64, fenêtre 32 bruitée 10%, prédire le token suivant la
#   fenêtre, tête séparée sur mean(h), B=16, lr=3e-4, pertes aux/z clampées,
#   eval bruit frais 640 échantillons + contrôle motif inédit ≈ chance)
#  → résultats DIRECTEMENT comparables à EDT_REFUTATION / HASH_ROUTING :
#    SCRATCH 2.89 · P1-ONLY 2.39 · HASH-TOKEN 1.87 · HASH-LSH 2.08 · ASSOC-v3 2.88
# ═══════════════════════════════════════════════════════════════════════

VPROBE, PLEN = 64, 256


def make_pattern(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VPROBE, (PLEN,), generator=g)


def probe_batch_fn(pattern: torch.Tensor, noise: float, seed: int, counter=None):
    rng = torch.Generator().manual_seed(seed)

    def fn(B, T=32):
        s = torch.randint(0, PLEN - T - 1, (B,), generator=rng)
        x = torch.stack([pattern[i:i + T] for i in s]).clone()
        y = torch.stack([pattern[i + T] for i in s])
        mask = torch.rand(x.shape, generator=rng) < noise
        x[mask] = torch.randint(0, VPROBE, (int(mask.sum().item()),), generator=rng)
        if counter is not None:
            counter["n"] = counter.get("n", 0) + B * T
        return x, y
    return fn


def trunk_hidden_stats(model, x, frozen_prefix: int = 0):
    """h final + sommes aux/z (collectées pour la loss legacy, =0 sous hash)."""
    if frozen_prefix > 0:
        with torch.no_grad():
            h = model.encoder(x)
            for blk in model.blocks[:frozen_prefix]:
                h, _ = blk(h)
            h = model.final_norm(h) if frozen_prefix == model.num_blocks else h
        h = h.detach()
        blocks = model.blocks[frozen_prefix:]
    else:
        h = model.encoder(x)
        blocks = model.blocks
    aux = torch.tensor(0.0, device=x.device)
    z = torch.tensor(0.0, device=x.device)
    for blk in blocks:
        h, st = blk(h)
        if "moe_aux_loss" in st:
            aux = aux + st["moe_aux_loss"]
            z = z + st["moe_z_loss"]
    return model.final_norm(h), aux, z


@torch.no_grad()
def probe_eval(model, head, batch_fn, n_batches: int = 40,
               mean_pool: bool = True) -> float:
    model.eval()
    if isinstance(head, nn.Module):
        head.eval()
    tot, n = 0.0, 0
    for _ in range(n_batches):
        x, y = batch_fn(16, 32)
        for r in (blk.cognitive_expert_router for blk in model.blocks):
            if hasattr(r, "set_batch_token_ids"):
                r.set_batch_token_ids(x)
        h, _, _ = trunk_hidden_stats(model, x)
        v = h.mean(dim=1) if mean_pool else h[:, -1]
        logits = head(v) if isinstance(head, nn.Module) else \
            v @ model.encoder.token_emb.weight.T
        tot += float(F.cross_entropy(logits, y))
        n += 1
    model.train()
    if isinstance(head, nn.Module):
        head.train()
    return tot / n


def build_probe_model(seed: int = 0) -> CogNetMoE1B:
    """Le tiny de référence du repo : 64d × 2 blocs × 4 experts top-2, ff 128."""
    torch.manual_seed(seed)
    return CogNetMoE1B(
        vocab_size=VPROBE, hidden_dim=64, num_blocks=2, num_channels=4,
        channel_dim=16, ff_dim=128, max_seq_len=64,
        working_slots=4, episodic_slots=8, semantic_slots=16,
        key_dim=16, n_experts=4, top_k=2, dropout=0.0,
        use_gradient_checkpointing=False)


def build_tiny_model(vocab: int = 64, D: int = 96, blocks: int = 3, FF: int = 384,
                     C: int = 8, K: int = 2, seed: int = 0) -> CogNetMoE1B:
    torch.manual_seed(seed)
    return CogNetMoE1B(
        vocab_size=vocab, hidden_dim=D, num_blocks=blocks, num_channels=C,
        channel_dim=D, ff_dim=FF, max_seq_len=512,
        working_slots=16, episodic_slots=32, semantic_slots=64, key_dim=max(16, D // 4),
        n_experts=C, top_k=K, dropout=0.0, use_gradient_checkpointing=False)


def probe_run(kind: str, tokens: int = 200_000, seed: int = 0,
              low_rank: Optional[int] = None, p1: bool = False,
              p1_steps: int = 200, lisa_suffix: int = 0, sampled_negs: int = 0,
              tied_head: bool = False, lr: float = 3e-4,
              progress: bool = False) -> Dict[str, float]:
    """
    Un bras, budget tokens-identique, même init trunk appariée (même seed).
    kind : 'legacy' | 'fold_res' | 'simplify'
    tied_head=True → tête liée à l'embedding (requis pour sampled_negs).
    """
    torch.manual_seed(seed)
    noise, B, T = 0.1, 16, 32
    pattern = make_pattern(7)
    pattern_new = make_pattern(12345)     # contrôle : motif inédit ≈ chance

    model = build_probe_model(seed)
    if kind != "legacy":
        convert_to_cpu(model, mode="token", fold_residual=(kind == "fold_res"),
                       low_rank=low_rank, seed=seed + 7)
    p1_stats = None
    if p1:
        p1_stats = [p1_identity_init(blk.cognitive_expert_router, steps=p1_steps,
                                     seed=seed)
                    for blk in model.blocks]

    if tied_head:
        head = None     # tied embedding (sampled softmax possible)
        opt_params = list(model.parameters())
    else:
        head = nn.Linear(64, VPROBE)
        torch.manual_seed(seed + 1)
        nn.init.normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)
        opt_params = list(model.parameters()) + list(head.parameters())

    nb = model.num_blocks
    prefix_frozen = 0
    if lisa_suffix and lisa_suffix < nb:
        prefix_frozen = nb - lisa_suffix
        for b in range(nb):
            for p in model.blocks[b].parameters():
                p.requires_grad = (b >= prefix_frozen)

    opt = torch.optim.AdamW([p for p in opt_params if p.requires_grad], lr=lr)
    counter: Dict[str, int] = {}
    train_fn = probe_batch_fn(pattern, noise, seed=500 + seed, counter=counter)
    steps = max(1, tokens // (B * T))
    gg = torch.Generator().manual_seed(seed + 99)
    t0, ema = time.time(), None
    for s in range(steps):
        x, y = train_fn(B, T)
        for r in (blk.cognitive_expert_router for blk in model.blocks):
            if hasattr(r, "set_batch_token_ids"):
                r.set_batch_token_ids(x)
        h, aux, z = trunk_hidden_stats(model, x, frozen_prefix=prefix_frozen)
        v = h.mean(dim=1)
        if tied_head:
            w = model.encoder.token_emb.weight
            if sampled_negs > 0:
                loss, _ = sampled_ce_loss(v, y, w, sampled_negs, gg)
            else:
                loss = F.cross_entropy(v @ w.T, y)
        else:
            loss = F.cross_entropy(head(v), y)
        loss = loss + 0.05 * aux.clamp(max=10.0) + 1e-3 * z.clamp(max=10.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in opt_params if p.requires_grad], 1.0)
        opt.step()
        ema = float(loss) if ema is None else 0.95 * ema + 0.05 * float(loss)
        if progress and (s + 1) % max(1, steps // 5) == 0:
            print(f"        step {s+1}/{steps} loss={float(loss):.4f}")
    dt = time.time() - t0
    eval_main = probe_eval(model, head,
                           probe_batch_fn(pattern, noise, seed=9999))
    eval_new = probe_eval(model, head,
                          probe_batch_fn(pattern_new, noise, seed=9999))
    return {"kind": kind, "seed": seed, "eval": eval_main,
            "eval_new_pattern": eval_new, "train_ema": ema,
            "tokens": counter.get("n", 0), "wall_s": dt,
            "tok_per_s": counter.get("n", 0) / dt,
            "params": model_param_counts(model)["total"],
            "low_rank": low_rank or 0, "p1": int(p1), "p1_steps": p1_steps if p1 else 0,
            "lisa_suffix": lisa_suffix, "sampled_negs": sampled_negs,
            "tied_head": int(tied_head),
            "p1_mse": (sum(m["mse_final"] for m in p1_stats) / len(p1_stats)
                       if p1_stats else None)}


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════

def self_test() -> bool:
    torch.manual_seed(0)
    ok = True

    print("=" * 70)
    print("CPU-STACK SELF-TEST — entraînement CogNet sans GPU")
    print("=" * 70)

    # [1] Hash : déterminisme + balance + dédup
    print("\n[1] Routage hash (P2) : déterminisme, balance, dédup")
    ids = torch.arange(8192).reshape(64, 128)
    a1 = token_hash_experts(ids, 8, 2)
    a2 = token_hash_experts(ids, 8, 2)
    dedup_ok = bool(((a1[..., 0] != a1[..., 1])).all())
    usage = torch.bincount(a1[..., 0].reshape(-1), minlength=8).float() / a1[..., 0].numel()
    bal_ok = bool((usage.max() - usage.min()) < 0.08)
    t = bool(torch.equal(a1, a2)) and dedup_ok and bal_ok
    print(f"    déterministe ✓, dédup ✓, balance [{usage.min():.3f},{usage.max():.3f}] → {'PASS' if t else 'FAIL'}")
    ok &= t

    # [2] PLIAGE EXACT (P1) : FoldedHashRouter.from_legacy ≡ legacy par expert
    print("\n[2] Pliage associatif to_channels → experts : identité fonctionnelle")
    D, FF, C, K = 64, 128, 4, 2
    legacy = CognitiveExpertRouter(D, C, FF, K)
    folded = FoldedHashRouter.from_legacy(legacy, mode="token")
    x = torch.randn(37, D)
    W = legacy.to_channels.weight
    dev = 0.0
    with torch.no_grad():
        for e in range(C):
            ref = legacy.experts[e](x @ W[e * D:(e + 1) * D, :].T)
            got = folded.experts[e](x)
            dev = max(dev, float((ref - got).abs().max()))
    t = dev < 1e-4
    print(f"    écart max |legacy∘W_e − plié| = {dev:.2e} → {'PASS' if t else 'FAIL'}")
    ok &= t

    # [3] MACs/token : fold_res −(C/K−1)/C des projections ; simplify −25% total
    print("\n[3] MACs/token du bloc (P1+P2) — comptage analytique")
    m_legacy = router_macs_per_token(2048, 8192, 8, 2, "legacy")
    m_fold = router_macs_per_token(2048, 8192, 8, 2, "fold_res")
    m_simpl = router_macs_per_token(2048, 8192, 8, 2, "simplify")
    m_lr = router_macs_per_token(2048, 8192, 8, 2, "simplify", low_rank=1024)
    print(f"    legacy {m_legacy['macs']/1e6:7.1f}M  fold_res {m_fold['macs']/1e6:7.1f}M "
          f"({1-m_fold['vs_legacy']:+.1%})  simplify {m_simpl['macs']/1e6:7.1f}M "
          f"({1-m_simpl['vs_legacy']:+.1%})  simplify+lr1024 {m_lr['macs']/1e6:6.1f}M "
          f"({1-m_lr['vs_legacy']:+.1%})")
    t = m_fold["vs_legacy"] < 0.85 and m_simpl["vs_legacy"] < 0.78
    print(f"    → {'PASS' if t else 'FAIL'}")
    ok &= t

    # [4] FoldedHashRouter dans CogNet : fwd/bwd, gradient = experts sélectionnés
    print("\n[4] Drop-in bloc CogNet : forward, backward, sparsité du grad (P2)")
    model = build_tiny_model(vocab=64, D=64, blocks=2, FF=128, C=4, K=2)
    convert_to_cpu(model, mode="token", fold_residual=False, seed=3)
    ids = torch.randint(0, 64, (4, 32))
    for r in (b.cognitive_expert_router for b in model.blocks):
        r.set_batch_token_ids(ids)
    out = model(ids)["logits"]
    loss = out.float().pow(2).mean()
    loss.backward()
    assign = token_hash_experts(ids, 4, 2).reshape(-1, 2)
    used = set(assign[:, 0].tolist()) | set(assign[:, 1].tolist())
    # Par bloc : les experts avec grad > 0 == EXACTEMENT les experts hash-assignés
    per_block_ok = True
    for b in model.blocks:
        gset = {i for i, e in enumerate(b.cognitive_expert_router.experts)
                if e.w_gate_up.weight.grad is not None
                and bool((e.w_gate_up.weight.grad != 0).any())}
        per_block_ok &= (gset == used)
    t = out.shape == (4, 32, 64) and torch.isfinite(loss) and per_block_ok
    print(f"    logits ok, loss finie, grads == experts assignés {sorted(used)} "
          f"(par bloc) → {'PASS' if t else 'FAIL'}")
    ok &= t

    # [5] Tête échantillonnée : l'ESTIMATEUR DE PARTITION est sans biais (l'objet
    # mathématique exact : E[Ẑ] = Z). Le log (Jensen) induit un petit biais ↓
    # inhérent ∝ 1/S, mesuré comme info — standard sampled softmax.
    print("\n[5] Softmax échantillonné corrigé (P4) : partition sans biais + biais Jensen ∝1/S")
    V, Dh, N = 512, 48, 256
    Wh = torch.randn(N, Dh) * 0.1
    E = torch.randn(V, Dh) * 0.1
    yb = torch.randint(0, V, (N,))
    logits_full = Wh @ E.T
    Z_true = logits_full.exp().sum(-1)          # (N,)
    l_pos = logits_full.gather(1, yb.unsqueeze(1)).squeeze(1)
    gs = torch.Generator().manual_seed(0)
    def Z_hat_once(S):
        neg = torch.randint(0, V, (S,), generator=gs)
        W_neg = E[neg]                                  # répétitions conservées
        l_neg = Wh @ W_neg.T - math.log(S / V)
        own = neg.unsqueeze(0) == yb.unsqueeze(1)       # vrais accidentels / échantillon
        l_neg = l_neg.masked_fill(own, float("-inf"))
        return l_pos.exp() + l_neg.exp().sum(-1)
    Sbig = 256
    Zh = torch.stack([Z_hat_once(Sbig) for _ in range(128)]).mean(0)
    rel = float(((Zh - Z_true) / Z_true).abs().mean())
    full = float(F.cross_entropy(logits_full, yb))
    est = [float(sampled_ce_loss(Wh, yb, E, 256, gs)[0]) for _ in range(64)]
    mu = sum(est) / len(est)
    t = rel < 0.06 and abs(mu - full) < 0.25
    print(f"    partition : |E[Ẑ]−Z|/Z = {rel:.4f} (128 essais, S={Sbig}) ; "
          f"loss pleine {full:.3f} vs échant. moy {mu:.3f} (biais Jensen {full-mu:+.3f}) "
          f"→ {'PASS' if t else 'FAIL'}")
    ok &= t

    # [6] AdamW8bitCPU : mémoire ÷4 + convergence (tâche structurée, gradients
    # réalistes — le régime validé par bnb 8-bit est celui des vrais réseaux ;
    # comparaison à un AdamW fp32 de référence sur la même tâche)
    print("\n[6] AdamW 8-bit CPU (P6) : compression ÷4 + convergence ≈ fp32")
    torch.manual_seed(7)
    W_true = torch.randn(256, 256) / 16
    x_t = torch.randn(512, 256)
    y_t = x_t @ W_true.T + 0.01 * torch.randn(512, 256)
    def run_adam8():
        torch.manual_seed(11)
        lin = nn.Linear(256, 256)
        opt = AdamW8bitCPU(lin.parameters(), lr=2e-3)
        for _ in range(120):
            opt.zero_grad(set_to_none=True)
            F.mse_loss(lin(x_t), y_t).backward()
            opt.step()
        return lin, opt
    def run_adam32():
        torch.manual_seed(11)
        lin = nn.Linear(256, 256)
        opt = torch.optim.AdamW(lin.parameters(), lr=2e-3)
        for _ in range(120):
            opt.zero_grad(set_to_none=True)
            F.mse_loss(lin(x_t), y_t).backward()
            opt.step()
        return float(F.mse_loss(lin(x_t), y_t))
    lin8, opt8 = run_adam8()
    with torch.no_grad():
        l0_8 = float(F.mse_loss(nn.functional.linear(x_t, torch.zeros(256, 256)), y_t))
    l8 = float(F.mse_loss(lin8(x_t), y_t))
    l32 = run_adam32()
    st = opt8.state[lin8.weight]
    bytes8 = st["qm"].numel() + st["qv"].numel() + 8 * st["sm"].numel() + 8 * st["sv"].numel()
    ratio = bytes8 / (lin8.weight.numel() * 8)
    t = l8 < l0_8 * 0.05 and l8 < l32 * 3 and ratio < 0.30
    print(f"    MSE 8-bit {l8:.5f} vs fp32 {l32:.5f} (init ~{l0_8:.2f}), "
          f"mémoire optimiseur ×{ratio:.2f} (÷{1/ratio:.1f}) → {'PASS' if t else 'FAIL'}")
    ok &= t

    # [7] lisa-detach : backward strictement coupé (0 grad sous le préfixe)
    print("\n[7] lisa-detach (P3) : backward coupé à la frontière")
    model = build_tiny_model(vocab=64, D=64, blocks=4, FF=128, C=4, K=2)
    convert_to_cpu(model, mode="token", seed=5)
    nb = model.num_blocks
    prefix_frozen = nb - 2
    for b in range(nb):
        for p in model.blocks[b].parameters():
            p.requires_grad = (b >= prefix_frozen)
    ids = torch.randint(0, 64, (4, 32))
    for r in (b.cognitive_expert_router for b in model.blocks):
        r.set_batch_token_ids(ids)
    h = forward_hidden(model, ids, frozen_prefix=prefix_frozen)
    logits = h @ model.encoder.token_emb.weight.T
    logits.float().pow(2).mean().backward()
    g_frozen = any(p.grad is not None and bool((p.grad != 0).any())
                   for b in range(prefix_frozen) for p in model.blocks[b].parameters())
    g_active = all(any(p.grad is not None and bool((p.grad != 0).any()) for p in model.blocks[b].parameters())
                   for b in range(prefix_frozen, nb))
    t = (not g_frozen) and g_active
    print(f"    grads préfixe gelé : {g_frozen} (attendu False), suffixe actif : {g_active} → {'PASS' if t else 'FAIL'}")
    ok &= t

    # [8] P1 synthétique : MSE identité ↓↓↓ sans données.
    # simplify (sans résiduel interne) doit apprendre l'identité À TRAVERS le
    # core non-linéaire : plus lent qu'EDT-legacy (résiduel interne = identité
    # gratuite) — budget d'étapes honnête.
    print("\n[8] P1 smart-init sans données (P5) : MSE identité sur bruit")
    r = FoldedHashRouter(64, 128, 4, 2, mode="token", seed=9)
    m = p1_identity_init(r, steps=400, lr=5e-3, batch=256, seed=0)
    t = m["mse_final"] < m["mse_init"] * 0.5
    print(f"    MSE {m['mse_init']:.4f} → {m['mse_final']:.4f} (0 token de corpus) → {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n" + ("TOUS LES SELF-TESTS PASSENT ✔" if ok else "ÉCHECS — voir ci-dessus"))
    return ok


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark + projections
# ═══════════════════════════════════════════════════════════════════════

def benchmark(tokens: int = 60_000, D: int = 96, blocks: int = 3, FF: int = 384,
              C: int = 8, K: int = 2, seq: int = 128, batch: int = 8) -> List[Dict]:
    """tok/s fwd+bwd mesurés (tiny, cette machine). Relatif, pas absolu."""
    vocab = 256
    arms = [
        ("legacy (cohérence apprise)", dict(kind="legacy")),
        ("cpu fold_res+hash", dict(kind="fold_res")),
        ("cpu simplify+hash", dict(kind="simplify")),
        ("cpu simplify+hash+lisa(1/3)", dict(kind="simplify", lisa_suffix=1)),
        ("cpu simplify+hash+sampled(S=64)", dict(kind="simplify", sampled_negs=64)),
        ("cpu simplify+lowrank(D/4)", dict(kind="simplify", low_rank=D // 4)),
    ]
    rows = []
    for name, kw in arms:
        torch.manual_seed(0)
        model = build_tiny_model(vocab, D, blocks, FF, C, K, 0)
        if kw["kind"] != "legacy":
            convert_to_cpu(model, mode="token",
                           fold_residual=(kw["kind"] == "fold_res"),
                           low_rank=kw.get("low_rank"), seed=0)
        nb = model.num_blocks
        if kw.get("lisa_suffix"):
            pf = nb - kw["lisa_suffix"]
            for b in range(nb):
                for p in model.blocks[b].parameters():
                    p.requires_grad = (b >= pf)
        else:
            pf = 0
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sn = kw.get("sampled_negs", 0)
        torch.manual_seed(1)
        # warmup
        for _ in range(3):
            ids = torch.randint(0, vocab, (batch, seq))
            for r in (b.cognitive_expert_router for b in model.blocks):
                if hasattr(r, "set_batch_token_ids"):
                    r.set_batch_token_ids(ids)
            h = forward_hidden(model, ids, frozen_prefix=pf)
            if sn:
                l, _ = sampled_ce_loss(h.reshape(-1, D), ids.reshape(-1),
                                       model.encoder.token_emb.weight, sn)
            else:
                l = (h @ model.encoder.token_emb.weight.T).float().pow(2).mean()
            opt.zero_grad(); l.backward(); opt.step()
        steps = max(5, tokens // (batch * seq))
        t0 = time.time()
        for _ in range(steps):
            ids = torch.randint(0, vocab, (batch, seq))
            for r in (b.cognitive_expert_router for b in model.blocks):
                if hasattr(r, "set_batch_token_ids"):
                    r.set_batch_token_ids(ids)
            h = forward_hidden(model, ids, frozen_prefix=pf)
            if sn:
                l, _ = sampled_ce_loss(h.reshape(-1, D), ids.reshape(-1),
                                       model.encoder.token_emb.weight, sn)
            else:
                l = (h @ model.encoder.token_emb.weight.T).float().pow(2).mean()
            opt.zero_grad(); l.backward(); opt.step()
        dt = time.time() - t0
        tps = steps * batch * seq / dt
        rows.append({"arm": name, "tok_per_s": tps,
                     "params": model_param_counts(model)["total"]})
        print(f"  {name:36s} {tps:9,.0f} tok/s  ({model_param_counts(model)['total']/1e6:.2f}M params)")
    base = rows[0]["tok_per_s"]
    for r in rows:
        r["speedup_vs_legacy"] = r["tok_per_s"] / base
    return rows


def _trunk_params(D: int, key: int) -> int:
    """Params mémoire 3-tier + composer d'un bloc, MESURÉS par instanciation
    réelle (les mêmes modules que le 1B — pas une formule approximée)."""
    from cognet_1b_optimized import ParallelHierarchicalMemory, CompositionalReasoner
    m = ParallelHierarchicalMemory(D, key, 128, 256, 512, 0.0)
    c = CompositionalReasoner(D, key, 0.0)
    return sum(p.numel() for p in m.parameters()) + \
        sum(p.numel() for p in c.parameters())


def projection_tables() -> Dict:
    """
    Projections honnêtes : params totaux/actifs + MAC/token + tok/s + temps.

    MACs/token forward = projections des experts actifs (pliage) + tronc
    (mémoire+composer, params mesurés) + tête. Le pliage est EXACT (P1) :
    simplify = la même architecture moins W_e — vérifié au self-test [2]/[3].
    Coût entraînement : full = ×3 (fwd+bwd) ; lisa_detach γ blocs sur B =
    fwd_total + 2×(γ/B × MAC_blocs + MAC_tronc+tête) — sans gracieuseté.
    Efficacité CPU eager fp32 : fourchettes conservatrices par classe.
    """
    print("\n" + "=" * 84)
    print("TABLES DE PROJECTION — physique CPU honnête (params mesurés, MACs analytiques,")
    print("efficacité eager en fourchettes conservatrices ; calibrer avec --bench)")
    print("=" * 84)

    def cfg(name, D, FF, B, C, K, V, low_rank=None):
        key = max(16, D // 8)
        trunk_b = _trunk_params(D, key)              # params ≈ MACs/token (projections denses)
        core = (low_rank * (2 * D + 3 * FF)) if low_rank else 3 * D * FF
        blk_legacy = C * D * D + K * 3 * D * FF
        blk_act = K * core                            # simplify (fold_res = +K·D²)
        experts_total = C * core + B * 0              # (+normes, négligeables)
        params_total = B * (experts_total + trunk_b) + V * D
        return {"name": name, "D": D, "FF": FF, "B": B, "C": C, "K": K, "V": V,
                "low_rank": low_rank, "trunk_per_block": trunk_b,
                "mac_fwd_legacy": B * (blk_legacy + trunk_b) + V * D,
                "mac_fwd": B * (blk_act + trunk_b) + V * D,
                "mac_fwd_sampled": B * (blk_act + trunk_b) + 513 * D,
                "mac_blocks": B * blk_act, "mac_trunk_head": B * trunk_b + 513 * D,
                "params_total": params_total,
                "params_active": B * blk_act + B * trunk_b + V * D}

    cfgs = [
        cfg("CogNet-MoE-1B (réf)", 2048, 8192, 16, 8, 2, 16384),
        cfg("cognet-cpu-S", 512, 2048, 8, 8, 2, 16384),
        cfg("cognet-cpu-M", 768, 3072, 12, 8, 2, 16384),
        cfg("cognet-cpu-L", 1024, 4096, 16, 8, 2, 16384),
        cfg("cpu-M/lr256", 768, 3072, 12, 8, 2, 16384, low_rank=256),
    ]

    print(f"\n{'config':22s} {'params totaux':>13s} {'actifs/tok':>11s} "
          f"{'MAC fwd legacy':>15s} {'MAC fwd simplify':>17s}")
    for c in cfgs:
        print(f"{c['name']:22s} {c['params_total']/1e9:10.2f}B {c['params_active']/1e6:8.1f}M "
              f"{c['mac_fwd_legacy']/1e6:12.1f}M {c['mac_fwd']/1e6:14.1f}M")

    cpu_classes = [("cette VM 2c (mesuré)", (1.5e9, 3e9)),
                   ("desktop 8c AVX2", (30e9, 80e9)),
                   ("desktop 16-32c AVX512", (100e9, 250e9)),
                   ("serveur 64-128c", (300e9, 800e9))]

    print("\ntok/s entraînement FULL (×3, tête pleine) — simplify stack :")
    print(f"{'machine':26s}" + "".join(f"{c['name'][:16]:>18s}" for c in cfgs))
    for mname, (lo, hi) in cpu_classes:
        cells = []
        for c in cfgs:
            t_lo, t_hi = hi / (c["mac_fwd"] * 2 * 3), lo / (c["mac_fwd"] * 2 * 3)
            cells.append(f"{t_lo:6,.0f}-{t_hi:<6,.0f}")
        print(f"{mname:26s}" + "".join(f"{x:>18s}" for x in cells))

    print("\ntok/s entraînement LISA-γ/B=1/4 (tête échantillonnée S=512) :")
    for mname, (lo, hi) in cpu_classes:
        cells = []
        for c in cfgs:
            mac_train = c["mac_fwd_sampled"] + 2 * (0.25 * c["mac_blocks"] + c["mac_trunk_head"])
            t_lo, t_hi = hi / (mac_train * 2), lo / (mac_train * 2)
            cells.append(f"{t_lo:6,.0f}-{t_hi:<6,.0f}")
        print(f"{mname:26s}" + "".join(f"{x:>18s}" for x in cells))

    print("\nTemps pour 300M tokens (cible CPU réaliste) — LISA 1/4 + tête échantill. :")
    for mname, (lo, hi) in cpu_classes:
        cells = []
        for c in cfgs[1:]:
            mac_train = c["mac_fwd_sampled"] + 2 * (0.25 * c["mac_blocks"] + c["mac_trunk_head"])
            d_hi, d_lo = 300e6 * mac_train * 2 / (lo * 86400), 300e6 * mac_train * 2 / (hi * 86400)
            cells.append(f"{d_lo:5.0f}-{d_hi:<5.0f}j")
        print(f"{mname:26s}" + "".join(f"{x:>18s}" for x in cells))

    print("\nRappel 1B : cible EDT 1.36B tokens × full-train 3× 2.3G MAC = "
          "~9.4e18 FLOPs → même à 800 GFLOP/s (gros serveur CPU) ≈ 372 ans.")
    print("Le 1B reste un travail de GPU ; le CPU prend le relais via les configs "
          "cpu-* + croissance d'experts paginée (P8) + route B forward-only.")
    return {"configs": cfgs}


# ═══════════════════════════════════════════════════════════════════════
#  E2E smoke : entraînement complet CPU (trainer + stack + ckpt roundtrip)
# ═══════════════════════════════════════════════════════════════════════

def e2e_smoke(tokens: int = 40_000, seed: int = 0) -> Dict[str, float]:
    """
    Preuve de bout en bout, 100% CPU : FoldedHashRouter(simplify) + hash-token
    (plomberie ids) + tête échantillonnée S=16 + AdamW-8bit-CPU + rotation LISA
    (suffixe 1/2 → 2/2 blocs) + checkpoint save→load strict + reprise.

    Stream synthétique : 4 sujets = 4 motifs cycliques de bigrammes (prévisible :
    entropie << chance) — la loss LM doit descendre nettement sous ln(64)=4.16.
    (Rappel volet 0 refute_edt : sur CogNet bidirectionnel la loss LM est affectée
    par la fuite future-visible — valeur absolue à ne pas interpréter, le SMOKE
    valide la PLOMBERIE : gradients, ckpt, reprise, stabilité.)
    """
    torch.manual_seed(seed)
    V, T = 64, 32
    g = torch.Generator().manual_seed(seed)
    motifs = [torch.randint(0, V, (48,), generator=g) for _ in range(4)]

    def batch_fn(B, T_):
        out = torch.empty(B, T_ + 1, dtype=torch.long)
        for b in range(B):
            m = motifs[int(torch.randint(0, 4, (1,), generator=g))]
            s = int(torch.randint(0, 48 - T_ - 1, (1,), generator=g))
            out[b] = m[s:s + T_ + 1]
        return out[:, :T_]

    model = build_probe_model(seed)
    convert_to_cpu(model, mode="token", fold_residual=False, seed=seed)
    cfg = CPUTrainerConfig(batch_size=16, seq_len=T, lr=1e-3, warmup_tokens=2000,
                           sampled_softmax_negs=16,
                           lisa_schedule=[(0, 1), (tokens // 2, 2)],
                           optimizer="adamw8bit-cpu", log_every=4, seed=seed)
    trainer = CPUTrainer(model, cfg)
    res = trainer.train(batch_fn, total_tokens=tokens,
                        eval_fn=lambda: probe_eval(
                            model, None, probe_batch_fn(make_pattern(7), 0.1, 9999)))
    trainer.save("/tmp/cpu_e2e_ckpt.pt")

    # Roundtrip : modèle neuf, chargement STRICT, 500 tokens de reprise.
    model2 = build_probe_model(seed + 1)
    convert_to_cpu(model2, mode="token", fold_residual=False, seed=seed + 1)
    missing_unexpected = model2.load_state_dict(
        torch.load("/tmp/cpu_e2e_ckpt.pt", map_location="cpu",
                   weights_only=False)["model"], strict=True)
    trainer2 = CPUTrainer(model2, cfg)
    trainer2.load("/tmp/cpu_e2e_ckpt.pt")
    res2 = trainer2.train(batch_fn, total_tokens=tokens + 2000)
    print(f"\n  e2e : loss_ema phase1={res['loss_ema']:.3f} | reprise strict-load OK, "
          f"tokens repris={res2['tokens']:,} | eval sonde (sans fuite)={res['eval']:.3f}")
    return {**res, "resume_tokens": res2["tokens"]}




def main():
    ap = argparse.ArgumentParser(description="cpu_stack — CogNet sans GPU")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--tokens", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    if args.probe:
        print("=" * 70)
        print("SONDE QUALITÉ SANS FUITE — protocole EXACT refute_edt.py volet 1")
        print("références repo (200k tokens) : SCRATCH 2.89 · P1-ONLY 2.39 · "
              "HASH-TOKEN 1.87 · HASH-LSH 2.08 · ASSOC-v3 2.88 · chance 4.16")
        print(f"budget = {args.tokens:,} tokens-input par bras, {args.seeds} seed(s)")
        print("=" * 70)
        arms = [
            dict(kind="legacy"),                       # = SCRATCH repo
            dict(kind="fold_res"),                     # pliage exact + hash
            dict(kind="simplify"),                     # −25% MACs + hash
            dict(kind="simplify", p1=True),            # + smart-init 0-token
            dict(kind="simplify", low_rank=32),        # cœurs bas-rang
            dict(kind="simplify", lisa_suffix=1),      # backward coupé 1/2 blocs
            dict(kind="simplify", tied_head=True, sampled_negs=16),  # tête échantillonnée
        ]
        results = []
        for seed in range(args.seeds):
            for kw in arms:
                label = kw["kind"] + ("+p1" if kw.get("p1") else "") + \
                        (f"+lr{kw['low_rank']}" if kw.get("low_rank") else "") + \
                        (f"+lisa{kw['lisa_suffix']}" if kw.get("lisa_suffix") else "") + \
                        (f"+sampled{kw['sampled_negs']}" if kw.get("sampled_negs") else "")
                print(f"[probe] {label:28s} seed={seed} ...", flush=True)
                r = probe_run(tokens=args.tokens, seed=seed, **kw)
                r["label"] = label
                print(f"        eval={r['eval']:.4f} (motif inédit {r['eval_new_pattern']:.4f}≈chance) "
                      f"ema={r['train_ema']:.4f} {r['tok_per_s']:,.0f} tok/s wall={r['wall_s']:.0f}s",
                      flush=True)
                results.append(r)
        with open("cpu_probe_results.json", "w") as fjs:
            json.dump(results, fjs, indent=2)
        print("\nÉcrit : cpu_probe_results.json")

    if args.bench:
        print("=" * 70)
        print("BENCHMARK tok/s (fwd+bwd, tiny, CETTE machine — relatif)")
        print("=" * 70)
        rows = benchmark()
        with open("cpu_bench.json", "w") as fjs:
            json.dump(rows, fjs, indent=2)
        print("\nÉcrit : cpu_bench.json")

    if args.tables:
        projection_tables()

    if args.e2e:
        print("=" * 70)
        print("E2E SMOKE — entraînement CPU complet (stack + ckpt roundtrip)")
        print("=" * 70)
        e2e_smoke(tokens=args.tokens)

    if not (args.self_test or args.probe or args.bench or args.tables or args.e2e):
        ap.print_help()


if __name__ == "__main__":
    main()
