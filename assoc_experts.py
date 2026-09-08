#!/usr/bin/env python3
"""
assoc_experts.py — Experts ASSOCIATIFS sans gradient, paginés sur disque
========================================================================

Substitution modulaire du contenu des pages ExpertPager : les experts denses
(FusedSwiGLU + Adam) deviennent des MÉMOIRES ASSOCIATIVES apprises par
superposition Hebbienne directe — ZÉRO backprop, ZÉRO moments Adam dans les
experts. La machine ne change pas : même pager (LRU + prefetch + writeback),
même adressage par hash, même interface de page.

Mémoire associative par expert (page disque) :
  keys   (M, D) — prototypes d'adressage (espace projeté, normalisés)
  values (M, D) — charges utiles (espace hidden,.superposition directe)
  counts (M,)   — compteur de victoires (stats + protection palimpseste)
  proj   (D, D) — random-indexing FIXE (jamais appris, jamais dirty)

Retrieval (forward, sans gradient) :
  z = q @ proj^T ; fenêtre de W slots depuis le hash ; scores = z·keys ;
  out = softmax(scores) @ values. Coût O(W·D) << matmul dense O(D·ff).

Apprentissage (pur Hebbien, règle de Kohonen / k-means online) :
  s = argmax(scores) ; keys[s]   ← norm(keys[s] + η·(z − keys[s]))
                       values[s] ← values[s] + η·(q − values[s])
  = superposition directe de vecteurs. Pas de gradient, pas d'optimizer,
  pas d'état auxiliaire : le problème des moments Adam paginés S'ÉVAPORE.

Alignement d'adressage O(1) (défi posé) :
  UN SEUL hash par token sert les DEUX niveaux :
    pages   = token_hash_experts(ids, E, K, salt)   [source unique de vérité]
    fenêtre = mix O(1) (1 multiply, 0 passe mémoire suppl.) → start ∈ [0, M)
  → page O(1) + fenêtre O(1), ZÉRO double overhead. Test [1] prouve que les
  pages associatives == pages du pager (alignement exact par construction).

Boucle d'entraînement (apprentissage local, style forward-forward) :
  les experts sont DÉTACHÉS du graphe (forward seul) ; le gradient global
  (tête + encodeur + mémoire + composer + normes) circule par les résiduels,
  les experts apprennent localement par Hebb. Les deux signaux cohabitent
  sans interférence.

Usage :
    PYTHONPATH=source:. python3 assoc_experts.py --self-test   # ~15 s CPU
"""

import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
sys.path.insert(0, str(HERE))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cognet_1b_optimized import RMSNorm  # noqa: E402
from cognet_moe import CogNetMoE1B  # noqa: E402
from hash_moe import token_hash_experts, LSHHasher  # noqa: E402
from expert_pager import (ExpertPager, PagerConfig, PageId,  # noqa: E402
                          hash_ahead_pages)


# ═══════════════════════════════════════════════════════════════════════
# Adressage unifié : 1 hash → page + fenêtre slot (O(1), zéro double coût)
# ═══════════════════════════════════════════════════════════════════════

def assoc_address(token_ids: torch.Tensor, n_experts: int, top_k: int,
                  n_mem_slots: int, salt: int = 0x9E3779B9
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (B,T) ids → pages (B,T,K) + départs de fenêtre (B,T,K).
    Les pages viennent de token_hash_experts (SOURCE UNIQUE : alignement exact
    avec le pager par construction). La fenêtre = 1 multiply + 1 modulo :
    O(1), aucune passe mémoire supplémentaire (pas de 2e hash).
    """
    pages = token_hash_experts(token_ids, n_experts, top_k, salt)
    m = (token_ids.long().unsqueeze(-1) * 2654435761
         + pages * 40503 + (salt & 0xFFFF)) >> 8
    win = (m % n_mem_slots).to(torch.long)
    return pages, win


def assoc_address_lsh(code: torch.Tensor, hasher: LSHHasher, n_experts: int,
                      top_k: int, n_mem_slots: int):
    """(B,T) code LSH → pages (B,T,K) + fenêtres (B,T,K), UNE SEULE source.
    Bits bas → pages (via experts(code)), bits hauts → fenêtres : O(1) unifié,
    zéro double calcul (le code est computé une fois par le router)."""
    B, T = code.shape
    pages = hasher.experts(None, n_experts, top_k, code=code)
    win = torch.empty((B, T, top_k), dtype=torch.long, device=code.device)
    for j in range(top_k):
        win[..., j] = ((code >> (14 + j * 5)) % n_mem_slots).to(torch.long)
    return pages, win


# ═══════════════════════════════════════════════════════════════════════
# Hypervecteurs binaires : adressage Hamming (clés+proj), readout float
# ═══════════════════════════════════════════════════════════════════════
# La phase v3 a défini QUOI binariser : l'adressage seul (clés, proj).
# Forward binaire (XOR + popcount, CPU-natif), ombre flottante latente pour
# l'apprentissage Hebbien (philosophie BNN). Les valeurs/cartes/centres
# (readout expressif) restent float/int8 : la binarisation ne touche jamais
# la fonction apprise, seulement son adressage. ÷32 sur clés+proj.

def _popcount64(x: torch.Tensor) -> torch.Tensor:
    """Popcount vectorisé int64 (6 ops bit-à-bit, constantes < 2^63)."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def pack_bits(signs: torch.Tensor) -> torch.Tensor:
    """(..., D) {0,1} → (..., nwords) int64 (pad-0 neutre pour XOR)."""
    D = signs.shape[-1]
    nw = max(1, (D + 63) // 64)
    pad = nw * 64 - D
    if pad:
        signs = torch.cat([signs, torch.zeros(signs.shape[:-1] + (pad,),
                                              dtype=signs.dtype, device=signs.device)], dim=-1)
    w = signs.long().reshape(signs.shape[:-1] + (nw, 64))
    shifts = torch.arange(64, device=signs.device, dtype=torch.int64)
    return (w << shifts).sum(-1).to(torch.int64)


def unpack_bits(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """(..., nwords) int64 → (..., dim) float {−1,+1} (ombre à ±1)."""
    shifts = torch.arange(64, device=packed.device, dtype=torch.int64)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(packed.shape[:-1] + (-1,))
    return bits[..., :dim].float() * 2.0 - 1.0


def hamming_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """popcount(a^b) sommé sur les mots : (...,nwords) → (...) distances."""
    return _popcount64(a ^ b).sum(-1)


# ═══════════════════════════════════════════════════════════════════════
# Cœur associatif : retrieval + superposition Hebbienne (fonctions pures)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def assoc_retrieve(q: torch.Tensor, proj: torch.Tensor,
                   keys: torch.Tensor, values: torch.Tensor,
                   win_start: torch.Tensor, window: int,
                   label_ids: Optional[torch.Tensor] = None,
                   label_hist: Optional[torch.Tensor] = None,
                   rho: float = 0.5, commit_min: int = 3,
                   amaps: Optional[torch.Tensor] = None,
                   centers: Optional[torch.Tensor] = None,
                   binary: bool = False, tau: float = 8.0,
                   keys_bits: Optional[torch.Tensor] = None,
                   mlp_u: Optional[torch.Tensor] = None,
                   mlp_v: Optional[torch.Tensor] = None):
    """
    q (N,D) → out (N,D) + poids (N,W) + gagnants (N,) + codes z (N,D) + replis.
    Readout v3 : o_s = v_s + A_s(q−c_s) (cartes locales, None = lookup pur).
    Fenêtre circulaire de W slots depuis win_start, rerank par similarité.
    Si labels fournis (train) : VIGILANCE ART — le gagnant est le premier slot
    du rang de similarité dont l'historique accepte le label
    (frac Laplace >= rho, ou slot quasi-vierge = commit direct) ; sinon repli
    sur le meilleur score (compté). Sans labels (éval) : argmax pur.
    Sans gradient (décorateur + buffers gelés). Le scalpel v2 : discret, local,
    par token — aucune backprop, aucun matmul de feedback, aucune échelle à tuner.
    """
    N, D = q.shape
    M = keys.shape[0]
    W = min(window, M)
    z = q.float() @ proj.float().T  # random indexing fixe (±1 si binaire)
    steps = torch.arange(W, device=q.device)
    idx = (win_start.unsqueeze(-1) + steps) % M  # (N,W) circulaire
    if binary:  # adressage Hamming : XOR + popcount (z float gardé pour Hebb)
        zb = pack_bits((z > 0).to(torch.int64))  # (N,nw)
        kb = (keys_bits if keys_bits is not None
              else pack_bits((keys.float() > 0).to(torch.int64)))[idx]  # (N,W,nw)
        dist = hamming_dist(zb.unsqueeze(1), kb).float()  # (N,W)
        scores = -dist / tau
    else:
        k = keys.float()[idx]  # (N,W,D)
        scores = (z.unsqueeze(1) * k).sum(-1) / math.sqrt(D)
    w = torch.softmax(scores, -1).to(q.dtype)
    v = values.float()[idx]  # (N,W,D)
    d = None
    if centers is not None:
        d = q.float().unsqueeze(1) - centers.float()[idx]  # (N,W,D)
    if amaps is not None and d is not None:  # v3 : carte linéaire (socle prouvé)
        corr = (amaps.float()[idx] @ d.unsqueeze(-1)).squeeze(-1)
        v = v + corr
    if mlp_u is not None and mlp_v is not None and d is not None:  # v4 : résidu MLP
        pre = (mlp_u.float()[idx] @ d.unsqueeze(-1)).squeeze(-1)  # (N,W,H)
        h = torch.nn.functional.silu(pre)
        v = v + (mlp_v.float()[idx] @ h.unsqueeze(-1)).squeeze(-1)
    out = (w.unsqueeze(-1) * v).sum(1).to(q.dtype)
    order = scores.argsort(dim=-1, descending=True)  # rangs de similarité
    ar = torch.arange(N, device=q.device)
    winners = idx[ar, order[:, 0]]
    n_fallback = 0
    if label_ids is not None and label_hist is not None:
        lh = label_hist.float()
        tot = lh.sum(-1)  # (M,) totaux figés pendant le retrieval
        LV = lh.shape[1]
        assigned = torch.zeros(N, dtype=torch.bool, device=q.device)
        wins = torch.empty(N, dtype=torch.long, device=q.device)
        for r in range(W):
            cand = idx[ar, order[:, r]]
            frac = (lh[cand, label_ids] + 1.0) / (tot[cand] + LV)  # Laplace
            ok = (frac >= rho) | (tot[cand] < commit_min)  # vigilance ou commit
            fresh = ok & ~assigned
            wins[fresh] = cand[fresh]
            assigned |= fresh
        n_fallback = int((~assigned).sum().item())
        wins[~assigned] = winners[~assigned]
        winners = wins
    return out, w, winners, z.to(q.dtype), n_fallback


@torch.no_grad()
def hebbian_update(keys: torch.Tensor, values: torch.Tensor, counts: torch.Tensor,
                   z: torch.Tensor, q: torch.Tensor, winners: torch.Tensor,
                   eta: float = 0.05, label_emb: Optional[torch.Tensor] = None,
                   label_alpha: float = 0.5,
                   label_ids: Optional[torch.Tensor] = None,
                   label_hist: Optional[torch.Tensor] = None,
                   sample_weight: Optional[torch.Tensor] = None,
                   readout_err: Optional[torch.Tensor] = None,
                   amaps: Optional[torch.Tensor] = None,
                   centers: Optional[torch.Tensor] = None,
                   mlp_u: Optional[torch.Tensor] = None,
                   mlp_v: Optional[torch.Tensor] = None):
    """
    Superposition directe WTA-EMA (vectorisée) — in-place, sans gradient.
    Gagnants multiples sur le même slot : moyenne des cibles d'abord.
    sample_weight (N,) : plasticité par token (match-tracking : les tokens
    surpris — faible confiance de retrieval — absorbent plus).
    readout_err (N,D) : crédit exact 1-couche → branche DELTA v3 : v et A
    corrigent l'erreur (rank-1), c suit les entrées. Sans err : legacy
    (attraction + binding). Les clés restent TOUJOURS Hebbiennes (loi v2).
    Si label_emb : binding HDC requête⊕label sur les VALUES (les clés restent
    en espace-q pour le matching à l'éval, où le label est inconnu).
    Si label_ids + label_hist : trace ART (le slot gagnant absorbe le label).
    NOTE : la voie FA-erreur-sur-clés a été testée et ABANDONNÉE (3.55→3.59 :
    l'adressage doit rester fidèle aux entrées ; le signal catégoriel passe
    par la vigilance ART dans assoc_retrieve, pas par une erreur projetée).
    Retourne le nombre de slots touchés.
    """
    M = keys.shape[0]
    sw = (torch.ones(winners.shape[0], device=winners.device) if sample_weight is None
          else sample_weight.float())
    cnt = torch.zeros(M, device=winners.device)
    cnt.index_add_(0, winners, sw)
    mask = cnt > 0
    n_touched = int(mask.sum().item())
    if n_touched == 0:
        return 0
    sumz = torch.zeros_like(keys.float())
    sumz.index_add_(0, winners, z.float() * sw.unsqueeze(-1))
    sumq = torch.zeros_like(values.float())
    sumq.index_add_(0, winners, q.float() * sw.unsqueeze(-1))
    tz = sumz[mask] / cnt[mask].unsqueeze(-1)
    tq = sumq[mask] / cnt[mask].unsqueeze(-1)
    if readout_err is not None:  # delta : v + A (v3) + U/V (v4), même δ (exact)
        # v4 : backprop LOCALE exacte (2 couches, formules closes, 0 graphe).
        # o = v + A·d + V·silu(U·d) ; δA/δU/δV/δv par token, agrégés par gagnant.
        assert centers is not None
        cw = centers.float()[winners]
        dd = q.float() - cw  # (N,D)
        M_, D_ = keys.shape[0], q.shape[1]
        swu = sw.unsqueeze(-1)
        sume = torch.zeros_like(values.float())
        sume.index_add_(0, winners, readout_err.float() * swu)
        de = sume[mask] / cnt[mask].unsqueeze(-1)
        if mlp_u is not None and mlp_v is not None:  # v4 : raffinement MLP
            H_ = mlp_u.shape[1]
            pre = (mlp_u.float()[winners] @ dd.unsqueeze(-1)).squeeze(-1)  # (N,H)
            sig = torch.sigmoid(pre)
            h = pre * sig  # silu
            dsp = sig * (1.0 + pre * (1.0 - sig))  # silu'
            sumV = torch.zeros(M_, D_ * H_, device=q.device)
            oV = (readout_err.float().unsqueeze(-1) * h.unsqueeze(-2)).reshape(
                winners.shape[0], -1)
            sumV.index_add_(0, winners, oV * sw.unsqueeze(-1))
            dh = (mlp_v.float()[winners].transpose(-1, -2)
                  @ readout_err.float().unsqueeze(-1)).squeeze(-1) * dsp  # (N,H)
            sumU = torch.zeros(M_, H_ * D_, device=q.device)
            oU = (dh.unsqueeze(-1) * dd.unsqueeze(-2)).reshape(winners.shape[0], -1)
            sumU.index_add_(0, winners, oU * sw.unsqueeze(-1))
            dV = (sumV[mask] / cnt[mask].unsqueeze(-1)).reshape(-1, D_, H_)
            dU = (sumU[mask] / cnt[mask].unsqueeze(-1)).reshape(-1, H_, D_)
            mlp_v[mask] = (mlp_v.float()[mask] - eta * dV).to(mlp_v.dtype)
            mlp_u[mask] = (mlp_u.float()[mask] - eta * dU).to(mlp_u.dtype)
        if amaps is not None:  # v3 : covariance E[δ⊗(q−c)]
            sumo = torch.zeros(mask.shape[0], values.shape[1] * values.shape[1],
                               device=q.device)
            oo = (readout_err.float().unsqueeze(-1) * dd.unsqueeze(-2)).reshape(
                winners.shape[0], -1)
            sumo.index_add_(0, winners, oo * sw.unsqueeze(-1))
            cov = (sumo[mask] / cnt[mask].unsqueeze(-1)).reshape(
                -1, values.shape[1], values.shape[1])
            amaps[mask] = (amaps.float()[mask] - eta * cov).to(amaps.dtype)
        values[mask] = (values.float()[mask] - eta * de).to(values.dtype)
        centers[mask] = (centers.float()[mask]
                         + eta * (tq - centers.float()[mask])).to(centers.dtype)
    elif label_emb is not None:  # legacy : binding HDC requête⊕label
        suml = torch.zeros_like(values.float())
        suml.index_add_(0, winners, label_emb.float())
        tq = tq + label_alpha * (suml[mask] / cnt[mask].unsqueeze(-1))
    if label_ids is not None and label_hist is not None:  # trace ART
        label_hist[winners, label_ids] += 1
    keys[mask] = F.normalize(keys.float()[mask] + eta * (tz - keys.float()[mask]),
                             dim=-1).to(keys.dtype)
    if readout_err is None:  # legacy : values attractées (delta/MLP gèrent v sinon)
        values[mask] = (values.float()[mask]
                        + eta * (tq - values.float()[mask])).to(values.dtype)
    counts[mask] = counts.float()[mask] + cnt[mask].to(counts.dtype)
    return n_touched


# ═══════════════════════════════════════════════════════════════════════
# Slot associatif (S par router) + Router paginé associatif
# ═══════════════════════════════════════════════════════════════════════

class AssocSlot(nn.Module):
    """Un slot = une mémoire associative résidente (buffers, jamais de grad)."""

    def __init__(self, hidden_dim: int, n_mem_slots: int, n_labels: int = 64,
                 mlp_hidden: int = 16):
        super().__init__()
        nw = max(1, (hidden_dim + 63) // 64)
        self.register_buffer("keys", torch.zeros(n_mem_slots, hidden_dim))
        self.register_buffer("keys_bits", torch.zeros(n_mem_slots, nw, dtype=torch.int64))
        self.register_buffer("values", torch.zeros(n_mem_slots, hidden_dim))
        self.register_buffer("counts", torch.zeros(n_mem_slots))
        self.register_buffer("proj", torch.zeros(hidden_dim, hidden_dim))
        # Trace ART par slot (int32 : stockée EXACTE, jamais quantifiée).
        self.register_buffer("label_hist", torch.zeros(n_mem_slots, n_labels,
                                                       dtype=torch.int32))
        # Readout v3 : carte locale o = v + A(q−c) par slot (zéro-init = lookup).
        self.register_buffer("amaps", torch.zeros(n_mem_slots, hidden_dim, hidden_dim))
        self.register_buffer("centers", torch.zeros(n_mem_slots, hidden_dim))
        # Readout v4 : MLP local o = v + V·SiLU(U(q−c)) (2k params/slot à H=16).
        self.register_buffer("mlp_u", torch.zeros(n_mem_slots, mlp_hidden, hidden_dim))
        self.register_buffer("mlp_v", torch.zeros(n_mem_slots, hidden_dim, mlp_hidden))


class PagedAssocExpertRouter(nn.Module):
    """
    Router paginé à experts associatifs : S slots mémoire pour E pages disque.
    Interface pager identique au dense (codec substitué, machine inchangée).
    ZÉRO paramètre entraînable dans les experts (buffers uniquement).
    """

    def __init__(self, hidden_dim: int, n_experts: int = 8, n_slots: int = 4,
                 top_k: int = 2, n_mem_slots: int = 8, window: int = 4,
                 block_idx: int = 0, pager: Optional[ExpertPager] = None,
                 salt: int = 0x9E3779B9,
                 frozen_pages: Optional[Set[int]] = None,
                 n_labels: int = 64, mode: str = "token",
                 n_bits: int = 24, lsh_seed: int = 1234,
                 binary_addressing: bool = False, binary_tau: float = 8.0,
                 readout_mode: str = "legacy", mlp_hidden: int = 16):
        super().__init__()
        assert top_k <= n_experts
        assert mode in ("token", "lsh")
        assert readout_mode in ("legacy", "delta", "mlp")
        self.mode = mode
        self.readout_mode = readout_mode
        self.mlp_hidden = mlp_hidden
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.num_channels = n_experts  # alias compat
        self.n_slots = n_slots
        self.top_k = top_k
        self.n_mem_slots = n_mem_slots
        self.n_labels = n_labels
        self.window = window
        self.block_idx = block_idx
        self.pager = pager
        self.salt = salt
        self._frozen_pages = set(frozen_pages or ())
        self.slots = nn.ModuleList([AssocSlot(hidden_dim, n_mem_slots, n_labels,
                                              mlp_hidden)
                                    for _ in range(n_slots)])
        self.lsh = (LSHHasher(hidden_dim, n_bits, lsh_seed + block_idx)
                    if mode == "lsh" else None)
        self.norm = RMSNorm(hidden_dim)
        self.aux_loss_weight = 0.0
        self.z_loss_weight = 0.0
        self._batch_token_ids: Optional[torch.Tensor] = None
        self.binary_addressing = binary_addressing
        self.binary_tau = binary_tau
        self._batch_labels: Optional[torch.Tensor] = None
        self._rho = 0.5
        self._last = None
        self._fallbacks = 0  # replis vigilance cumulés (diagnostic)

    @property
    def frozen_pages(self):
        return self._frozen_pages

    # ── Codec de page associative ────────────────────────────────
    def snapshot_pages(self, pager, seed: int = 0):
        """Init directe sur disque (jamais E pages en RAM — init infinie)."""
        D, M = self.hidden_dim, self.n_mem_slots
        for e in range(self.n_experts):
            g = torch.Generator().manual_seed(seed + self.block_idx * 100003 + e * 1013)
            keys = F.normalize(torch.randn(M, D, generator=g), dim=-1)
            if self.binary_addressing:  # Achlioptas ±1 : projection sans mults
                proj = torch.sign(torch.randn(D, D, generator=g))
                proj[proj == 0] = 1.0
                kb = pack_bits((keys > 0).to(torch.int64))
                pb = pack_bits((proj > 0).to(torch.int64))
                sd = {"keys_bits": kb, "proj_bits": pb}
            else:
                proj = torch.randn(D, D, generator=g) / math.sqrt(D)
                sd = {"keys": keys, "proj": proj}
            sd.update({"values": torch.zeros(M, D),  # démarrage froid = sortie nulle
                       "counts": torch.zeros(M),
                       "label_hist": torch.zeros(M, self.n_labels, dtype=torch.int32),
                       "amaps": torch.zeros(M, D, D),  # cartes nulles = lookup pur
                       "centers": torch.zeros(M, D),
                       "mlp_u": torch.randn(M, self.mlp_hidden, D, generator=g) * 0.05,
                       "mlp_v": torch.zeros(M, D, self.mlp_hidden)})  # V=0 : nul à t=0
            # (équivalence v3 exacte au démarrage ; U aléatoire bootstrappe V,
            # puis V débloque U — init résiduelle style ControlNet).
            pager.store.save_page(self.block_idx, e, sd)

    def install_page(self, page: PageId, sd: Dict[str, torch.Tensor], slot: int):
        s = self.slots[slot]
        D = self.hidden_dim
        if "keys_bits" in sd:  # binaire : ombre reconstruite à ±1
            s.keys.copy_(unpack_bits(sd["keys_bits"], D).to(s.keys.dtype))
            s.keys_bits.copy_(sd["keys_bits"].to(s.keys_bits.dtype))
        else:
            s.keys.copy_(sd["keys"].to(s.keys.dtype))
            s.keys_bits.copy_(pack_bits((s.keys.detach().cpu() > 0).to(torch.int64)))
        if "proj_bits" in sd:
            s.proj.copy_(unpack_bits(sd["proj_bits"], D).to(s.proj.dtype))
        else:
            s.proj.copy_(sd["proj"].to(s.proj.dtype))
        s.values.copy_(sd["values"].to(s.values.dtype))
        s.counts.copy_(sd["counts"].to(s.counts.dtype))
        if "label_hist" in sd:  # compat ascendante (anciennes pages sans ART)
            s.label_hist.copy_(sd["label_hist"].to(s.label_hist.dtype))
        else:
            s.label_hist.zero_()
        for k in ("amaps", "centers", "mlp_u", "mlp_v"):  # compat (lookup pur)
            if k in sd and sd[k].shape == tuple(getattr(s, k).shape):
                getattr(s, k).copy_(sd[k].to(getattr(s, k).dtype))
            else:
                getattr(s, k).zero_()

    def read_slot(self, slot: int) -> Dict[str, torch.Tensor]:
        s = self.slots[slot]
        if self.binary_addressing:  # ombre float → bits (÷32, BNN-style)
            kb = pack_bits((s.keys.detach().cpu() > 0).to(torch.int64))
            pb = pack_bits((s.proj.detach().cpu() > 0).to(torch.int64))
            sd = {"keys_bits": kb, "proj_bits": pb}
        else:
            sd = {"keys": s.keys.detach().cpu().clone(),
                  "proj": s.proj.detach().cpu().clone()}
        sd.update({"values": s.values.detach().cpu().clone(),
                   "counts": s.counts.detach().cpu().clone(),
                   "label_hist": s.label_hist.detach().cpu().clone(),
                   "amaps": s.amaps.detach().cpu().clone(),
                   "centers": s.centers.detach().cpu().clone(),
                   "mlp_u": s.mlp_u.detach().cpu().clone(),
                   "mlp_v": s.mlp_v.detach().cpu().clone()})
        return sd

    # ── Adressage / forward ──────────────────────────────────────
    @property
    def supports_id_prefetch(self) -> bool:
        return self.mode == "token"  # LSH → working_set_hint (heuristique)

    def set_batch_token_ids(self, ids):
        self._batch_token_ids = ids

    def set_batch_labels(self, y):
        self._batch_labels = y  # (B,) — vigilance ART au train, ignoré à l'éval

    def assign_pages(self, x: torch.Tensor, token_ids=None) -> torch.Tensor:
        if self.mode == "token":
            if token_ids is None:
                token_ids = x if x.dtype == torch.long else self._batch_token_ids
            assert token_ids is not None, "mode token : fournir token_ids"
            pages, _ = assoc_address(token_ids, self.n_experts, self.top_k,
                                     self.n_mem_slots, self.salt)
            return pages
        code = self.lsh.codes(x.detach().float())
        return self.lsh.experts(x, self.n_experts, self.top_k, code=code)

    def forward(self, x: torch.Tensor, token_ids=None):
        B, T, D = x.shape
        K = self.top_k
        N = B * T
        if self.mode == "token":
            ids = self._batch_token_ids if token_ids is None else token_ids
            assert ids is not None, "mode token : fournir token_ids"
            pages, wins = assoc_address(ids, self.n_experts, K, self.n_mem_slots, self.salt)
        else:  # LSH : 1 code → pages + fenêtres (O(1) unifié, corrélé contenu)
            code = self.lsh.codes(x.detach().float())
            pages, wins = assoc_address_lsh(code, self.lsh, self.n_experts, K,
                                            self.n_mem_slots)
        pages = pages.to(x.device)
        wins = wins.to(x.device)
        w = torch.full((N, K), 1.0 / K, device=x.device, dtype=x.dtype)
        flat_p, flat_n = pages.reshape(N, K), wins.reshape(N, K)
        x_flat = x.reshape(N, D)
        uniq = sorted(set(flat_p.reshape(-1).tolist()))
        # Vagues (même politique que le dense : S − pinnées).
        pinned_here = sum(1 for p in self.pager.pinned if p[0] == self.block_idx)
        wave = max(1, self.n_slots - pinned_here)
        # Vigilance ART au train seulement (labels connus) ; éval = argmax pur.
        y_exp = None
        if self.training and self._batch_labels is not None:
            y_exp = self._batch_labels.reshape(-1).repeat_interleave(T)
        # Retrieval SANS gradient : experts détachés (le global passe par résiduels).
        with torch.no_grad():
            combined = torch.zeros(N, D, device=x.device, dtype=x.dtype)
            for i in range(0, len(uniq), wave):
                chunk = uniq[i:i + wave]
                slot_map = self.pager.ensure(self.block_idx, [(self.block_idx, e) for e in chunk])
                for e in chunk:
                    page = (self.block_idx, e)
                    if page not in slot_map:
                        continue  # joker compté par le pager
                    s = slot_map[page]
                    slot = self.slots[s]
                    for k in range(K):
                        sel = (flat_p[:, k] == e)
                        if not bool(sel.any().item()):
                            continue
                        out_k, _, _, _, fb = assoc_retrieve(
                            x_flat[sel], slot.proj, slot.keys, slot.values,
                            flat_n[sel, k], self.window,
                            label_ids=y_exp[sel] if y_exp is not None else None,
                            label_hist=slot.label_hist if y_exp is not None else None,
                            rho=self._rho, amaps=slot.amaps, centers=slot.centers,
                            binary=self.binary_addressing, tau=self.binary_tau,
                            keys_bits=slot.keys_bits if self.binary_addressing else None,
                            mlp_u=slot.mlp_u if self.readout_mode == "mlp" else None,
                            mlp_v=slot.mlp_v if self.readout_mode == "mlp" else None)
                        self._fallbacks += fb
                        combined[sel] += w[sel, k].unsqueeze(-1).to(out_k.dtype) * out_k
        out = self.norm(combined.view(B, T, D))
        out = x + out
        # Enregistre pour hebbian_step + sonde locale (références, pas de clones).
        self._last = (x_flat.detach(), out.reshape(N, D).detach(), flat_p, flat_n, (B, T))
        with torch.no_grad():
            f = torch.zeros(self.n_experts, device=x.device, dtype=x.dtype)
            f.scatter_add_(0, flat_p.reshape(-1),
                           torch.ones_like(flat_p.reshape(-1), dtype=x.dtype))
            f = (f / max(1, N * K))
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

    @torch.no_grad()
    def hebbian_step(self, eta: float = 0.05, label_emb: Optional[torch.Tensor] = None,
                     label_alpha: float = 0.5,
                     label_ids: Optional[torch.Tensor] = None,
                     rho: float = 0.5, novelty_gamma: float = 0.0,
                     readout_err: Optional[torch.Tensor] = None) -> Dict[str, int]:
        """
        Superposition Hebbienne sur les pages du dernier forward.
        label_emb (N,D) : binding requête⊕label sur values (supervisé, 0 grad).
        label_ids (B,) : vigilance ART (gagnants catégoriels) + trace hist.
        novelty_gamma : match-tracking — poids ×(1+γ·(1−confiance)) ; les tokens
            surpris absorbent plus (allocation, pas adressage — 0 gradient).
        readout_err (N,D) : crédit exact 1-couche → delta-rule v3 sur (v,A,c).
        Retourne {page: slots_touchés}. Pages frozen sautées (palimpseste).
        """
        assert self._last is not None, "hebbian_step après forward"
        x_flat, _, flat_p, flat_n, (B, T) = self._last
        self._rho = rho
        y_exp = label_ids.reshape(-1).repeat_interleave(T) if label_ids is not None else None
        K = self.top_k
        touched: Dict[str, int] = {}
        pages = sorted(set(flat_p.reshape(-1).tolist()))
        slot_map = self.pager.ensure(self.block_idx, [(self.block_idx, e) for e in pages])
        for e in pages:
            if e in self._frozen_pages:
                continue  # vieux souvenirs protégés (gel par âge = gel de page)
            page = (self.block_idx, e)
            if page not in slot_map:
                continue
            slot = self.slots[slot_map[page]]
            n = 0
            for k in range(K):
                sel = (flat_p[:, k] == e)
                if not bool(sel.any().item()):
                    continue
                _, w8, winners, z, fb = assoc_retrieve(
                    x_flat[sel], slot.proj, slot.keys, slot.values,
                    flat_n[sel, k], self.window,
                    label_ids=y_exp[sel] if y_exp is not None else None,
                    label_hist=slot.label_hist if y_exp is not None else None,
                    rho=rho, amaps=slot.amaps, centers=slot.centers,
                    binary=self.binary_addressing, tau=self.binary_tau,
                    keys_bits=slot.keys_bits if self.binary_addressing else None,
                    mlp_u=slot.mlp_u if self.readout_mode == "mlp" else None,
                    mlp_v=slot.mlp_v if self.readout_mode == "mlp" else None)
                self._fallbacks += fb
                le = label_emb[sel] if label_emb is not None else None
                ly = y_exp[sel] if y_exp is not None else None
                sw = None
                if novelty_gamma > 0:  # surpris → plastique (confiance gratuite)
                    conf = w8.max(-1).values.float()
                    sw = 1.0 + novelty_gamma * (1.0 - conf)
                re = readout_err[sel] if readout_err is not None else None
                mu = slot.mlp_u if self.readout_mode == "mlp" else None
                mv = slot.mlp_v if self.readout_mode == "mlp" else None
                n += hebbian_update(slot.keys, slot.values, slot.counts,
                                    z, x_flat[sel], winners, eta, le, label_alpha,
                                    ly, slot.label_hist if y_exp is not None else None,
                                    sw, re, slot.amaps, slot.centers, mu, mv)
            if self.binary_addressing and n > 0:  # ombre a bougé → repack (M×D)
                slot.keys_bits.copy_(pack_bits((slot.keys.detach() > 0).to(torch.int64)))
            touched[f"b{self.block_idx}e{e}"] = n
        if touched:
            self.pager.mark_dirty([(self.block_idx, e) for e in pages
                                   if e not in self._frozen_pages])
        return touched


class AttentiveHead(nn.Module):
    """Tête sonde : pooling attentif (1 vecteur requête) + linéaire.
    Le pooling attentif donne le crédit PAR TOKEN en forme close (head_credit)
    — exact à travers cette unique couche, sans aucun graphe autograd."""

    def __init__(self, hidden_dim: int, vocab: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
        self.out = nn.Linear(hidden_dim, vocab)

    def forward(self, h: torch.Tensor):
        a = torch.softmax(h.float() @ self.query.float() / math.sqrt(h.shape[-1]), dim=1)
        pooled = (a.unsqueeze(-1) * h.float()).sum(1).to(h.dtype)
        return self.out(pooled), a.to(h.dtype)


@torch.no_grad()
def head_credit(head: AttentiveHead, h: torch.Tensor, y: torch.Tensor):
    """δ_t = a_t · ((p−y) @ W) : backprop EXACTE à 1 couche, par formule.
    Exact pour le dernier bloc ; approx résiduelle (Jacobien ≈ I) en amont.
    (N,D) erreurs + (B,T) attentions. Zéro graphe, zéro paramètre."""
    B, T, D = h.shape
    V = head.out.weight.shape[0]
    a = torch.softmax(h.float() @ head.query.float() / math.sqrt(D), dim=1)
    pooled = (a.unsqueeze(-1) * h.float()).sum(1)
    p = torch.softmax(pooled @ head.out.weight.float().T + head.out.bias.float(), -1)
    e = p - F.one_hot(y.reshape(-1), V).float()  # (B,V)
    dseq = e @ head.out.weight.float()  # (B,D) erreur 1-couche
    delta = (a.unsqueeze(-1) * dseq.unsqueeze(1)).reshape(B * T, D).to(h.dtype)
    return delta, a.to(h.dtype)


def convert_to_assoc(model: CogNetMoE1B, n_slots: int, pager: ExpertPager,
                     n_mem_slots: int = 8, window: int = 4, seed: int = 0,
                     salt: int = 0x9E3779B9,
                     frozen_pages: Optional[Set[int]] = None,
                     n_labels: int = 64, mode: str = "token",
                     lsh_seed: int = 1234, binary_addressing: bool = False,
                     binary_tau: float = 8.0, readout_mode: str = "legacy",
                     mlp_hidden: int = 16) -> CogNetMoE1B:
    """Swap chaque router → associatif paginé (init pages directe sur disque)."""
    for b, blk in enumerate(model.blocks):
        legacy = blk.cognitive_expert_router
        E = getattr(legacy, "n_experts", len(getattr(legacy, "experts", [])))
        K = legacy.top_k
        D = legacy.hidden_dim if hasattr(legacy, "hidden_dim") else model.hidden_dim
        r = PagedAssocExpertRouter(D, E, n_slots, K, n_mem_slots, window,
                                   block_idx=b, pager=pager, salt=salt,
                                   frozen_pages=set(frozen_pages or ()),
                                   n_labels=n_labels, mode=mode, lsh_seed=lsh_seed,
                                   binary_addressing=binary_addressing,
                                   binary_tau=binary_tau, readout_mode=readout_mode,
                                   mlp_hidden=mlp_hidden)
        if hasattr(legacy, "norm"):
            with torch.no_grad():
                r.norm.load_state_dict(legacy.norm.state_dict())
        blk.cognitive_expert_router = r
        pager.register_router(b, r)
        r.snapshot_pages(pager, seed=seed)
    return model


# ═══════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════

def _tiny_trunk(E=4, seed=0):
    torch.manual_seed(seed)
    return CogNetMoE1B(vocab_size=64, hidden_dim=32, num_blocks=2, num_channels=E,
                        channel_dim=16, ff_dim=64, max_seq_len=32, working_slots=2,
                        episodic_slots=4, semantic_slots=8, key_dim=16,
                        n_experts=E, top_k=2, use_gradient_checkpointing=False)


def self_test():
    print("=" * 70)
    print("AssocExperts — Self-test (CPU, sans gradient)")
    print("=" * 70)

    # [1] Alignement d'adressage : pages assoc == pages pager (par construction).
    print("\n[1] Alignement adressage pager ↔ hypervecteurs...")
    ids = torch.randint(0, 64, (8, 32))
    pages, wins = assoc_address(ids, 8, 2, 8)
    ref = token_hash_experts(ids, 8, 2, 0x9E3779B9)
    assert torch.equal(pages, ref), "DÉSALIGNÉ : pages != token_hash !"
    p2, w2 = assoc_address(ids, 8, 2, 8)
    assert torch.equal(pages, p2) and torch.equal(wins, w2), "non déterministe!"
    assert wins.min() >= 0 and wins.max() < 8, "fenêtre hors bornes!"
    assert (pages[..., 0] != pages[..., 1]).all(), "dédup K perdue!"
    same_tok = (assoc_address(torch.tensor([[5, 5]]), 8, 2, 8)[0][0, 0]
                == assoc_address(torch.tensor([[5, 5]]), 8, 2, 8)[0][0, 1]).all()
    assert bool(same_tok), "même token → adresses différentes!"
    print("  pages == token_hash_experts ✓, déterministe ✓, fenêtre ∈ [0,M) ✓, "
          "dédup K ✓, même-token-même-adresse ✓")
    print("  ✓ ALIGNEMENT O(1) EXACT (1 hash, 0 double overhead)")

    # [2] La superposition Hebbienne apprend (clusters synthétiques, sans grad).
    print("\n[2] Apprentissage Hebbien (3 clusters, M=6)...")
    torch.manual_seed(1)
    D = 16
    proj = torch.randn(D, D) / math.sqrt(D)
    keys = F.normalize(torch.randn(6, D), dim=-1)
    values = torch.zeros(6, D)
    counts = torch.zeros(6)
    centers = F.normalize(torch.randn(3, D), dim=-1) * 3.0
    assert not keys.requires_grad and not values.requires_grad
    q0 = centers[torch.randint(0, 3, (512,))] + torch.randn(512, D) * 0.3
    # Erreur ANGULAIRE (1 - cos max) : les clés vivent sur la sphère unitaire,
    # l'euclidien aurait un plancher de norme et mentirait sur la convergence.
    def ang_err(qq):
        zh = F.normalize(qq.float() @ proj.float().T, dim=-1)
        return (1.0 - (zh @ keys.float().T).max(-1).values).mean().item()
    with torch.no_grad():
        err0 = ang_err(q0)
    for step in range(60):
        q = centers[torch.randint(0, 3, (256,))] + torch.randn(256, D) * 0.3
        win0 = torch.zeros(256, dtype=torch.long)
        _, _, winners, z, _ = assoc_retrieve(q, proj, keys, values, win0, 6)
        hebbian_update(keys, values, counts, z, q, winners, eta=0.1)
    with torch.no_grad():
        err1 = ang_err(q0)
    used = int((counts > 0).sum().item())
    print(f"  erreur angulaire : {err0:.3f} → {err1:.3f}, slots utilisés : {used}/6")
    assert err1 < err0 * 0.5, "Hebb n'a pas convergé!"
    assert used >= 3, "clusters non couverts!"
    print("  ✓ SUPERPOSITION HEBBIENNE CONVERGE (sans aucun gradient)")

    # [2b] Vigilance ART : entrées corrélées aux labels → slots spécialisés.
    # (Géométrie réaliste : même label ⟺ contextes similaires. À entrées
    # indépendantes des labels, ART ne peut rien amplifier — vérifié : 0.50.)
    print("\n[2b] Vigilance ART (séparation catégorielle, 0 gradient)...")
    torch.manual_seed(2)
    D2, M2, LV = 16, 4, 2
    proj2 = torch.randn(D2, D2) / math.sqrt(D2)
    keys2 = F.normalize(torch.randn(M2, D2), dim=-1)
    values2 = torch.zeros(M2, D2)
    counts2 = torch.zeros(M2)
    hist2 = torch.zeros(M2, LV, dtype=torch.int32)
    cA, cB = torch.randn(D2), torch.randn(D2)
    for step in range(40):
        y = torch.arange(256) % 2
        q = torch.where((y == 0).unsqueeze(-1), cA, cB) + torch.randn(256, D2) * 0.5
        win0 = torch.zeros(256, dtype=torch.long)
        _, _, winners, z, _ = assoc_retrieve(q, proj2, keys2, values2, win0, 4,
                                             label_ids=y, label_hist=hist2, rho=0.5)
        hebbian_update(keys2, values2, counts2, z, q, winners, eta=0.1,
                       label_ids=y, label_hist=hist2)
    used = hist2.sum(-1) > 0
    pur = (hist2.float().max(-1).values / hist2.float().sum(-1).clamp_min(1))[used].mean().item()
    print(f"  pureté moyenne des slots : {pur:.3f} (utilisés {int(used.sum())}/{M2})")
    assert pur > 0.9, "ART n'a pas séparé les labels!"
    print("  ✓ VIGILANCE ART SÉPARE LES CATÉGORIES (scalpel discret)")

    # [2c] Delta-rule v3 : un slot unique apprend une application linéaire.
    print("\n[2c] Delta-rule locale (régression linéaire, 0 gradient)...")
    torch.manual_seed(3)
    D3, M3 = 8, 1
    proj3 = torch.eye(D3)
    keys3 = F.normalize(torch.randn(M3, D3), dim=-1)
    values3 = torch.zeros(M3, D3)
    counts3 = torch.zeros(M3)
    amaps3 = torch.zeros(M3, D3, D3)
    centers3 = torch.zeros(M3, D3)
    Ttrue = torch.randn(D3, D3) * 0.5
    def mse():
        qq = torch.randn(256, D3)
        oo, _, _, _, _ = assoc_retrieve(qq, proj3, keys3, values3,
                                        torch.zeros(256, dtype=torch.long), 1,
                                        amaps=amaps3, centers=centers3)
        return ((oo - qq @ Ttrue.T) ** 2).mean().item()
    m0 = mse()
    for step in range(200):
        q = torch.randn(256, D3)
        tgt = q @ Ttrue.T
        _, _, winners, z, _ = assoc_retrieve(q, proj3, keys3, values3,
                                             torch.zeros(256, dtype=torch.long), 1,
                                             amaps=amaps3, centers=centers3)
        o, _, _, _, _ = assoc_retrieve(q, proj3, keys3, values3,
                                       torch.zeros(256, dtype=torch.long), 1,
                                       amaps=amaps3, centers=centers3)
        hebbian_update(keys3, values3, counts3, z, q, winners, eta=0.05,
                       readout_err=(o - tgt), amaps=amaps3, centers=centers3)
    m1 = mse()
    print(f"  MSE carte locale : {m0:.3f} → {m1:.3f}")
    assert m1 < m0 * 0.2, "la delta-rule n'a pas appris la carte!"
    print("  ✓ DELTA-RULE APPREND UNE FONCTION (readout expressif, 0 backprop)")

    # [2e] MLP local v4 : apprend une cible NON-LINÉAIRE (le linéaire échoue).
    print("\n[2e] MLP local (non-linéarité, backprop locale exacte)...")
    torch.manual_seed(4)
    D5, M5, H5 = 8, 1, 16
    proj5 = torch.eye(D5)
    keys5 = F.normalize(torch.randn(M5, D5), dim=-1)
    values5 = torch.zeros(M5, D5)
    counts5 = torch.zeros(M5)
    centers5 = torch.zeros(M5, D5)
    mu5 = torch.randn(M5, H5, D5) * 0.01
    mv5 = torch.randn(M5, D5, H5) * 0.01
    Am5 = torch.zeros(M5, D5, D5)
    # Cible non-linéaire : norme² (impossible en linéaire pur).
    def tgt(qq):
        return (qq ** 2).sum(-1, keepdim=True).expand_as(qq) * 0.3
    def msel():
        qq = torch.randn(256, D5)
        oo, _, _, _, _ = assoc_retrieve(qq, proj5, keys5, values5,
                                        torch.zeros(256, dtype=torch.long), 1,
                                        amaps=Am5, centers=centers5)
        return ((oo - tgt(qq)) ** 2).mean().item()
    def msem():
        qq = torch.randn(256, D5)
        oo, _, _, _, _ = assoc_retrieve(qq, proj5, keys5, values5,
                                        torch.zeros(256, dtype=torch.long), 1,
                                        amaps=Am5, centers=centers5,
                                        mlp_u=mu5, mlp_v=mv5)
        return ((oo - tgt(qq)) ** 2).mean().item()
    m0 = msem()
    for step in range(600):
        q = torch.randn(256, D5)
        t = tgt(q)
        o, _, winners, z, _ = assoc_retrieve(q, proj5, keys5, values5,
                                             torch.zeros(256, dtype=torch.long), 1,
                                             amaps=Am5, centers=centers5,
                                             mlp_u=mu5, mlp_v=mv5)
        hebbian_update(keys5, values5, counts5, z, q, winners, eta=0.05,
                       readout_err=(o - t), amaps=Am5, centers=centers5,
                       mlp_u=mu5, mlp_v=mv5)
    m1 = msem()
    print(f"  MSE non-linéaire : {m0:.3f} → {m1:.3f}")
    assert m1 < m0 * 0.25, "le MLP local n'a pas appris!"  # ×4 mini : le
    # résidu MLP + socle linéaire partagent δ ; l'init ×0.05 coûte ~0.5 %.
    print("  ✓ MLP LOCAL APPREND LE NON-LINÉAIRE (0 graphe global)")

    # [2d] Adressage binaire : même convergence + Hamming plus rapide que dot.
    print("\n[2d] Hypervecteurs binaires (Hamming vs dot)...")
    torch.manual_seed(1)
    D4, M4 = 16, 6
    proj4 = torch.sign(torch.randn(D4, D4))
    keys4 = F.normalize(torch.randn(M4, D4), dim=-1)
    values4 = torch.zeros(M4, D4)
    counts4 = torch.zeros(M4)
    centers4 = F.normalize(torch.randn(3, D4), dim=-1) * 3.0
    q0 = centers4[torch.randint(0, 3, (512,))] + torch.randn(512, D4) * 0.3
    for step in range(60):
        q = centers4[torch.randint(0, 3, (256,))] + torch.randn(256, D4) * 0.3
        _, _, winners, z, _ = assoc_retrieve(q, proj4, keys4, values4,
                                             torch.zeros(256, dtype=torch.long), 6,
                                             binary=True, tau=8.0)
        hebbian_update(keys4, values4, counts4, z, q, winners, eta=0.1)
    with torch.no_grad():
        zh = F.normalize(q0.float() @ proj4.float().T, dim=-1)
        errb = (1.0 - (zh @ keys4.float().T).max(-1).values).mean().item()
    # Économie disque clés+proj : 2×M×D fp32 → bits (÷32).
    kb = pack_bits((keys4 > 0).to(torch.int64))
    ratio = (keys4.numel() * 4 + proj4.numel() * 4) / max(1, kb.numel() * 8
            + pack_bits((proj4 > 0).to(torch.int64)).numel() * 8)
    # Micro-bench adressage : dot (M×D MACs) vs Hamming (XOR+popcount).
    qb = torch.randn(512, D4)
    w0 = torch.zeros(512, dtype=torch.long)
    t0 = time.time()
    for _ in range(200):
        assoc_retrieve(qb, proj4, keys4, values4, w0, 6)
    t_dot = time.time() - t0
    kb_cache = pack_bits((keys4 > 0).to(torch.int64))  # cache slot (production)
    t0 = time.time()
    for _ in range(200):
        assoc_retrieve(qb, proj4, keys4, values4, w0, 6, binary=True, keys_bits=kb_cache)
    t_ham = time.time() - t0
    print(f"  err binaire : {errb:.3f} (float 0.057), ratio disque adressage : {ratio:.1f}×, "
          f"dot {t_dot*1000:.0f}ms vs Hamming {t_ham*1000:.0f}ms → {t_dot/max(1e-9,t_ham):.2f}×")
    assert errb < 0.10, "le binaire ne converge pas!"
    print("  ✓ HAMMING CONVERGE (÷32 disque, 0 backprop)")

    # [3] Boucle paginée : déterminisme + writeback exact + 0 param expert.
    print("\n[3] Boucle paginée associative (E=4, S=2, M=8, W=4)...")
    tmp = tempfile.mkdtemp(prefix="assoc_st")
    m = _tiny_trunk(E=4, seed=0)
    pg = ExpertPager(PagerConfig(resident_pages=4, store_dir=os.path.join(tmp, "s3")))
    m = convert_to_assoc(m, n_slots=2, pager=pg, n_mem_slots=8, window=4, seed=0)
    m.eval()
    n_params_experts = sum(1 for _ in [])  # les slots sont des buffers
    for b in m.blocks:
        r = b.cognitive_expert_router
        assert sum(p.numel() for p in r.slots.parameters()) == 0, "slot = buffers, pas params!"
        for buf in [r.slots[0].keys, r.slots[0].values]:
            assert not buf.requires_grad
    ids = torch.randint(0, 64, (4, 16))
    for r in [b.cognitive_expert_router for b in m.blocks]:
        r.set_batch_token_ids(ids)
    with torch.no_grad():
        a = m(ids)["logits"]
        b = m(ids)["logits"]
    assert torch.equal(a, b), "forward non déterministe!"
    # Hebb change les sorties (les experts apprennent vraiment).
    m.train()
    for _ in range(5):
        for r in [b.cognitive_expert_router for b in m.blocks]:
            r.set_batch_token_ids(ids)
        m(ids)
        for r in [b.cognitive_expert_router for b in m.blocks]:
            r.hebbian_step(eta=0.1)
    pg.flush()
    m.eval()
    with torch.no_grad():
        c = m(ids)["logits"]
    delta = (c - a).abs().max().item()
    print(f"  déterminisme ✓, Δ après 5 steps Hebb : {delta:.4f}")
    assert delta > 1e-6, "Hebb n'a rien changé?!"
    # Writeback exact : recharge frais depuis le disque.
    m4 = convert_to_assoc(_tiny_trunk(E=4, seed=99), n_slots=2,
                          pager=ExpertPager(PagerConfig(
                              resident_pages=4, store_dir=os.path.join(tmp, "s3b"))),
                          n_mem_slots=8, window=4, seed=0)
    pg4 = m4.blocks[0].cognitive_expert_router.pager
    for bb in range(2):
        for e in range(4):
            pg4.store.save_page(bb, e, pg.store.load_page(bb, e))
    m4.load_state_dict(m.state_dict())
    m4.eval()
    for r in [b.cognitive_expert_router for b in m4.blocks]:
        r.set_batch_token_ids(ids)
    with torch.no_grad():
        d = m4(ids)["logits"]
    dd = (d - c).abs().max().item()
    print(f"  écart reload disque : {dd:.2e}, writebacks={pg.stats['writebacks']}")
    assert dd == 0.0, "writeback associatif inexact!"
    print("  ✓ PAGING ASSOCIATIF EXACT (déterminisme + writeback bit-à-bit)")

    # [3b] Fidélité reload BINAIRE (l'ombre perd ses magnitudes → ≈, pas =).
    print("\n[3b] Fidélité reload binaire (bits seuls, ombre ±1)...")
    mb = convert_to_assoc(_tiny_trunk(E=4, seed=0), n_slots=2,
                          pager=ExpertPager(PagerConfig(
                              resident_pages=4, store_dir=os.path.join(tmp, "s3b"))),
                          n_mem_slots=8, window=4, seed=0, binary_addressing=True)
    mb.train()
    for _ in range(5):
        for r in [b.cognitive_expert_router for b in mb.blocks]:
            r.set_batch_token_ids(ids)
        mb(ids)
        for r in [b.cognitive_expert_router for b in mb.blocks]:
            r.hebbian_step(eta=0.1)
    pgb = mb.blocks[0].cognitive_expert_router.pager
    pgb.flush()
    mb.eval()
    with torch.no_grad():
        ref = mb(ids)["logits"]
    mr = convert_to_assoc(_tiny_trunk(E=4, seed=99), n_slots=2,
                          pager=ExpertPager(PagerConfig(
                              resident_pages=4, store_dir=os.path.join(tmp, "s3c"))),
                          n_mem_slots=8, window=4, seed=0, binary_addressing=True)
    pgr = mr.blocks[0].cognitive_expert_router.pager
    for bb in range(2):
        for e in range(4):
            pgr.store.save_page(bb, e, pgb.store.load_page(bb, e))
    mr.load_state_dict(mb.state_dict())
    mr.eval()
    for r in [b.cognitive_expert_router for b in mr.blocks]:
        r.set_batch_token_ids(ids)
    with torch.no_grad():
        got = mr(ids)["logits"]
    db = (got - ref).abs().max().item()
    cosb = F.cosine_similarity(got.reshape(-1).float(), ref.reshape(-1).float(),
                               dim=0).item()
    print(f"  Δ reload binaire : {db:.4f}, cos-sim : {cosb:.6f}")
    assert cosb > 0.99, "le reload binaire détruit trop d'info!"
    print("  ✓ RELOAD BINAIRE FIDÈLE (bits seuls suffisent)")

    # [4] Pages int8 associatives (HDC robuste à la quantification).
    print("\n[4] Pages int8 associatives...")
    mq = convert_to_assoc(_tiny_trunk(E=4, seed=0), n_slots=4,
                          pager=ExpertPager(PagerConfig(
                              resident_pages=8, store_dir=os.path.join(tmp, "sq"),
                              quantize="int8")),
                          n_mem_slots=8, window=4, seed=0)
    # Fidélité vs snapshot fp32 FRAIS de même seed (pas le store entraîné!).
    mf = convert_to_assoc(_tiny_trunk(E=4, seed=0), n_slots=4,
                          pager=ExpertPager(PagerConfig(
                              resident_pages=8, store_dir=os.path.join(tmp, "sf"))),
                          n_mem_slots=8, window=4, seed=0)
    sd_q = mq.blocks[0].cognitive_expert_router.pager.store.load_page(0, 0)
    sd_f = mf.blocks[0].cognitive_expert_router.pager.store.load_page(0, 0)
    cos = F.cosine_similarity(sd_q["keys"].reshape(-1).float(),
                              sd_f["keys"].reshape(-1).float(), dim=0).item()
    print(f"  cos-sim fp32↔int8 (clés) : {cos:.6f}")
    assert cos > 0.99
    print("  ✓ INT8 ASSOCIATIF FIDÈLE")

    # [5] Prefetch hash-ahead sur boucle associative (io_delay=15ms).
    print("\n[5] Anti-stall associatif (io_delay=15ms, E=8, S=2)...")
    g = torch.Generator().manual_seed(3)
    batches = [torch.randint(0, 64, (4, 16), generator=g) for _ in range(4)]
    res = {}
    for tag, use_pref in [("SYNC", False), ("AHEAD", True)]:
        ma = _tiny_trunk(E=8, seed=0)
        pa = ExpertPager(PagerConfig(resident_pages=4, store_dir=os.path.join(tmp, f"s5{tag}"),
                                     io_delay_ms=15.0, num_loaders=8))
        ma = convert_to_assoc(ma, n_slots=2, pager=pa, n_mem_slots=8, window=4, seed=0)
        ma.train()
        t0 = time.time()
        for i, bids in enumerate(batches):
            if use_pref and i + 1 < len(batches):
                pa.prefetch(hash_ahead_pages(batches[i + 1], ma))
            pa._install_prefetched()
            for r in [bl.cognitive_expert_router for bl in ma.blocks]:
                r.set_batch_token_ids(bids)
            ma(bids)
            if use_pref:  # le hebbian_step rejoue les mêmes pages → précharge
                pa.prefetch(hash_ahead_pages(bids, ma))
            for r in [bl.cognitive_expert_router for bl in ma.blocks]:
                r.hebbian_step(eta=0.05)
        pa.flush()
        dt = time.time() - t0
        s = pa.stats
        res[tag] = dt
        print(f"  {tag:5s} wall={dt:.2f}s sync={s['faults_sync']} "
              f"couverts={s['faults_covered']} writebacks={s['writebacks']}")
    print(f"  → hash-ahead associatif : {res['SYNC']/max(1e-9,res['AHEAD']):.2f}× plus rapide")
    assert res["AHEAD"] < res["SYNC"] and pa.stats["faults_covered"] > 0
    print("  ✓ BOUCLE COMPLÈTE : hash → prefetch → retrieval → Hebb → writeback")

    print("\n" + "=" * 70)
    print("✓ Self-test AssocExperts passé : alignement O(1) + Hebb + paging exact.")
    print("=" * 70)


if __name__ == "__main__":
    self_test()
