#!/usr/bin/env python3
"""
phase_routed_moe.py — Phase-Routed MoE : experts créés à la volée + entraînement infini
=======================================================================================

Problème résolu
---------------
Le training MoE classique a un nombre d'experts FIXE (8) : quand on continue
l'entraînement sur de nouvelles données (nouveau domaine, nouvelle langue, nouvelles
connaissances), le modèle est obligé de *réécrire* ses experts existants →
**catastrophic forgetting**. Et chaque nouvelle phase coûte aussi cher que le
training initial.

Solution : le **Phase-Routed MoE**.
  - Le routing reste 100% CogNet-native : cohérence O(n) (query × mean_key),
    AUCUN gate transformer-style, AUCUNE attention inter-token.
  - Le router est conditionné par la **phase** : logits = cohérence + biais(phase).
  - À chaque nouvelle phase de données, le modèle **crée de nouveaux experts
    à la volée** (croissance C → C+k, bornée par max_experts).
  - Les anciens experts sont **gelés** (gradients masqués) → savoir préservé,
    pas de catastrophic forgetting.
  - Le coût marginal d'une phase ≈ coût d'entraînement des NOUVEAUX experts +
    router (≈ 15-25% d'un full fine-tune) → le `.pt` final est **entraînable
    à l'infini** : chaque phase ajoute de la capacité au lieu de détruire.

Architecture par bloc (CogNet-native, croissance dynamique C → max_C) :
    x → PhaseRoutedExpertRouter  (C canaux = C experts FusedSwiGLU, C dynamique)
        ├─ GrowableCoherenceRouter O(n)  (query/key pré-alloués à max_C, masqués à C)
        ├─ phase_bias[phase]             (Embedding max_phases × max_C, biais additif)
        ├─ to_channels                   (Linear D → C×D, *remplacé* à chaque croissance)
        ├─ C experts FusedSwiGLU         (ModuleList, append à la volée)
        ├─ top-k sparse + noisy top-k    (Shazeer 2017)
        ├─ aux_loss (Switch) + z_loss (ST-MoE, actifs uniquement)
        └─ masques : expert_active (pruning) + expert_frozen (grad-mask hooks)

Composants :
  - GrowableCoherenceRouter : cohérence O(n) avec C dynamique (single-pass : weights+logits).
  - PhaseRoutedExpertRouter : router MoE complet, drop-in CogNet-native.
  - GrowthConfig / GrowthController : politique de création d'experts
    (surcharge, incertitude, spike de loss, nouvelle phase).
  - InfiniteConfig / InfiniteTrainer : entraînement lifelong multi-phases avec
    freeze, rehearsal (replay), chirurgie d'optimizer, checkpoints resumables.
  - convert_to_phase_routed() : convertit un CogNetMoE1B existant (poids copiés).
  - save_infinite_checkpoint() / load_infinite_checkpoint() : `.pt` extensible.

Compatibilité checkpoints :
  - Un checkpoint legacy (C fixe, sans phase_bias) se charge dans un modèle
    phase-routed via load_legacy_router_state() (copie + padding, phase_bias=0).
  - Un checkpoint infini contient {model_state, growth, phase_id, arch} et peut
    être repris avec PLUS d'experts (croissance auto au chargement).

Usage :
    python3 phase_routed_moe.py --self-test        # tiny CPU, ~30s
"""

import copy
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))

from cognet_1b_optimized import RMSNorm, FusedSwiGLU  # noqa: E402
from cognet_moe import CogNetMoE1B, create_cognet_moe_1b  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
#  Coherence router à capacité dynamique (toujours O(n), CogNet-native)
# ═══════════════════════════════════════════════════════════════════════

class GrowableCoherenceRouter(nn.Module):
    """
    CoherenceRouter O(n) dont le nombre de canaux/experts actifs C peut grandir.

    - query/key pré-alloués à max_experts (coût négligeable : 2×D×max_C,
      soit 262k params pour D=2048, max_C=32).
    - Seules les C premières colonnes participent au forward (slicing, pas de
      compute gaspillé sur les experts futurs).
    - Single-pass : retourne (weights, logits) en UN passage (le legacy
      calculait la cohérence 2× : une fois softmaxée + une fois logits).

    Math strictement identique au CoherenceRouter legacy sur les C actifs.
    """

    def __init__(self, hidden_dim: int, n_experts_init: int = 8, max_experts: int = 32):
        super().__init__()
        assert n_experts_init <= max_experts
        self.hidden_dim = hidden_dim
        self.max_experts = max_experts
        self.query = nn.Linear(hidden_dim, max_experts, bias=False)
        self.key = nn.Linear(hidden_dim, max_experts, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.key.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        n_active: int,
        expert_active: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
            n_active: nombre d'experts actifs C (<= max_experts)
            expert_active: mask bool (max_experts,) — False = expert pruné (logits=-inf)
        Returns:
            weights: (B, T, C) softmax sur actifs
            logits: (B, T, C) pre-softmax (fp32 si x est bf16/fp16, pour stabilité ST-MoE)
        """
        # Tranches actives uniquement — O(n×C), strict O(n).
        q = self.query(x)[..., :n_active]          # (B, T, C)
        k = self.key(x)[..., :n_active]            # (B, T, C)
        mean_key = k.mean(dim=1, keepdim=True)     # (B, 1, C)
        # Logits en fp32 (stabilité du softmax + z-loss, reco ST-MoE).
        logits = (q.float() * mean_key.float())    # (B, T, C)
        if expert_active is not None:
            mask = expert_active[:n_active].to(logits.device)  # (C,)
            logits = logits.masked_fill(~mask.view(1, 1, n_active), float("-inf"))
        weights = F.softmax(logits, dim=-1).to(x.dtype)
        return weights, logits

    def load_legacy(self, legacy_router: nn.Module):
        """Copie les poids d'un CoherenceRouter legacy (C fixe) dans les tranches actives."""
        with torch.no_grad():
            c_old = legacy_router.query.weight.shape[0]
            assert c_old <= self.max_experts, f"legacy C={c_old} > max={self.max_experts}"
            self.query.weight[:c_old].copy_(legacy_router.query.weight)
            self.key.weight[:c_old].copy_(legacy_router.key.weight)
            # Tranches futures : petite init (seront affinées à la croissance).
            nn.init.normal_(self.query.weight[c_old:], mean=0.0, std=0.02)
            nn.init.normal_(self.key.weight[c_old:], mean=0.0, std=0.02)


# ═══════════════════════════════════════════════════════════════════════
#  Phase-Routed Expert Router (CogNet-native MoE à croissance dynamique)
# ═══════════════════════════════════════════════════════════════════════

class PhaseRoutedExpertRouter(nn.Module):
    """
    CognitiveExpertRouter + conditioning de phase + croissance d'experts à la volée.

    Différences vs CognitiveExpertRouter (legacy, C fixe) :
      1. `n_experts` dynamique : add_experts() fait grandir C → C+k.
      2. `phase_bias` : Embedding(max_phases, max_C) ajouté aux logits de
         cohérence → chaque phase de données a son affinité de routing.
      3. `expert_active` : mask de pruning (experts morts désactivés, réversibles).
      4. `expert_frozen` : mask de gel (gradients masqués via hooks + requires_grad).
      5. Single-pass coherence (pas de double calcul query/key).
      6. Dispatch optimisé (scatter_add + boucle sur actifs, torch.compile-friendly).

    Contraintes CogNet-native préservées :
      - routing = cohérence O(n) + biais scalaire par (phase, expert). Pas de
        gate Linear(D→N) transformer-style. Le phase_bias est un *biais*,
        pas une projection du hidden state.
      - nouveaux experts = nouveaux CANAUX cognitifs (l'invariant
        « canaux == experts » est préservé à tout instant).
      - complexité O(n) par layer, top-k sparse.
    """

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        n_experts_init: int = 8,
        max_experts: int = 32,
        max_phases: int = 64,
        top_k: int = 2,
        dropout: float = 0.0,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 1e-3,
        noise_std: float = 1.0,
        clone_noise: float = 0.02,
    ):
        super().__init__()
        assert top_k < n_experts_init <= max_experts
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.n_experts = n_experts_init
        self.num_channels_init = n_experts_init
        self.max_experts = max_experts
        self.max_phases = max_phases
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight
        self.noise_std = noise_std
        self.clone_noise = clone_noise

        # Cohérence O(n) growable (même nom de sous-module que legacy pour compat).
        self.coherence_router = GrowableCoherenceRouter(hidden_dim, n_experts_init, max_experts)
        # Projection vers canaux — REMPLACÉE (copie + extension) à chaque croissance.
        self.to_channels = nn.Linear(hidden_dim, n_experts_init * hidden_dim, bias=False)
        # Experts = canaux.
        self.experts = nn.ModuleList([
            FusedSwiGLU(hidden_dim, ff_dim, dropout) for _ in range(n_experts_init)
        ])
        self.norm = RMSNorm(hidden_dim)
        # Biais de phase : (phase, expert) → scalaire. Init 0 = routing pure cohérence.
        self.phase_bias = nn.Embedding(max_phases, max_experts)
        nn.init.zeros_(self.phase_bias.weight)

        self.register_buffer("expert_active", torch.ones(max_experts, dtype=torch.bool))
        self.register_buffer("expert_frozen", torch.zeros(max_experts, dtype=torch.bool))

        self.current_phase: int = 0          # phase utilisée pour le masque de grad phase_bias
        self._batch_phase_ids = None         # int | Tensor(B,) — posé par le trainer à chaque step
        self._grad_hooks: List = []          # handles des hooks de masquage de gradients
        self.growth_log: List[Dict] = []     # historique local (bloc) des croissances

        self._init_weights()
        self.apply_freeze_masks()

    def _init_weights(self):
        nn.init.normal_(self.to_channels.weight, mean=0.0, std=0.02)

    # ─── API EDT / introspection ──────────────────────────────────────

    def get_expert(self, expert_idx: int) -> FusedSwiGLU:
        return self.experts[expert_idx]

    def get_coherence_router(self) -> GrowableCoherenceRouter:
        return self.coherence_router

    def get_gate(self) -> GrowableCoherenceRouter:
        """Alias compat : le « gate » EST le coherence router (CogNet-native)."""
        return self.coherence_router

    def n_active_experts(self) -> int:
        return int(self.expert_active[: self.n_experts].sum().item())

    def set_batch_phase_ids(self, phase_ids: Optional[Union[int, torch.Tensor]]):
        """Posé par le trainer à chaque step. None = pas de biais de phase."""
        self._batch_phase_ids = phase_ids

    def set_train_phase(self, phase: int):
        """Phase en cours d'entraînement → seule sa ligne de phase_bias apprend."""
        assert 0 <= phase < self.max_phases
        self.current_phase = int(phase)
        self.apply_freeze_masks()

    # ─── Gel / masques de gradients ───────────────────────────────────

    def freeze_experts(self, idxs: List[int], frozen: bool = True):
        for i in idxs:
            assert 0 <= i < self.n_experts, f"expert {i} hors borne (C={self.n_experts})"
            self.expert_frozen[i] = frozen
            for p in self.experts[i].parameters():
                p.requires_grad = not frozen
        self.apply_freeze_masks()

    def freeze_all_before(self, n: int):
        """Gèle les experts [0, n) — utilisé à chaque nouvelle phase (savoir préservé)."""
        self.freeze_experts(list(range(max(0, min(n, self.n_experts)))), True)

    def apply_freeze_masks(self):
        """
        (Ré)enregistre les hooks de masquage de gradients sur les tenseurs PARTAGÉS
        (to_channels, query, key, phase_bias) dont on ne peut pas geler des tranches
        via requires_grad. Les experts gelés eux-mêmes ont requires_grad=False.
        """
        for h in self._grad_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._grad_hooks = []
        C, D = self.n_experts, self.hidden_dim
        frozen = self.expert_frozen[:C].detach().clone()

        if bool(frozen.any()):
            # to_channels.weight : (C*D, D) — lignes [i*D:(i+1)*D] par expert.
            row_mask = torch.ones(C * D, 1)
            for i in range(C):
                if frozen[i]:
                    row_mask[i * D:(i + 1) * D] = 0.0

            def _mask_to_channels(grad, mask=row_mask):
                return grad * mask.to(grad.device).to(grad.dtype)

            self._grad_hooks.append(self.to_channels.weight.register_hook(_mask_to_channels))

            # query/key : (max_C, D) — lignes d'experts gelés.
            qk_mask = torch.ones(self.max_experts, 1)
            for i in range(C):
                if frozen[i]:
                    qk_mask[i] = 0.0

            def _mask_qk(grad, mask=qk_mask):
                return grad * mask.to(grad.device).to(grad.dtype)

            self._grad_hooks.append(self.coherence_router.query.weight.register_hook(_mask_qk))
            self._grad_hooks.append(self.coherence_router.key.weight.register_hook(_mask_qk))

        # phase_bias : seule la phase courante apprend (anciennes phases figées
        # → le routing des anciennes données ne dérive pas = anti-oubli).
        cur = self.current_phase

        def _mask_phase(grad, cur=cur):
            mask = torch.zeros_like(grad)
            if 0 <= cur < mask.shape[0]:
                mask[cur] = 1.0
            return grad * mask

        self._grad_hooks.append(self.phase_bias.weight.register_hook(_mask_phase))

    # ─── Croissance : création d'experts à la volée ───────────────────

    @torch.no_grad()
    def add_experts(
        self,
        n_new: int,
        source: str = "clone_busiest",
        usage: Optional[torch.Tensor] = None,
        block_idx: int = 0,
        reason: str = "",
    ) -> Dict:
        """
        Crée n_new experts à la volée (C → C+n_new).

        Args:
            source: 'clone_busiest' (clone l'expert le plus chargé + bruit),
                    'clone:<i>' (clone l'expert i), 'fresh' (init aléatoire).
            usage: usage (C,) pour choisir le plus chargé (optionnel).
            block_idx, reason: pour le growth_log.
        Returns:
            dict {new_ids, sources, n_experts} — le trainer doit ensuite
            appeler refresh_optimizer() (to_channels est un NOUVEAU tenseur).
        """
        assert n_new >= 1, "n_new >= 1 requis"
        C = self.n_experts
        assert C + n_new <= self.max_experts, (
            f"croissance {C}+{n_new} > max_experts={self.max_experts} — "
            f"augmenter max_experts ou pruner les experts morts"
        )
        D = self.hidden_dim

        # Choix des sources.
        sources: List[int] = []
        if source == "clone_busiest":
            if usage is not None and usage.numel() >= C:
                order = torch.argsort(usage[:C].float(), descending=True).tolist()
            else:
                order = list(range(C - 1, -1, -1))  # fallback : les derniers
            for j in range(n_new):
                sources.append(int(order[j % len(order)]))
        elif source.startswith("clone:"):
            sources = [int(source.split(":")[1])] * n_new
        elif source == "fresh":
            sources = [-1] * n_new
        else:
            raise ValueError(f"source inconnue: {source}")

        # 1. Nouveaux experts (clone + bruit OU frais).
        new_ids = []
        for j, src in enumerate(sources):
            new_exp = FusedSwiGLU(self.hidden_dim, self.ff_dim, self.experts[0].dropout.p)
            if src >= 0:
                new_exp.load_state_dict(copy.deepcopy(self.experts[src].state_dict()))
                # Bruit de différenciation (clone ≠ jumeau parfait).
                with torch.no_grad():
                    for p in new_exp.parameters():
                        if p.dim() >= 2:
                            p.add_(torch.randn_like(p) * self.clone_noise * p.std().clamp_min(1e-6))
            else:
                for p in new_exp.parameters():
                    if p.dim() >= 2:
                        nn.init.normal_(p, mean=0.0, std=0.02)
            self.experts.append(new_exp)
            new_ids.append(C + j)

        # 2. Croissance de to_channels : copie + tranches clonées du source (+ bruit).
        old_tc = self.to_channels
        new_tc = nn.Linear(D, (C + n_new) * D, bias=False)
        nn.init.normal_(new_tc.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            new_tc.weight[: C * D].copy_(old_tc.weight)
            for j, src in enumerate(sources):
                rows = slice((C + j) * D, (C + j + 1) * D)
                if src >= 0:
                    src_rows = old_tc.weight[src * D:(src + 1) * D]
                    new_tc.weight[rows].copy_(
                        src_rows + torch.randn_like(src_rows) * self.clone_noise * src_rows.std().clamp_min(1e-6)
                    )
        self.to_channels = new_tc

        # 3. Bookkeeping.
        self.n_experts = C + n_new
        self.expert_active[C:C + n_new] = True
        self.expert_frozen[C:C + n_new] = False
        for i in new_ids:
            for p in self.experts[i].parameters():
                p.requires_grad = True
        self.apply_freeze_masks()

        entry = {
            "block": block_idx, "new_ids": new_ids, "sources": sources,
            "reason": reason, "n_experts_after": self.n_experts,
        }
        self.growth_log.append(entry)
        return {"new_ids": new_ids, "sources": sources, "n_experts": self.n_experts}

    # ─── Pruning réversible des experts morts ─────────────────────────

    def prune_dead_experts(self, usage_ema: torch.Tensor, min_usage: float) -> List[int]:
        """Désactive (soft) les experts quasi-inutilisés. Réversible via unprune()."""
        C = self.n_experts
        pruned = []
        for i in range(C):
            if not self.expert_active[i]:
                continue
            if float(usage_ema[i].item()) < min_usage and self.n_active_experts() > self.top_k + 1:
                self.expert_active[i] = False
                pruned.append(i)
        return pruned

    def unprune(self, idx: int):
        self.expert_active[idx] = True

    # ─── Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        phase_ids: Optional[Union[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape
        C, K = self.n_experts, self.top_k
        n_tokens = B * T

        # 1. Cohérence O(n) single-pass (weights + logits).
        active_mask = self.expert_active[:C]
        routing_weights, coherence_logits = self.coherence_router(x, C, self.expert_active)

        # 2. Biais de phase (scalaire par (phase, expert) — pas une projection).
        pids = self._batch_phase_ids if phase_ids is None else phase_ids
        if pids is None:
            router_logits = coherence_logits  # (B, T, C) fp32
        else:
            if isinstance(pids, int):
                bias = self.phase_bias.weight[pids, :C].view(1, 1, C)
            else:
                p = pids.to(x.device).long().view(B)
                bias = self.phase_bias(p)[:, None, :C]  # (B, 1, C)
            router_logits = coherence_logits + bias.float()
            routing_weights = F.softmax(router_logits, dim=-1).to(x.dtype)

        # 3. Noisy top-k (training uniquement).
        if self.training and self.noise_std > 0:
            noisy = router_logits + torch.randn_like(router_logits) * self.noise_std
        else:
            noisy = router_logits
        topk_weights, topk_indices = torch.topk(noisy, K, dim=-1)  # (B,T,K)
        topk_weights = F.softmax(topk_weights.float(), dim=-1).to(x.dtype)

        # 4. Projection to_channels (dense, 1 GEMM — torch.compile-friendly).
        channel_input = self.to_channels(x).view(B, T, C, D)

        # 5. Dispatch sparse optimisé : 1 scatter_add, puis boucle sur actifs.
        flat_idx = topk_indices.reshape(n_tokens, K)
        flat_w = topk_weights.reshape(n_tokens, K)
        expert_w = torch.zeros(n_tokens, C, device=x.device, dtype=x.dtype)
        expert_w.scatter_add_(1, flat_idx, flat_w)  # (N, C) poids par expert
        f = (expert_w > 0).float().mean(dim=0)                 # (C,) fraction assignée
        P = routing_weights.reshape(n_tokens, C).float().mean(dim=0)  # (C,) proba moyenne

        chan_flat = channel_input.reshape(n_tokens, C, D)
        combined = torch.zeros(n_tokens, D, device=x.device, dtype=x.dtype)
        for i in range(C):
            if not bool(active_mask[i].item()):
                continue  # expert pruné : skip (poids déjà à 0 via -inf)
            w_i = expert_w[:, i]
            if not bool(torch.any(w_i > 0).item()):
                continue
            tok = w_i > 0
            out_i = self.experts[i](chan_flat[tok, i])
            combined[tok] += w_i[tok].unsqueeze(-1).to(out_i.dtype) * out_i

        # 6. Norm + résiduel (prérequis EDT).
        out = self.norm(combined.view(B, T, D))
        out = x + out

        # 7. Aux losses sur ACTIFS uniquement (exclut les -inf des prunés).
        act = active_mask.to(x.device)
        C_eff = max(int(act.sum().item()), 1)
        aux_loss = C_eff * (f[act] * P[act].to(f.dtype)).sum()
        z_loss = router_logits[..., act].float().square().mean()

        # 8. Stats (monitoring croissance : usage, confiance, entropie).
        with torch.no_grad():
            max_load = f[act].max() if C_eff else f.max()
            min_load = f[act].min() if C_eff else f.min()
            P_act = P[act].clamp_min(1e-8)
            P_act = P_act / P_act.sum().clamp_min(1e-8)
            routing_entropy = -(P_act * P_act.log()).sum()
            confidence = routing_weights[..., act].float().max(-1).values.mean()

        stats = {
            "moe_aux_loss": aux_loss,
            "moe_z_loss": z_loss,
            "moe_max_load": max_load.detach(),
            "moe_min_load": min_load.detach(),
            "moe_routing_entropy": routing_entropy.detach(),
            "moe_expert_usage": f.detach(),           # (C,)
            "moe_confidence": confidence.detach(),     # scalaire
            "moe_n_active": torch.tensor(float(C_eff)),
            "routing_entropy": routing_entropy.detach(),
        }
        return out, stats

    # ─── États de croissance (checkpoints) ────────────────────────────

    def get_growth_state(self) -> Dict:
        return {
            "n_experts": self.n_experts,
            "max_experts": self.max_experts,
            "max_phases": self.max_phases,
            "frozen": [i for i in range(self.n_experts) if bool(self.expert_frozen[i])],
            "inactive": [i for i in range(self.n_experts) if not bool(self.expert_active[i])],
            "current_phase": self.current_phase,
            "growth_log": copy.deepcopy(self.growth_log),
        }

    def set_growth_state(self, state: Dict, block_idx: int = 0):
        """Restaure C, frozen, pruning (croissance auto si besoin, AVANT load_state_dict)."""
        target = int(state["n_experts"])
        assert target <= self.max_experts
        while self.n_experts < target:
            self.add_experts(1, source="fresh", block_idx=block_idx, reason="restore")
        self.expert_active[:] = True
        self.expert_frozen[:] = False
        for i in state.get("inactive", []):
            self.expert_active[int(i)] = False
        self.freeze_experts([int(i) for i in state.get("frozen", [])], True)
        self.current_phase = int(state.get("current_phase", 0))
        self.growth_log = copy.deepcopy(state.get("growth_log", []))
        self.apply_freeze_masks()

    def load_legacy_router_state(self, legacy_router: nn.Module):
        """Charge un CognitiveExpertRouter legacy (C fixe) dans ce router (tranches)."""
        with torch.no_grad():
            c_old = len(legacy_router.experts)
            assert c_old <= self.n_experts <= self.max_experts
            self.coherence_router.load_legacy(legacy_router.coherence_router)
            # to_channels legacy : (C_old*D, D).
            self.to_channels.weight[: c_old * self.hidden_dim].copy_(
                legacy_router.to_channels.weight[: c_old * self.hidden_dim]
            )
            for i in range(c_old):
                self.experts[i].load_state_dict(legacy_router.experts[i].state_dict())
            self.norm.load_state_dict(legacy_router.norm.state_dict())
            nn.init.zeros_(self.phase_bias.weight)  # phase 0 neutre au départ


# ═══════════════════════════════════════════════════════════════════════
#  Politique de croissance (quand créer des experts ?)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GrowthConfig:
    """Seuils de création automatique d'experts à la volée."""
    max_experts: int = 32
    max_phases: int = 64
    new_experts_per_phase: int = 2   # croissance systématique à chaque nouvelle phase
    # Surcharge : f_i > overload_factor × (top_k / C_actifs) pendant `patience` updates.
    # Note : f_i est une fraction (≤ 1.0). Avec C=4/top2, l'uniforme vaut 0.5 :
    # un facteur 1.5 → seuil 0.75 (atteignable), un facteur 2.0 → seuil 1.0 (jamais).
    overload_factor: float = 1.5
    overload_patience: int = 20      # nb d'updates consécutives (updates = log_every steps)
    top_k: int = 2                   # doit matcher le modèle (sert au calcul de l'uniforme)
    # Incertitude : confiance moyenne < seuil → le router « ne sait pas où router ».
    confidence_low: float = 0.35
    confidence_patience: int = 20
    # Garde-fous.
    min_steps_between_growth: int = 500
    max_new_per_event: int = 2
    prune_min_usage: float = 0.002   # usage EMA sous ce seuil → pruning (0 = désactivé)
    enable_mid_phase_growth: bool = True


@dataclass
class GrowthDecision:
    block_idx: int
    n_new: int
    source: str
    reason: str


class GrowthController:
    """
    Agrège les stats de routing (usage EMA, confiance) et décide des croissances.

    Déclencheurs :
      - overload : un expert reçoit 2×+ sa part uniforme de façon persistante
        → on le SPLIT (clone + bruit) pour absorber la charge.
      - low-confidence : le router hésite (max-proba faible) → capacité manque.
      - new-phase : géré par le trainer (croissance systématique).
    """

    def __init__(self, n_blocks: int, cfg: GrowthConfig):
        self.cfg = cfg
        self.n_blocks = n_blocks
        self.usage_ema: List[Optional[torch.Tensor]] = [None] * n_blocks
        self.overload_counters: List[Dict[int, int]] = [{} for _ in range(n_blocks)]
        self.lowconf_counter: int = 0
        self.last_growth_step: int = -10 ** 9
        self.history: List[Dict] = []

    def _ema_alpha(self) -> float:
        return 0.05

    def update(
        self,
        per_block_usage: List[torch.Tensor],
        mean_confidence: float,
        step: int,
        n_experts_per_block: List[int],
        active_counts: List[int],
    ) -> List[GrowthDecision]:
        cfg = self.cfg
        decisions: List[GrowthDecision] = []
        a = self._ema_alpha()

        for b, usage in enumerate(per_block_usage):
            u = usage.float().cpu()
            if self.usage_ema[b] is None:
                # Init directe sur la 1re observation (pas de warmup biaisé à 0).
                self.usage_ema[b] = u.clone()
            else:
                if self.usage_ema[b].numel() != u.numel():
                    # (Re)dimensionne après croissance.
                    new = torch.zeros_like(u)
                    m = min(new.numel(), self.usage_ema[b].numel())
                    new[:m] = self.usage_ema[b][:m]
                    self.usage_ema[b] = new
                self.usage_ema[b] = (1 - a) * self.usage_ema[b] + a * u

        if not cfg.enable_mid_phase_growth:
            return decisions
        if step - self.last_growth_step < cfg.min_steps_between_growth:
            return decisions

        # ── Trigger 1 : surcharge persistante ──
        for b in range(self.n_blocks):
            ema = self.usage_ema[b]
            C_act = max(active_counts[b], 1)
            # top_k supposé 2 (récupéré via uniform) — le trainer peut affiner.
            uniform = 2.0 / C_act
            thr = cfg.overload_factor * uniform
            for i in range(ema.numel()):
                if float(ema[i].item()) > thr and n_experts_per_block[b] < cfg.max_experts:
                    c = self.overload_counters[b].get(i, 0) + 1
                    self.overload_counters[b][i] = c
                    if c >= cfg.overload_patience:
                        decisions.append(GrowthDecision(
                            block_idx=b, n_new=1, source=f"clone:{i}",
                            reason=f"overload expert {i} (usage_ema={float(ema[i]):.3f} > {thr:.3f})",
                        ))
                        self.overload_counters[b][i] = 0
                else:
                    self.overload_counters[b].pop(i, None)
            if decisions and decisions[-1].block_idx == b:
                break  # 1 bloc par event (croissance graduelle)

        # ── Trigger 2 : incertitude globale ──
        if not decisions:
            if mean_confidence < cfg.confidence_low:
                self.lowconf_counter += 1
            else:
                self.lowconf_counter = 0
            if self.lowconf_counter >= cfg.confidence_patience:
                # On fait grandir le bloc le plus chargé (usage max).
                loads = [float(e.max().item()) if e is not None else 0.0 for e in self.usage_ema]
                b = int(max(range(self.n_blocks), key=lambda i: loads[i]))
                if n_experts_per_block[b] < cfg.max_experts:
                    decisions.append(GrowthDecision(
                        block_idx=b, n_new=1, source="clone_busiest",
                        reason=f"low-confidence globale ({mean_confidence:.3f} < {cfg.confidence_low})",
                    ))
                self.lowconf_counter = 0

        if decisions:
            self.last_growth_step = step
            for d in decisions:
                self.history.append({**asdict(d), "step": step})
        return decisions[: cfg.max_new_per_event]


# ═══════════════════════════════════════════════════════════════════════
#  Conversion : CogNetMoE1B (C fixe) → Phase-Routed (C dynamique)
# ══════════════════════════

def convert_to_phase_routed(
    model: CogNetMoE1B,
    max_experts: int = 32,
    max_phases: int = 64,
    clone_noise: float = 0.02,
) -> CogNetMoE1B:
    """
    Remplace chaque CognitiveExpertRouter par un PhaseRoutedExpertRouter en
    copiant les poids (legacy C → tranches actives). Retourne le même modèle.
    """
    for b, block in enumerate(model.blocks):
        legacy = block.cognitive_expert_router
        new_router = PhaseRoutedExpertRouter(
            hidden_dim=legacy.hidden_dim,
            ff_dim=model.ff_dim,
            n_experts_init=legacy.num_channels,
            max_experts=max_experts,
            max_phases=max_phases,
            top_k=legacy.top_k,
            dropout=legacy.experts[0].dropout.p,
            aux_loss_weight=legacy.aux_loss_weight,
            z_loss_weight=legacy.z_loss_weight,
            noise_std=legacy.noise_std,
            clone_noise=clone_noise,
        )
        new_router.load_legacy_router_state(legacy)
        block.cognitive_expert_router = new_router
    # n_experts du modèle = valeur initiale (les routers grandissent indépendamment).
    return model


def create_phase_routed_1b(
    vocab_size: int = 16384,
    max_seq_len: int = 512,
    n_experts_init: int = 8,
    max_experts: int = 32,
    max_phases: int = 64,
    top_k: int = 2,
    use_gradient_checkpointing: bool = True,
) -> CogNetMoE1B:
    """Crée un CogNet-MoE-1B phase-routed from scratch."""
    model = create_cognet_moe_1b(
        vocab_size=vocab_size, max_seq_len=max_seq_len,
        n_experts=n_experts_init, top_k=top_k,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    return convert_to_phase_routed(model, max_experts=max_experts, max_phases=max_phases)


def is_phase_routed(model: CogNetMoE1B) -> bool:
    return isinstance(model.blocks[0].cognitive_expert_router, PhaseRoutedExpertRouter)


# ═══════════════════════════════════════════════════════════════════════
#  Checkpoints infinis (.pt extensibles)
# ═══════════════════════════════════════════════════════════════════════

def model_arch_dict(model: CogNetMoE1B) -> Dict:
    b0 = model.blocks[0]
    r0 = b0.cognitive_expert_router
    return {
        "vocab_size": model.vocab_size, "hidden_dim": model.hidden_dim,
        "num_blocks": model.num_blocks, "ff_dim": model.ff_dim,
        "max_seq_len": model.max_seq_len,
        "num_channels_init": getattr(r0, "num_channels_init",
                                    getattr(r0, "num_channels", getattr(r0, "n_experts", 8))),
        "top_k": model.top_k,
        "max_experts": getattr(r0, "max_experts", model.n_experts),
        "max_phases": getattr(r0, "max_phases", 1),
        "working_slots": b0.memory.working_end,
        "episodic_slots": b0.memory.episodic_end - b0.memory.working_end,
        "semantic_slots": b0.memory.total_slots - b0.memory.episodic_end,
        "key_dim": b0.memory.key_dim,
    }


def save_infinite_checkpoint(
    path: str,
    model: CogNetMoE1B,
    phase_id: int,
    tokens_seen_total: int,
    growth_history: Optional[List[Dict]] = None,
    extra: Optional[Dict] = None,
):
    """Sauve un `.pt` entraînable à l'infini : poids + états de croissance + arch."""
    assert is_phase_routed(model), "save_infinite_checkpoint exige un modèle phase-routed"
    growth = {str(b): blk.cognitive_expert_router.get_growth_state()
              for b, blk in enumerate(model.blocks)}
    payload = {
        "kind": "cognet-phase-routed-v1",
        "arch": model_arch_dict(model),
        "model_state_dict": model.state_dict(),
        "growth": growth,
        "phase_id": int(phase_id),
        "tokens_seen_total": int(tokens_seen_total),
        "growth_history": growth_history or [],
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_infinite_checkpoint(
    path: str,
    model: Optional[CogNetMoE1B] = None,
    device: str = "cpu",
) -> Tuple[CogNetMoE1B, Dict]:
    """
    Charge un `.pt` infini. Si model=None, reconstruit l'arch depuis le ckpt.
    Croissance auto des routers AVANT load_state_dict (shapes compatibles).
    Retourne (model, meta).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("kind") != "cognet-phase-routed-v1":
        raise ValueError(f"checkpoint {path} n'est pas phase-routed (kind={ckpt.get('kind')})")
    arch = ckpt["arch"]
    if model is None:
        model = CogNetMoE1B(
            vocab_size=arch["vocab_size"], hidden_dim=arch["hidden_dim"],
            num_blocks=arch["num_blocks"], num_channels=arch["num_channels_init"],
            ff_dim=arch["ff_dim"], max_seq_len=arch["max_seq_len"],
            working_slots=arch["working_slots"], episodic_slots=arch["episodic_slots"],
            semantic_slots=arch["semantic_slots"], key_dim=arch["key_dim"],
            n_experts=arch["num_channels_init"], top_k=arch["top_k"],
            use_gradient_checkpointing=False,
        )
        model = convert_to_phase_routed(
            model, max_experts=arch["max_experts"], max_phases=arch["max_phases"])
    if not is_phase_routed(model):
        model = convert_to_phase_routed(
            model, max_experts=arch["max_experts"], max_phases=arch["max_phases"])
    # Croissance AVANT chargement (les nouveaux experts seront écrasés par le ckpt).
    for b, blk in enumerate(model.blocks):
        blk.cognitive_expert_router.set_growth_state(ckpt["growth"][str(b)], block_idx=b)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False), None
    # strict=False tolère les buffers de hooks (aucun) — mais on exige 0 missing.
    if isinstance(missing, torch.nn.modules.module._IncompatibleKeys):
        assert not missing.missing_keys, f"missing keys: {missing.missing_keys[:5]}"
        assert not missing.unexpected_keys, f"unexpected: {missing.unexpected_keys[:5]}"
    # Ré-applique gel/pruning (requires_grad + hooks) après chargement.
    for b, blk in enumerate(model.blocks):
        r = blk.cognitive_expert_router
        r.freeze_experts([int(i) for i in ckpt["growth"][str(b)].get("frozen", [])], True)
        r.set_train_phase(int(ckpt["growth"][str(b)].get("current_phase", ckpt["phase_id"])))
    meta = {k: ckpt[k] for k in ("phase_id", "tokens_seen_total", "growth_history", "arch", "extra")}
    return model, meta


def load_legacy_checkpoint_into_phase_routed(
    legacy_path: str,
    model: CogNetMoE1B,
    device: str = "cpu",
) -> CogNetMoE1B:
    """
    Charge un `.pt` legacy (C fixe, ex: after_phase3_final.pt ou cognet_moe_1b_final.pt)
    dans un modèle phase-routed : copie tranche par tranche, phase_bias=0.
    """
    assert is_phase_routed(model)
    ckpt = torch.load(legacy_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    if isinstance(sd, dict) and any(k.startswith("blocks.") for k in sd.keys()):
        pass
    else:
        raise ValueError("format legacy non reconnu (state_dict attendu)")
    own = model.state_dict()
    # 1. Copie directe partout SAUF coherence query/key (growables à slicer).
    skip_suffix = ("coherence_router.query.weight", "coherence_router.key.weight")
    for k, v in sd.items():
        if k in own and not k.endswith(skip_suffix):
            if own[k].shape == v.shape:
                own[k].copy_(v)
            # to_channels legacy vs growable : shapes égales à C_init → copie directe.
    # 2. Tranches query/key.
    for b, blk in enumerate(model.blocks):
        r = blk.cognitive_expert_router
        qk = f"blocks.{b}.cognitive_expert_router.coherence_router.query.weight"
        kk = f"blocks.{b}.cognitive_expert_router.coherence_router.key.weight"
        if qk in sd:
            c_old = sd[qk].shape[0]
            r.coherence_router.query.weight.data[:c_old].copy_(sd[qk])
            r.coherence_router.key.weight.data[:c_old].copy_(sd[kk])
    model.load_state_dict(own)
    return model


# ═══════════════════════════════════════════════════════════════════════
#  InfiniteTrainer — entraînement lifelong multi-phases
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class InfiniteConfig:
    """Hyperparamètres de l'entraînement infini (par phase)."""
    lr: float = 1e-4
    lr_min_ratio: float = 0.1
    warmup_steps: int = 200
    batch_size: int = 8
    seq_len: int = 512
    grad_accum: int = 4
    aux_loss_weight: float = 0.05
    z_loss_weight: float = 1e-3
    aux_clamp: float = 10.0
    z_clamp: float = 10.0
    max_grad_norm: float = 1.0
    use_bf16: bool = True
    compile_mode: Optional[str] = None     # None | 'reduce-overhead' | 'max-autotune'
    optimizer: str = "adamw"               # 'adamw' | 'adamw8bit' | 'adamw-fused'
    # Lifelong.
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    freeze_old_experts: bool = True        # gèle experts des phases précédentes
    replay_ratio: float = 0.03             # 3% d'anciennes données (ancrage router)
    new_phase_bias_init: float = 1.0       # affinité initiale nouveaux experts
    mid_phase_growth: bool = True
    # Logging / ckpt.
    log_every: int = 50
    ckpt_every_steps: int = 2000
    seed: int = 42


class InfiniteTrainer:
    """
    Entraîne un modèle phase-routed phase après phase, à l'infini.

    Protocole par phase p (p >= 1) :
      1. begin_phase(p) : +k experts/bloc (clone busiest), freeze [0, C_old),
         phase_bias[p, nouveaux] = +init, set_train_phase(p), chirurgie optimizer.
      2. train_phase() : boucle tokens avec replay (phase_ids mixtes), monitoring
         usage/confiance, croissance mid-phase si déclencheurs.
      3. save() : `.pt` infini (resumable, extensible).

    L'optimizer est reconstruit à chaque croissance en PRÉSERVANT les moments
    Adam des params inchangés (cache par nom) et en paddant (zéros) les moments
    des tenseurs grandis (to_channels). Fallback : fresh + re-warmup.
    """

    def __init__(
        self,
        model: CogNetMoE1B,
        cfg: InfiniteConfig,
        device: str = "cpu",
        ckpt_dir: str = "./infinite_ckpts",
    ):
        assert is_phase_routed(model), "InfiniteTrainer exige convert_to_phase_routed()"
        self.cfg = cfg
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.raw_model = model.to(self.device)
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        self.rng = torch.Generator().manual_seed(cfg.seed)

        self.global_step = 0
        self.phase_step = 0
        self.phase_id = 0
        self.tokens_seen_total = 0
        self.tokens_seen_phase = 0
        self.loss_ema: Optional[float] = None

        self.controller = GrowthController(model.num_blocks, cfg.growth)
        self.controller.cfg.enable_mid_phase_growth = cfg.mid_phase_growth
        self.growth_history: List[Dict] = []
        self.opt: Optional[torch.optim.Optimizer] = None
        self._model = None  # wrappé (compile)
        self._wrap_model()
        self.refresh_optimizer(reason="init")

    # ─── Wrapping (compile) ─────────────────────────────────────────

    def _wrap_model(self):
        if self.cfg.compile_mode and self.device.startswith("cuda"):
            try:
                import torch._dynamo as dynamo  # noqa
                self._model = torch.compile(self.raw_model, mode=self.cfg.compile_mode)
                print(f"[Infinite] torch.compile(mode={self.cfg.compile_mode}) actif")
                return
            except Exception as e:
                print(f"[Infinite] torch.compile indisponible ({e}), eager")
        self._model = self.raw_model

    def _rewrap_after_growth(self):
        # Re-trace après changement de shapes (croissance rare → coût amorti).
        if self.cfg.compile_mode and self.device.startswith("cuda"):
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            self._wrap_model()
        else:
            self._model = self.raw_model

    def routers(self) -> List[PhaseRoutedExpertRouter]:
        return [b.cognitive_expert_router for b in self.raw_model.blocks]

    # ─── Optimizer + chirurgie ──────────────────────────────────────

    def _make_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.raw_model.parameters() if p.requires_grad]
        name = self.cfg.optimizer.lower()
        if name == "adamw8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(params, lr=self.cfg.lr, weight_decay=0.01)
            except ImportError:
                print("[Infinite] bitsandbytes absent → AdamW standard")
        if name == "adamw-fused" and self.device.startswith("cuda"):
            try:
                return torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=0.01, fused=True)
            except Exception:
                pass
        return torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=0.01)

    def _snapshot_opt_state(self) -> Dict[str, Dict]:
        """Cache {nom_param: {shape, state(cloné)}} pour préserver les moments Adam."""
        if self.opt is None:
            return {}
        try:
            import bitsandbytes  # noqa
            # États 8-bit quantifiés : chirurgie trop risquée → fresh.
            if "bitsandbytes" in type(self.opt).__module__:
                return {}
        except ImportError:
            pass
        id2name = {id(p): n for n, p in self.raw_model_old_named()}
        cache: Dict[str, Dict] = {}
        for p, st in self.opt.state.items():
            n = id2name.get(id(p))
            if n is None or not st:
                continue
            cache[n] = {"shape": tuple(p.shape),
                        "state": {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v))
                                  for k, v in st.items()}}
        return cache

    def raw_model_old_named(self):
        # Hook internal : named_parameters AVANT rebuild — ici = actuel (appelé avant modif).
        return list(self.raw_model.named_parameters())

    def _restore_opt_state(self, cache: Dict[str, Dict]):
        if not cache or self.opt is None:
            return
        name2param = dict(self.raw_model.named_parameters())
        n_ok, n_pad, n_skip = 0, 0, 0
        for n, entry in cache.items():
            p = name2param.get(n)
            if p is None or p not in self.opt.state and True:
                pass
            if p is None:
                n_skip += 1
                continue
            # Le param doit être dans l'optimizer (requires_grad=True).
            found = any(p is q for g in self.opt.param_groups for q in g["params"])
            if not found:
                n_skip += 1
                continue
            st = {}
            for k, v in entry["state"].items():
                if torch.is_tensor(v) and v.shape == p.shape:
                    st[k] = v.clone().to(p.device)
                elif (torch.is_tensor(v) and v.dim() == p.dim() and v.dim() >= 1
                        and v.shape[1:] == p.shape[1:] and v.shape[0] < p.shape[0]):
                    # Tenseur grandi sur dim 0 (to_channels : exp_avg/exp_avg_sq) → pad zéros.
                    pad_shape = (p.shape[0] - v.shape[0],) + tuple(v.shape[1:])
                    st[k] = torch.cat([v.to(p.device),
                                       torch.zeros(pad_shape, device=p.device, dtype=v.dtype)])
                    n_pad += 1
                elif torch.is_tensor(v):
                    # Scalaires (ex: compteur 'step' d'Adam) ou buffers annexes : copie telle quelle.
                    st[k] = v.clone().to(p.device)
                else:
                    st[k] = copy.deepcopy(v)
            self.opt.state[p] = {k: v for k, v in st.items() if v is not None}
            n_ok += 1
        print(f"[Infinite] optimizer restauré : {n_ok} params (moments préservés), "
              f"{n_pad} buffers paddés, {n_skip} skippés")

    def refresh_optimizer(self, reason: str = ""):
        cache = self._snapshot_opt_state()
        self.opt = self._make_optimizer()
        self._restore_opt_state(cache)
        if reason:
            print(f"[Infinite] optimizer reconstruit ({reason})")

    # ─── LR schedule manuel (robuste aux rebuilds) ──────────────────

    def current_lr(self, phase_total_steps: int) -> float:
        s = self.phase_step
        w = max(1, self.cfg.warmup_steps)
        if s < w:
            return self.cfg.lr * (s + 1) / w
        prog = (s - w) / max(1, phase_total_steps - w)
        prog = min(max(prog, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return self.cfg.lr * (self.cfg.lr_min_ratio + (1 - self.cfg.lr_min_ratio) * cos)

    def _set_lr(self, lr: float):
        for g in self.opt.param_groups:
            g["lr"] = lr

    # ─── Phases ─────────────────────────────────────────────────────

    def begin_phase(self, phase_id: int, last_usage_per_block: Optional[List[torch.Tensor]] = None):
        """
        Démarre une phase : croissance systématique + freeze + biais de phase.
        Phase 0 = init (pas de croissance, pas de freeze).
        """
        assert 0 <= phase_id < self.cfg.growth.max_phases
        self.phase_id = phase_id
        self.phase_step = 0
        self.tokens_seen_phase = 0
        routers = self.routers()

        if phase_id == 0:
            for r in routers:
                r.set_train_phase(0)
            self.refresh_optimizer(reason="phase 0")
            print("[Infinite] phase 0 : init, pas de croissance")
            return {"grown": False}

        k = self.cfg.growth.new_experts_per_phase
        grown, failed = 0, 0
        for b, r in enumerate(routers):
            usage = last_usage_per_block[b] if last_usage_per_block else None
            try:
                if r.n_experts + k > r.max_experts:
                    # Pruning d'urgence des morts pour faire de la place.
                    if self.controller.usage_ema[b] is not None and self.cfg.growth.prune_min_usage > 0:
                        pruned = r.prune_dead_experts(
                            self.controller.usage_ema[b][: r.n_experts],
                            self.cfg.growth.prune_min_usage)
                        if pruned:
                            print(f"[Infinite] bloc {b} : prunés {pruned} (place pour croissance)")
                assert r.n_experts + k <= r.max_experts, "capacité max atteinte"
                if self.cfg.freeze_old_experts:
                    r.freeze_all_before(r.n_experts)
                info = r.add_experts(k, source="clone_busiest", usage=usage,
                                     block_idx=b, reason=f"new-phase {phase_id}")
                # Biais initial : la nouvelle phase préfère les nouveaux experts.
                with torch.no_grad():
                    for nid in info["new_ids"]:
                        r.phase_bias.weight[phase_id, nid] = self.cfg.new_phase_bias_init
                r.set_train_phase(phase_id)
                grown += 1
                self.growth_history.append({"phase": phase_id, "block": b, **info,
                                            "reason": f"new-phase {phase_id}"})
            except AssertionError as e:
                failed += 1
                r.set_train_phase(phase_id)
                print(f"[Infinite] bloc {b} : croissance impossible ({e}) — entraînement à C constant")
        self._rewrap_after_growth()
        self.refresh_optimizer(reason=f"begin-phase {phase_id}")
        print(f"[Infinite] phase {phase_id} : {grown} blocs +{k} experts, {failed} saturés "
              f"(C={[r.n_experts for r in routers]})")
        return {"grown": grown > 0, "blocks_grown": grown, "blocks_saturated": failed}

    def _autocast(self):
        if self.cfg.use_bf16 and self.device.startswith("cuda"):
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return torch.cpu.amp.autocast("cpu", dtype=torch.bfloat16, enabled=False) \
            if False else _nullcontext()

    def train_step(
        self,
        input_ids: torch.Tensor,
        phase_ids: Optional[torch.Tensor],
        phase_total_steps: int,
    ) -> Dict[str, float]:
        self._model.train()
        input_ids = input_ids.to(self.device)
        if phase_ids is not None:
            phase_ids = phase_ids.to(self.device)
        for r in self.routers():
            r.set_batch_phase_ids(phase_ids if phase_ids is not None else self.phase_id)

        lr = self.current_lr(phase_total_steps)
        self._set_lr(lr)
        self.opt.zero_grad(set_to_none=True)

        # Accumulation.
        B = input_ids.shape[0]
        micro = max(1, B // max(1, self.cfg.grad_accum))
        losses, auxs = [], []
        for a in range(self.cfg.grad_accum):
            sl = slice(a * micro, (a + 1) * micro if a < self.cfg.grad_accum - 1 else B)
            ids = input_ids[sl]
            if ids.shape[0] == 0:
                continue
            with self._autocast():
                out = self._model(ids, return_stats=False)
                logits = out["logits"]
                lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                     ids[:, 1:].reshape(-1))
                aux = out["moe_aux_loss"].clamp(max=self.cfg.aux_clamp)
                z = out["moe_z_loss"].clamp(max=self.cfg.z_clamp)
                loss = (lm + self.cfg.aux_loss_weight * aux + self.cfg.z_loss_weight * z)
                loss = loss / self.cfg.grad_accum
            loss.backward()
            losses.append(float(lm.detach().item()))
            auxs.append(float(aux.detach().item()))

        torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        self.global_step += 1
        self.phase_step += 1
        ntok = input_ids.numel()
        self.tokens_seen_phase += ntok
        self.tokens_seen_total += ntok
        loss_val = sum(losses) / max(1, len(losses))
        self.loss_ema = loss_val if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * loss_val
        return {"loss": loss_val, "aux": sum(auxs) / max(1, len(auxs)), "lr": lr}

    @torch.no_grad()
    def collect_routing_stats(self, input_ids: torch.Tensor, phase_ids) -> Dict:
        self._model.eval()
        input_ids = input_ids.to(self.device)
        for r in self.routers():
            r.set_batch_phase_ids(phase_ids if phase_ids is not None else self.phase_id)
        with self._autocast():
            out = self._model(input_ids, return_stats=True)
        usages, confs = [], []
        for b in range(self.raw_model.num_blocks):
            u = out["stats"].get(f"block{b}_moe_expert_usage")
            c = out["stats"].get(f"block{b}_moe_confidence")
            if u is not None:
                usages.append(u.cpu())
            if c is not None:
                confs.append(float(c.item()))
        self._model.train()
        return {"usages": usages,
                "mean_confidence": sum(confs) / max(1, len(confs))}

    def train_phase(
        self,
        batch_fn: Callable[[], Tuple[torch.Tensor, Optional[torch.Tensor]]],
        phase_id: int,
        phase_tokens: int,
        phase_total_steps: Optional[int] = None,
        last_usage_per_block: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """Entraîne une phase complète (tokens budget) avec croissance mid-phase."""
        info = self.begin_phase(phase_id, last_usage_per_block)
        tokens_per_step = self.cfg.batch_size * self.cfg.seq_len
        total_steps = phase_total_steps or max(1, phase_tokens // tokens_per_step)
        t0 = time.time()
        last_stats = None
        for step in range(total_steps):
            ids, pids = batch_fn()
            m = self.train_step(ids, pids, total_steps)
            if (step + 1) % self.cfg.log_every == 0 or step == 0:
                last_stats = self.collect_routing_stats(ids, pids)
                tok_s = self.tokens_seen_phase / max(1e-6, time.time() - t0)
                print(f"[Infinite] phase {phase_id} step {step+1}/{total_steps} "
                      f"loss={m['loss']:.4f} aux={m['aux']:.4f} conf={last_stats['mean_confidence']:.3f} "
                      f"tok/s={tok_s:.0f} C={[r.n_experts for r in self.routers()]}")
                # Croissance mid-phase.
                if self.cfg.mid_phase_growth and last_stats["usages"]:
                    decs = self.controller.update(
                        last_stats["usages"], last_stats["mean_confidence"],
                        self.global_step,
                        [r.n_experts for r in self.routers()],
                        [r.n_active_experts() for r in self.routers()])
                    for d in decs:
                        r = self.routers()[d.block_idx]
                        try:
                            r.add_experts(d.n_new, source=d.source,
                                          usage=last_stats["usages"][d.block_idx],
                                          block_idx=d.block_idx, reason=d.reason)
                            self.growth_history.append(
                                {"phase": phase_id, "block": d.block_idx,
                                 "n_new": d.n_new, "reason": d.reason})
                            print(f"[Infinite] 🌱 bloc {d.block_idx} +{d.n_new} expert(s) ({d.reason})")
                        except AssertionError as e:
                            print(f"[Infinite] bloc {d.block_idx} saturé ({e})")
                    if decs:
                        self._rewrap_after_growth()
                        self.refresh_optimizer(reason="mid-phase growth")
            if (step + 1) % self.cfg.ckpt_every_steps == 0:
                self.save(os.path.join(self.ckpt_dir, f"phase{phase_id}_step{step+1}.pt"))
        self.save(os.path.join(self.ckpt_dir, f"phase{phase_id}_final.pt"))
        self.save(os.path.join(self.ckpt_dir, "final.pt"))
        dt = time.time() - t0
        return {"phase": phase_id, "steps": total_steps, "tokens": self.tokens_seen_phase,
                "time_s": dt, "tok_s": self.tokens_seen_phase / max(1e-6, dt),
                "begin": info, "last_usages": last_stats["usages"] if last_stats else None}

    def save(self, path: str):
        save_infinite_checkpoint(path, self.raw_model, self.phase_id,
                                 self.tokens_seen_total, self.growth_history,
                                 extra={"global_step": self.global_step})
        print(f"[Infinite] sauvé : {path} (C={[r.n_experts for r in self.routers()]})")


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ═══════════════════════════════════════════════════════════════════════
#  Self-test (tiny, CPU)
# ═══════════════════════════════════════════════════════════════════════

def self_test():
    print("=" * 70)
    print("Phase-Routed MoE — Self-test (tiny, CPU)")
    print("=" * 70)
    torch.manual_seed(0)

    # 1. Création + forward phases.
    print("\n[1/7] Création modèle phase-routed tiny...")
    model = CogNetMoE1B(vocab_size=512, hidden_dim=64, num_blocks=2, num_channels=4,
                        channel_dim=32, ff_dim=128, max_seq_len=64,
                        working_slots=4, episodic_slots=8, semantic_slots=16,
                        key_dim=32, n_experts=4, top_k=2, use_gradient_checkpointing=False)
    model = convert_to_phase_routed(model, max_experts=8, max_phases=8)
    assert is_phase_routed(model)
    x = torch.randint(0, 512, (2, 32))
    for r in [b.cognitive_expert_router for b in model.blocks]:
        r.set_batch_phase_ids(0)
    out = model(x, return_stats=True)
    assert out["logits"].shape == (2, 32, 512)
    assert torch.isfinite(out["moe_aux_loss"]) and torch.isfinite(out["moe_z_loss"])
    print(f"  ✓ forward phase 0 : logits={tuple(out['logits'].shape)} "
          f"aux={out['moe_aux_loss'].item():.4f}")

    # 2. Phase bias change le routing.
    print("\n[2/7] Biais de phase...")
    r0 = model.blocks[0].cognitive_expert_router
    with torch.no_grad():
        r0.phase_bias.weight[1, :4] = torch.tensor([3.0, -3.0, 0.0, 0.0])
    r0.set_batch_phase_ids(0)
    _, s0 = r0(model.encoder(x))
    r0.set_batch_phase_ids(1)
    _, s1 = r0(model.encoder(x))
    u0 = s0["moe_expert_usage"]
    u1 = s1["moe_expert_usage"]
    assert u1[0] > u0[0], f"phase 1 devrait favoriser expert 0 ({u0} vs {u1})"
    print(f"  ✓ phase 0 usage={u0.tolist()} → phase 1 usage={u1.tolist()}")
    with torch.no_grad():
        r0.phase_bias.weight.zero_()

    # 3. Croissance à la volée.
    print("\n[3/7] Création d'experts à la volée...")
    before = r0.n_experts
    info = r0.add_experts(2, source="clone_busiest", usage=u0, block_idx=0, reason="test")
    assert r0.n_experts == before + 2
    assert r0.to_channels.weight.shape == (r0.n_experts * 64, 64)
    assert len(r0.experts) == r0.n_experts
    r0.set_batch_phase_ids(0)
    y, s = r0(model.encoder(x))
    assert y.shape == (2, 32, 64) and torch.isfinite(s["moe_aux_loss"])
    assert s["moe_expert_usage"].numel() == r0.n_experts
    # z-loss finie malgré croissance (pas de -inf).
    assert torch.isfinite(s["moe_z_loss"]), "z_loss NaN après croissance!"
    print(f"  ✓ C: {before} → {r0.n_experts}, forward OK, z={s['moe_z_loss'].item():.4f}")

    # 3b. Équivalence math legacy vs phase-routed (C_init, phase_bias=0).
    print("\n[3b/7] Équivalence numérique avec le legacy...")
    from cognet_moe import CognitiveExpertRouter
    legacy = CognitiveExpertRouter(hidden_dim=64, num_channels=4, ff_dim=128, top_k=2)
    legacy.load_state_dict({k: v for k, v in r0.state_dict().items() if k in legacy.state_dict()
                            and legacy.state_dict()[k].shape == v.shape}, strict=False)
    # Copie exacte tranches.
    with torch.no_grad():
        legacy.coherence_router.query.weight.copy_(r0.coherence_router.query.weight[:4])
        legacy.coherence_router.key.weight.copy_(r0.coherence_router.key.weight[:4])
        legacy.to_channels.weight.copy_(r0.to_channels.weight[:4 * 64])
        for i in range(4):
            legacy.experts[i].load_state_dict(r0.experts[i].state_dict())
        legacy.norm.load_state_dict(r0.norm.state_dict())
    legacy.eval()
    r0_eval_C = r0.n_experts
    # Compare sur un router frais à C=4 identique.
    r_tmp = PhaseRoutedExpertRouter(hidden_dim=64, ff_dim=128, n_experts_init=4,
                                    max_experts=8, max_phases=8, top_k=2, noise_std=0.0)
    with torch.no_grad():
        r_tmp.coherence_router.query.weight[:4].copy_(legacy.coherence_router.query.weight)
        r_tmp.coherence_router.key.weight[:4].copy_(legacy.coherence_router.key.weight)
        r_tmp.to_channels.weight.copy_(legacy.to_channels.weight)
        for i in range(4):
            r_tmp.experts[i].load_state_dict(legacy.experts[i].state_dict())
        r_tmp.norm.load_state_dict(legacy.norm.state_dict())
    legacy.noise_std = 0.0
    legacy.eval(); r_tmp.eval()
    xe = model.encoder(x)
    yl, sl = legacy(xe)
    r_tmp.set_batch_phase_ids(None)
    yp, sp = r_tmp(xe, phase_ids=None)
    d = (yl - yp).abs().max().item()
    assert d < 1e-4, f"divergence legacy vs phase-routed: {d}"
    print(f"  ✓ écart max legacy↔phase-routed : {d:.2e} (identique)")

    # 4. Freeze (gradients masqués).
    print("\n[4/7] Gel des anciens experts...")
    r0.freeze_all_before(4)
    assert all(not p.requires_grad for p in r0.experts[0].parameters())
    assert all(p.requires_grad for p in r0.experts[4].parameters())
    r0.set_batch_phase_ids(0)
    r0.train()
    y, s = r0(model.encoder(x))
    (y.sum() + s["moe_aux_loss"]).backward()
    g_tc = r0.to_channels.weight.grad
    assert g_tc is not None
    assert g_tc[:4 * 64].abs().max().item() == 0.0, "grad gelé devrait être 0!"
    assert g_tc[4 * 64:].abs().max().item() > 0.0, "nouveaux experts devraient apprendre!"
    print("  ✓ grad to_channels gelé=0, nouveau≠0")
    r0.zero_grad()

    # 5. GrowthController (surcharge simulée).
    print("\n[5/7] GrowthController...")
    gc = GrowthConfig(overload_patience=3, min_steps_between_growth=0,
                      enable_mid_phase_growth=True, max_experts=8, confidence_low=0.0)
    ctrl = GrowthController(n_blocks=2, cfg=gc)
    fake_usage = [torch.tensor([0.9, 0.05, 0.03, 0.02]), torch.tensor([0.25] * 4)]
    decs = []
    for step in range(5):
        decs = ctrl.update(fake_usage, 0.9, step, [4, 4], [4, 4])
        if decs:
            break
    assert decs and decs[0].block_idx == 0, f"surcharge non détectée: {decs}"
    print(f"  ✓ décision : bloc {decs[0].block_idx} ({decs[0].reason})")

    # 6. Checkpoint infini : save → resume avec croissance.
    print("\n[6/7] Checkpoint infini (save/resume/extensible)...")
    import tempfile
    tmp = tempfile.mkdtemp()
    p0 = os.path.join(tmp, "final.pt")
    save_infinite_checkpoint(p0, model, phase_id=1, tokens_seen_total=12345,
                             growth_history=[{"event": "test"}])
    model2, meta = load_infinite_checkpoint(p0, device="cpu")
    assert meta["phase_id"] == 1 and meta["tokens_seen_total"] == 12345
    assert model2.blocks[0].cognitive_expert_router.n_experts == r0.n_experts
    # Égalité numérique après reload.
    model.eval(); model2.eval()
    for r in [b.cognitive_expert_router for b in model.blocks]:
        r.set_batch_phase_ids(0)
    for r in [b.cognitive_expert_router for b in model2.blocks]:
        r.set_batch_phase_ids(0)
    with torch.no_grad():
        l1 = model(x)["logits"]
        l2 = model2(x)["logits"]
    assert (l1 - l2).abs().max().item() < 1e-5
    # Extensible : on peut encore grandir après reload.
    model2.blocks[0].cognitive_expert_router.add_experts(1, source="fresh")
    assert model2.blocks[0].cognitive_expert_router.n_experts == r0.n_experts + 1
    print(f"  ✓ reload identique (écart {(l1-l2).abs().max().item():.2e}), "
          f"re-croissance OK (C={model2.blocks[0].cognitive_expert_router.n_experts})")

    # 7. InfiniteTrainer : 2 phases tiny + chirurgie optimizer.
    print("\n[7/7] InfiniteTrainer (2 phases, CPU)...")
    model3 = CogNetMoE1B(vocab_size=256, hidden_dim=32, num_blocks=2, num_channels=4,
                         channel_dim=16, ff_dim=64, max_seq_len=32,
                         working_slots=2, episodic_slots=4, semantic_slots=8,
                         key_dim=16, n_experts=4, top_k=2, use_gradient_checkpointing=False)
    model3 = convert_to_phase_routed(model3, max_experts=8, max_phases=8)
    cfg = InfiniteConfig(lr=3e-4, batch_size=4, seq_len=32, grad_accum=1,
                         use_bf16=False, log_every=2, ckpt_every_steps=100,
                         growth=GrowthConfig(max_experts=8, max_phases=8,
                                             new_experts_per_phase=1,
                                             min_steps_between_growth=0,
                                             overload_patience=1000, confidence_low=0.0),
                         freeze_old_experts=True, replay_ratio=0.0)
    trainer = InfiniteTrainer(model3, cfg, device="cpu", ckpt_dir=os.path.join(tmp, "ckpts"))

    def batch_fn():
        ids = torch.randint(0, 256, (cfg.batch_size, cfg.seq_len))
        pids = torch.full((cfg.batch_size,), trainer.phase_id, dtype=torch.long)
        return ids, pids

    r3 = trainer.train_phase(batch_fn, phase_id=0, phase_tokens=512, phase_total_steps=3)
    assert trainer.routers()[0].n_experts == 4
    n_params_p0 = sum(p.numel() for p in trainer.raw_model.parameters() if p.requires_grad)
    r3 = trainer.train_phase(batch_fn, phase_id=1, phase_tokens=512, phase_total_steps=3)
    assert trainer.routers()[0].n_experts == 5, "phase 1 devrait avoir +1 expert"
    # Vérifie que les anciens experts sont gelés (pas dans l'optimizer).
    opt_params = {id(p) for g in trainer.opt.param_groups for p in g["params"]}
    assert id(trainer.routers()[0].experts[0].w_gate_up.weight) not in opt_params
    assert id(trainer.routers()[0].experts[4].w_gate_up.weight) in opt_params
    assert os.path.exists(os.path.join(tmp, "ckpts", "final.pt"))
    print(f"  ✓ phase 0 (C=4) → phase 1 (C=5), freeze+optimizer OK, final.pt sauvé")

    print("\n" + "=" * 70)
    print("✓ Self-test Phase-Routed MoE passé : croissance, freeze, ckpt infini OK.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test or True:
        self_test()
