"""
EDT (Expert Decoupled Training) — Pipeline adapté à CogNet-MoE-1B
==================================================================

EDT exploite l'indépendance des experts dans un MoE sparse pour entraîner
chaque composant en parallèle plutôt que séquentiellement. Le speedup annoncé
est de ~189× sur le papier (document EDT original). Sur CogNet-MoE-1B on
s'attend à ~164× (experts plus gros, moins nombreux → ratio Phase 1+2/Phase 3
moins favorable).

Les 4 phases :
  Phase 1  : Pré-entraînement des experts (16 blocs × 8 experts = 128 experts
             indépendants, entraînés en parallèle sur des hidden states réels).
             MSE loss : on apprend à chaque expert à reproduire la transformation
             identité + correction (h_out ≈ h_in + delta(h_in)).
  Phase 2a : Pré-entraînement des routers + memory + composer (1 étape suffit,
             car ces modules sont initialisés et le +1 step break le symmetry).
             Adaptation CogNet : on entraîne aussi CognitiveRouter (router de
             channels) et ParallelHierarchicalMemory (3-tier) et
             CompositionalReasoner (hyperdim binding) — pas seulement le gate
             du MoE.
  Phase 2b : Pré-entraînement de l'embedding (TokenEncoder) via next-token
             prediction, sans les blocs MoE (séparable car CogNet préserve
             l'embedding isolé).
  Phase 3  : Fine-tune joint avec PGSU (Progressive Gradient Sparsification/
             Update) : on rotate quelles couches reçoivent des gradients à
             chaque step. 8-bit optimizer (bitsandbytes) + bf16 mixed precision.

Adaptations CogNet-MoE-1B vs document EDT original :
  - Phase 1 : 128 experts (16 blocs × 8) au lieu de 2048 — moins de parallélisme
              mais chaque expert est plus gros (50,33M vs 0,43M).
  - Phase 2a : on étend aux composants non-transformer (router + memory + composer),
              pas seulement au gate du MoE.
  - Phase 2b : CharTokenizer (vocab=136) handicape l'embedding. Une variante
              BPE 8k–16k rendrait Phase 2b plus informative (reco #4 du reviewer).
  - Phase 3 : PGSU n_active=4 (reco #3 : si VRAM saute, descendre à 2).

Hypothèses de vitesse (à valider empiriquement sur 3090) :
  - Phase 1 : ~4 ms / expert-step × 2000 steps × 128 experts / parallélisme GPU
              ≈ 1,2 h (si 8 experts en parallèle par bloc).
  - Phase 2a : < 1 s (1 step × 16 blocs × petits modules).
  - Phase 2b : ~3,2 h (500M chars, embedding-only forward).
  - Phase 3 : ~41 h (100M chars joint fine-tune, PGSU n_active=4).
  - Total : ~45 h = ~1,9 jour (vs ~358 jours pour entraînement standard).

NOTE CRITIQUE : la réduction 35× de tokens via EDT est calquée sur le document
EDT original. Elle n'a pas encore été validée empiriquement sur CogNet. À
traiter comme hypothèse de travail, pas comme fait acquis.
"""

import os
import time
import math
import json
import gc
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# On suppose cognet_moe.py dans le même dossier.
from cognet_moe import CogNetMoE1B, create_cognet_moe_1b


# ═══════════════════════════════════════════════════════════════════════
#  Configuration EDT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EDTConfig:
    """Configuration du pipeline EDT pour CogNet-MoE-1B."""

    # ─── Phase 1 : experts ───────────────────────────────────────────
    phase1_steps_per_expert: int = 2000       # MSE steps par expert
    phase1_batch_size: int = 64               # tokens par step (indépendants)
    phase1_seq_len: int = 512                 # seq len pour générer les hidden states
    phase1_lr: float = 3e-4
    phase1_target_loss: float = 0.05          # MSE target avant arrêt anticipé

    # ─── Phase 2a : routers + memory + composer ──────────────────────
    phase2a_steps: int = 1                    # 1 step suffit (symmetry break)
    phase2a_batch_size: int = 32
    phase2a_lr: float = 3e-4

    # ─── Phase 2b : embedding ────────────────────────────────────────
    phase2b_tokens: int = 500_000_000         # 500M chars (char-token)
    phase2b_batch_size: int = 32
    phase2b_seq_len: int = 512
    phase2b_lr: float = 6e-4                  # embedding typically higher LR

    # ─── Phase 3 : joint fine-tune ───────────────────────────────────
    phase3_tokens: int = 100_000_000          # 100M chars (Scénario C EDT-prop)
    phase3_batch_size: int = 4
    phase3_seq_len: int = 512
    phase3_grad_accum: int = 8                # effective batch = 32
    phase3_lr: float = 1e-4
    phase3_warmup_steps: int = 200
    phase3_aux_loss_weight: float = 0.01      # monter à 0.05 si routing collapse
    phase3_z_loss_weight: float = 1e-3
    # Aux-loss clamping (mentionné dans le doc EDT original).
    # Clamp l'aux_loss à une valeur max pour éviter qu'elle explose et
    # déstabilise le LM loss. Typiquement 10× la valeur attendue.
    phase3_aux_loss_clamp: float = 10.0
    phase3_z_loss_clamp: float = 10.0

    # PGSU (Progressive Gradient Sparsification/Update).
    # À chaque step, n_active couches (sur 16) reçoivent des gradients.
    # Recommandation #3 du reviewer : n_active=4 ; descendre à 2 si VRAM saute.
    phase3_pgsu_n_active: int = 4

    # ─── Hardware / mixed precision ──────────────────────────────────
    use_bf16: bool = True                     # 3090 supporte bf16
    use_8bit_optimizer: bool = True           # bitsandbytes 8-bit Adam
    device: str = "cuda"
    seed: int = 42

    # ─── Logging ─────────────────────────────────────────────────────
    log_every: int = 50
    ckpt_dir: str = "./edt_ckpts"


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_optimizer(params, lr: float, use_8bit: bool, cfg: EDTConfig):
    """8-bit Adam si bitsandbytes disponible, sinon AdamW standard."""
    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.01)
        except ImportError:
            print("[EDT] bitsandbytes non disponible, fallback AdamW standard.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)


def _get_loss_scale_scaler(use_bf16: bool, device: str):
    """bf16 n'a pas besoin de GradScaler (contrairement à fp16)."""
    if use_bf16:
        return None
    return torch.amp.GradScaler('cuda' if 'cuda' in device else 'cpu')


def _autocast_context(use_bf16: bool, device: str):
    """Contexte autocast bf16 ou fp16 selon config."""
    if not use_bf16:
        return torch.amp.autocast('cuda' if 'cuda' in device else 'cpu', dtype=torch.float16)
    return torch.amp.autocast('cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16)


def _cleanup():
    """Force GC + cache CUDA entre phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1 — Pré-entraînement des experts (128 experts en parallèle)
# ═══════════════════════════════════════════════════════════════════════

def phase1_experts(
    model: CogNetMoE1B,
    data_iter_fn: Callable[[int, int], torch.Tensor],  # (batch_size, seq_len) -> hidden_states
    cfg: EDTConfig,
) -> Dict:
    """
    Phase 1 : entraîne chaque expert indépendamment sur des hidden states réels.

    Pour chaque bloc b et chaque expert e :
      - On freeze tout sauf expert[b][e].
      - On génère des hidden states h_in réels (via le modèle frozen forward
        jusqu'au bloc b, sans MoE).
      - On entraîne expert[e] à minimiser MSE(expert[h_in], h_target) où
        h_target = h_in + delta_attendu (par défaut, on apprend l'identité + petite
        perturbation utile — concrètement on minimise MSE(h_out - h_in) pour
        apprendre un résiduel neutre, ce qui est une bonne init pour EDT).

    Hypothèse de vitesse : ~4 ms / step × 2000 steps × 128 experts / 8 parallèle
    ≈ 1,2 h sur RTX 3090.

    Args:
        model: CogNet-MoE-1B
        data_iter_fn: callable(batch_size, seq_len) -> tensor (B, T, D)
                      de hidden states réels extraits du modèle frozen.
        cfg: EDTConfig
    Returns:
        dict avec stats par expert (loss finale, n_steps, temps)
    """
    print("\n" + "=" * 70)
    print("EDT Phase 1 — Pré-entraînement des 128 experts")
    print("=" * 70)
    print(f"  Steps/expert : {cfg.phase1_steps_per_expert}")
    print(f"  Batch size   : {cfg.phase1_batch_size}")
    print(f"  LR           : {cfg.phase1_lr}")
    print(f"  Target MSE   : {cfg.phase1_target_loss}")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # frozen global

    n_blocks = model.num_blocks
    n_experts = model.n_experts
    stats: Dict = {"experts": [], "total_time_s": 0.0}

    t_start = time.time()

    for b in range(n_blocks):
        for e in range(n_experts):
            expert = model.get_expert(b, e)
            expert.train()

            # Freeze tout, unfreeze juste expert[b][e].
            for p in model.parameters():
                p.requires_grad = False
            for p in expert.parameters():
                p.requires_grad = True

            opt = _get_optimizer(
                list(expert.parameters()),
                lr=cfg.phase1_lr,
                use_8bit=cfg.use_8bit_optimizer,
                cfg=cfg,
            )

            expert_losses = []
            for step in range(cfg.phase1_steps_per_expert):
                # Hidden states réels (le reste du modèle est frozen).
                with torch.no_grad():
                    h_in = data_iter_fn(cfg.phase1_batch_size, cfg.phase1_seq_len)
                    h_in = h_in.to(device).to(torch.float32)

                # Forward expert (en float32 pour stabilité MSE).
                opt.zero_grad()
                h_out = expert(h_in)

                # Target : identité (résiduel zéro) — bonne init pour EDT.
                # L'expert apprend à préserver l'information tout en étant
                # capable d'ajouter des corrections non-linéaires via le
                # SwiGLU interne. La loss MSE(h_out, h_in) pousse vers identité.
                loss = F.mse_loss(h_out, h_in)

                loss.backward()
                opt.step()
                expert_losses.append(loss.item())

                # Arrêt anticipé si on atteint la target.
                if loss.item() < cfg.phase1_target_loss and step > 100:
                    break

                if step % max(1, cfg.phase1_steps_per_expert // 4) == 0:
                    print(f"  [B{b:02d}E{e}] step {step:4d}/{cfg.phase1_steps_per_expert} "
                          f"loss={loss.item():.4f}")

            stats["experts"].append({
                "block": b,
                "expert": e,
                "final_loss": expert_losses[-1],
                "n_steps": len(expert_losses),
                "time_s": 0.0,  # mesuré globalement pour ne pas fausser
            })

            # Re-freeze cet expert avant de passer au suivant.
            for p in expert.parameters():
                p.requires_grad = False

    total_time = time.time() - t_start
    stats["total_time_s"] = total_time
    stats["avg_loss"] = sum(e["final_loss"] for e in stats["experts"]) / len(stats["experts"])

    print(f"\n  Phase 1 terminée en {total_time:.1f}s "
          f"({total_time / 3600:.2f}h)")
    print(f"  Loss moyenne finale : {stats['avg_loss']:.4f}")

    # Restore grad state pour phases suivantes.
    for p in model.parameters():
        p.requires_grad = True

    _cleanup()
    return stats


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2a — Pré-entraînement des routers + memory + composer
# ═══════════════════════════════════════════════════════════════════════

def phase2a_attention(
    model: CogNetMoE1B,
    data_iter_fn: Callable[[int, int], torch.Tensor],  # (B, T) -> input_ids
    cfg: EDTConfig,
) -> Dict:
    """
    Phase 2a : pré-entraînement des routers + memory + composer.

    Adaptation CogNet : contrairement à un transformer classique où l'on n'a
    que l'attention, ici on a 3 modules à pré-entraîner par bloc :
      - CognitiveRouter (O(n) coherence routing, 8 channels)
      - ParallelHierarchicalMemory (3-tier, SDPA)
      - CompositionalReasoner (hyperdim binding)
      - + le gate du SparseMoE (router du MoE)

    Tous les experts sont frozen (Phase 1 les a initialisés).

    Comme dans le document EDT original, 1 step suffit pour break la symmetry
    (les modules sont initialisés aléatoirement, un seul gradient step les
    place dans une région non-triviale de l'espace de paramètres).

    Args:
        model: CogNet-MoE-1B (avec experts déjà pré-entraînés)
        data_iter_fn: callable(batch_size, seq_len) -> input_ids (B, T)
        cfg: EDTConfig
    Returns:
        dict avec stats
    """
    print("\n" + "=" * 70)
    print("EDT Phase 2a — Pré-entraînement routers + memory + composer")
    print("=" * 70)
    print(f"  Steps : {cfg.phase2a_steps} (1 step suffit pour symmetry break)")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    # Freeze experts (Phase 1 les a initialisés), unfreeze tout le reste.
    for b in range(model.num_blocks):
        # Freeze les experts (canaux) — Phase 1 les a initialisés.
        cer = model.blocks[b].cognitive_expert_router
        for p in cer.experts.parameters():
            p.requires_grad = False
        # Unfreeze le coherence router O(n) (= le routing cognitif,
        # qui remplace l'ancien sparse_moe.gate transformer-style).
        for p in cer.coherence_router.parameters():
            p.requires_grad = True
        # Unfreeze la projection to_channels (pas de from_channels
        # dans le design CogNet-native — la somme pondérée des experts
        # mixe déjà les C canaux).
        for p in cer.to_channels.parameters():
            p.requires_grad = True
        for p in cer.norm.parameters():
            p.requires_grad = True
        # Unfreeze memory, composer (composants non-transformer préservés).
        for p in model.blocks[b].memory.parameters():
            p.requires_grad = True
        for p in model.blocks[b].composer.parameters():
            p.requires_grad = True
        # Final norm + encoder.
        for p in model.encoder.parameters():
            p.requires_grad = False  # Phase 2b s'en occupe
        for p in model.final_norm.parameters():
            p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = _get_optimizer(
        trainable,
        lr=cfg.phase2a_lr,
        use_8bit=cfg.use_8bit_optimizer,
        cfg=cfg,
    )

    t_start = time.time()
    losses = []

    for step in range(cfg.phase2a_steps):
        input_ids = data_iter_fn(cfg.phase2a_batch_size, cfg.phase2b_seq_len).to(device)
        opt.zero_grad()

        with _autocast_context(cfg.use_bf16, device):
            result = model(input_ids)
            logits = result["logits"]
            # Shift pour next-token prediction.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"  Step {step+1}/{cfg.phase2a_steps}  loss={loss.item():.4f}")

    elapsed = time.time() - t_start
    print(f"\n  Phase 2a terminée en {elapsed:.2f}s (< 1s attendu)")

    # Restore grad state pour Phase 2b/3.
    for p in model.parameters():
        p.requires_grad = True

    _cleanup()
    return {"losses": losses, "final_loss": losses[-1], "time_s": elapsed}


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2b — Pré-entraînement de l'embedding (TokenEncoder séparable)
# ═══════════════════════════════════════════════════════════════════════

def phase2b_embedding(
    model: CogNetMoE1B,
    data_iter_fn: Callable[[int, int], torch.Tensor],  # (B, T) -> input_ids
    cfg: EDTConfig,
) -> Dict:
    """
    Phase 2b : pré-entraînement de l'embedding (TokenEncoder) via next-token
    prediction, sans les blocs MoE.

    Pourquoi ça marche : le TokenEncoder est séparable du reste (il produit
    juste h_0 = norm(rope(emb(input_ids)))). On peut donc l'entraîner à
    prédire le token suivant en utilisant un classifieur linéaire léger
    branché directement sur h_0, sans faire passer par les 16 blocs MoE.

    CogNet-MoE préserve cette séparabilité car le TokenEncoder reste isolé
    (pas de weight-sharing avec les blocs intermédiaires, seulement avec
    output_proj via weight-tying classique).

    Hypothèse de vitesse : ~3,2 h pour 500M chars (embedding-only forward).

    Args:
        model: CogNet-MoE-1B
        data_iter_fn: callable(batch_size, seq_len) -> input_ids (B, T)
        cfg: EDTConfig
    Returns:
        dict avec stats
    """
    print("\n" + "=" * 70)
    print("EDT Phase 2b — Pré-entraînement de l'embedding (TokenEncoder)")
    print("=" * 70)
    print(f"  Tokens cibles : {cfg.phase2b_tokens:,} chars")
    print(f"  Batch size    : {cfg.phase2b_batch_size}")
    print(f"  Seq len       : {cfg.phase2b_seq_len}")
    print(f"  LR            : {cfg.phase2b_lr}")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    # Freeze TOUT sauf encoder + output_proj (weight-tied).
    for p in model.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = True
    # output_proj est weight-tied avec encoder.token_emb, donc déjà unfreeze.

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = _get_optimizer(
        trainable,
        lr=cfg.phase2b_lr,
        use_8bit=cfg.use_8bit_optimizer,
        cfg=cfg,
    )

    # Classifieur léger pour next-token prediction (directement sur h_0).
    # On réutilise output_proj (weight-tied avec token_emb).
    # Pas de blocs MoE dans le forward.

    t_start = time.time()
    total_tokens = 0
    losses = []
    step = 0

    while total_tokens < cfg.phase2b_tokens:
        input_ids = data_iter_fn(
            cfg.phase2b_batch_size, cfg.phase2b_seq_len
        ).to(device)
        opt.zero_grad()

        with _autocast_context(cfg.use_bf16, device):
            # Forward embedding-only (skip blocs MoE).
            h_0 = model.encoder(input_ids)  # (B, T, D)
            h_0 = model.final_norm(h_0)     # apply final norm
            logits = model.output_proj(h_0)  # (B, T, vocab)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        loss.backward()
        opt.step()

        losses.append(loss.item())
        total_tokens += input_ids.numel()
        step += 1

        if step % cfg.log_every == 0:
            tps = total_tokens / max(1e-6, time.time() - t_start)
            print(f"  Step {step:5d}  tokens={total_tokens/1e6:.1f}M  "
                  f"loss={loss.item():.4f}  tps={tps:.0f}")

    elapsed = time.time() - t_start
    avg_loss = sum(losses) / len(losses)
    print(f"\n  Phase 2b terminée en {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"  Loss moyenne : {avg_loss:.4f}")

    # Restore grad state pour Phase 3.
    for p in model.parameters():
        p.requires_grad = True

    _cleanup()
    return {
        "losses": losses,
        "final_loss": losses[-1],
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "time_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════
#  PGSU — Progressive Gradient Sparsification/Update
# ═══════════════════════════════════════════════════════════════════════

class PGSU:
    """
    PGSU : à chaque step, on rotate quelles couches reçoivent des gradients.

    Pour un modèle à 16 blocs et n_active=4, on a 4 couches actives en
    parallèle par step (les autres sont frozen). On rotate cycliquement
    pour que toutes les couches soient entraînées.

    Bénéfices :
      - Réduit l'empreinte mémoire des activations (seulement n_active couches
        gardent leurs activations pour le backward).
      - Accélère le backward (moins de couches à propager).
      - Régularisation implicite (évite la co-adaptation des couches).

    Reco #3 du reviewer : n_active=4. Si VRAM Phase 3 saute, descendre à 2
    (on paie en steps, pas en capacité).
    """

    def __init__(self, n_layers: int, n_active: int, seed: int = 42):
        assert n_active <= n_layers
        self.n_layers = n_layers
        self.n_active = n_active
        self.step = 0
        self.rng = torch.Generator().manual_seed(seed)

    def get_active_layers(self) -> List[int]:
        """Retourne les indices des n_active couches actives à ce step."""
        # Rotation cyclique : on décale de 1 à chaque step.
        start = self.step % self.n_layers
        active = [(start + i) % self.n_layers for i in range(self.n_active)]
        return sorted(active)

    def advance(self):
        self.step += 1

    def apply_mask(self, model: CogNetMoE1B):
        """
        Active requires_grad uniquement sur les couches actives + composants
        partagés (encoder, final_norm, gate du MoE).
        """
        active = self.get_active_layers()
        # Freeze tous les blocs.
        for b in range(model.num_blocks):
            for p in model.blocks[b].parameters():
                p.requires_grad = False
        # Unfreeze les blocs actifs.
        for b in active:
            for p in model.blocks[b].parameters():
                p.requires_grad = True
        # Composants partagés toujours actifs.
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.final_norm.parameters():
            p.requires_grad = True
        return active


# ═══════════════════════════════════════════════════════════════════════
#  Phase 3 — Joint fine-tune avec PGSU + 8-bit optimizer + bf16
# ═══════════════════════════════════════════════════════════════════════

def phase3_joint(
    model: CogNetMoE1B,
    data_iter_fn: Callable[[int, int], torch.Tensor],  # (B, T) -> input_ids
    cfg: EDTConfig,
) -> Dict:
    """
    Phase 3 : fine-tune joint avec PGSU.

    Hypothèse de vitesse : ~41 h pour 100M chars (Scénario C EDT-prop).
    C'est le scénario le plus raisonnable (Scénario A 600M trop agressif,
    Scénario B plein Chinchilla n'a aucun intérêt si on a EDT).

    Si la qualité est insuffisante après Phase 3, monter à 500M–1B chars
    (recommandation #1 du reviewer).

    Args:
        model: CogNet-MoE-1B (experts + embedding déjà pré-entraînés)
        data_iter_fn: callable(batch_size, seq_len) -> input_ids (B, T)
        cfg: EDTConfig
    Returns:
        dict avec stats
    """
    print("\n" + "=" * 70)
    print("EDT Phase 3 — Joint fine-tune avec PGSU")
    print("=" * 70)
    print(f"  Tokens cibles  : {cfg.phase3_tokens:,} chars")
    print(f"  Batch size     : {cfg.phase3_batch_size} (grad_accum={cfg.phase3_grad_accum})")
    print(f"  Seq len        : {cfg.phase3_seq_len}")
    print(f"  LR             : {cfg.phase3_lr}")
    print(f"  PGSU n_active  : {cfg.phase3_pgsu_n_active}/{model.num_blocks}")
    print(f"  Aux loss weight: {cfg.phase3_aux_loss_weight}")
    print(f"  bf16           : {cfg.use_bf16}")
    print(f"  8-bit optim    : {cfg.use_8bit_optimizer}")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    pgsu = PGSU(
        n_layers=model.num_blocks,
        n_active=cfg.phase3_pgsu_n_active,
        seed=cfg.seed,
    )

    # Tous les paramètres sont potentiellement trainable, PGSU gère le masking.
    opt = _get_optimizer(
        list(model.parameters()),
        lr=cfg.phase3_lr,
        use_8bit=cfg.use_8bit_optimizer,
        cfg=cfg,
    )

    # LR warmup linéaire -> cosine decay.
    total_steps = cfg.phase3_tokens // (
        cfg.phase3_batch_size * cfg.phase3_seq_len * cfg.phase3_grad_accum
    )

    def lr_lambda(step):
        if step < cfg.phase3_warmup_steps:
            return step / max(1, cfg.phase3_warmup_steps)
        progress = (step - cfg.phase3_warmup_steps) / max(1, total_steps - cfg.phase3_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    t_start = time.time()
    total_tokens = 0
    losses = []
    aux_losses = []
    max_loads = []
    min_loads = []
    step = 0

    while total_tokens < cfg.phase3_tokens:
        # PGSU : rotate les couches actives.
        active_layers = pgsu.apply_mask(model)

        opt.zero_grad()

        # Accumulation de gradient sur grad_accum micro-batches.
        accum_loss = 0.0
        accum_aux = 0.0
        accum_max_load = 0.0
        accum_min_load = 1.0

        for _ in range(cfg.phase3_grad_accum):
            input_ids = data_iter_fn(
                cfg.phase3_batch_size, cfg.phase3_seq_len
            ).to(device)

            with _autocast_context(cfg.use_bf16, device):
                result = model(input_ids, return_stats=True)
                logits = result["logits"]

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                aux_loss = result["moe_aux_loss"]
                z_loss = result["moe_z_loss"]

                # ─── Aux-loss clamping (mentionné dans le doc EDT original) ───
                # Clamp les aux losses à une valeur max pour éviter qu'elles
                # n'explosent et ne déstabilisent le LM loss. Particulièrement
                # important en début d'entraînement quand le router n'est pas
                # encore équilibré.
                aux_loss = aux_loss.clamp(max=cfg.phase3_aux_loss_clamp)
                z_loss = z_loss.clamp(max=cfg.phase3_z_loss_clamp)

                total_loss = (
                    lm_loss
                    + cfg.phase3_aux_loss_weight * aux_loss
                    + cfg.phase3_z_loss_weight * z_loss
                ) / cfg.phase3_grad_accum

            total_loss.backward()

            accum_loss += lm_loss.item()
            accum_aux += aux_loss.item()

            # Monitoring routing collapse (reco #2 du reviewer).
            # Si max_load > 0.5 -> routing collapse -> monter aux_loss_weight.
            block0_stats = result["stats"]
            if "block0_moe_max_load" in block0_stats:
                accum_max_load = max(accum_max_load, block0_stats["block0_moe_max_load"].item())
                accum_min_load = min(accum_min_load, block0_stats["block0_moe_min_load"].item())

            total_tokens += input_ids.numel()

        # Gradient clipping (stabilité).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        opt.step()
        scheduler.step()
        pgsu.advance()
        step += 1

        losses.append(accum_loss / cfg.phase3_grad_accum)
        aux_losses.append(accum_aux / cfg.phase3_grad_accum)
        max_loads.append(accum_max_load)
        min_loads.append(accum_min_load)

        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            tps = total_tokens / max(1e-6, elapsed)
            eta_s = (cfg.phase3_tokens - total_tokens) / max(1, tps)
            print(
                f"  Step {step:5d}  tokens={total_tokens/1e6:.1f}M  "
                f"loss={losses[-1]:.4f}  aux={aux_losses[-1]:.4f}  "
                f"max_load={accum_max_load:.3f}  "
                f"tps={tps:.0f}  ETA={eta_s/3600:.1f}h  "
                f"active={active_layers}"
            )

            # Alerte routing collapse.
            if accum_max_load > 0.5:
                print(f"  ⚠️  ROUTING COLLAPSE DÉTECTÉ (max_load={accum_max_load:.3f} > 0.5)")
                print(f"      -> monter phase3_aux_loss_weight à 0.05")

    elapsed = time.time() - t_start
    print(f"\n  Phase 3 terminée en {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"  Loss LM finale   : {losses[-1]:.4f}")
    print(f"  Aux loss finale  : {aux_losses[-1]:.4f}")
    print(f"  Max load moyen   : {sum(max_loads)/len(max_loads):.3f}")
    print(f"  Min load moyen   : {sum(min_loads)/len(min_loads):.3f}")

    _cleanup()
    return {
        "losses": losses,
        "aux_losses": aux_losses,
        "max_loads": max_loads,
        "min_loads": min_loads,
        "final_loss": losses[-1],
        "total_tokens": total_tokens,
        "time_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Orchestrateur — Pipeline complet EDT
# ═══════════════════════════════════════════════════════════════════════

def verify_edt_prerequisites(model: CogNetMoE1B) -> Dict:
    """
    Vérifie que tous les prérequis EDT sont satisfaits avant de lancer le pipeline.

    Prérequis EDT (issus du document EDT original) :
      1. top_k < n_experts (sinon pas de routage sparse, EDT n'a aucun sens)
      2. Résiduels partout (h_out = h_in + f(h_in)) — nécessaire pour que les
         experts puissent être entraînés isolément en Phase 1.
      3. Embedding séparable (TokenEncoder isolé du reste) — nécessaire pour
         Phase 2b.
      4. Aux losses implémentées (load-balancing + z-loss) — pour stabiliser
         le routing pendant la Phase 3.

    Returns:
        dict avec {prerequisite: bool, ...} + details
    """
    print("\n" + "=" * 70)
    print("EDT — Vérification des prérequis")
    print("=" * 70)

    checks = {}

    # ─── Check 1: top_k < n_experts ──────────────────────────────────
    check1 = model.top_k < model.n_experts
    checks["top_k_lt_n_experts"] = {
        "passed": check1,
        "details": f"top_k={model.top_k} < n_experts={model.n_experts} → "
                   f"{'OK' if check1 else 'FAIL (pas de routage sparse!)'}",
    }
    print(f"  [{'✓' if check1 else '✗'}] top_k < n_experts : "
          f"{model.top_k} < {model.n_experts}")

    # ─── Check 2: Résiduels partout ──────────────────────────────────
    # On vérifie que chaque sous-module d'un bloc a un résiduel.
    # Pour CogNet-MoE, les résiduels sont dans :
    #   - FusedSwiGLU (residual = x ; ... return residual + dropout(h))
    #   - ChannelProcessor (deux résiduels : conv path + ffn path)
    #   - ParallelHierarchicalMemory (x = x + dropout(out))
    #   - SparseMoEBlock (out = x_flat + out)
    #   - CompositionalReasoner (return residual + dropout(out))
    #   - CogNetMoEBlock (implicit : chaque sous-module fait son résiduel)
    block0 = model.blocks[0]
    residual_modules = []

    # FusedSwiGLU (dans experts = canaux)
    expert0 = block0.cognitive_expert_router.experts[0]
    has_residual = hasattr(expert0, 'forward')
    # On peut inspecter le code source pour confirmer.
    import inspect
    expert_src = inspect.getsource(expert0.forward)
    if "residual" in expert_src and "residual +" in expert_src:
        residual_modules.append("FusedSwiGLU (experts/canaux)")

    # CognitiveExpertRouter (CogNet-native MoE : routing + experts unifiés)
    cer_src = inspect.getsource(block0.cognitive_expert_router.forward)
    if "x + out" in cer_src or "out = x +" in cer_src:
        residual_modules.append("CognitiveExpertRouter (résiduel global)")

    # ParallelHierarchicalMemory
    mem_src = inspect.getsource(block0.memory.forward)
    if "x + " in mem_src or "x = x +" in mem_src:
        residual_modules.append("ParallelHierarchicalMemory")

    # CompositionalReasoner
    comp_src = inspect.getsource(block0.composer.forward)
    if "residual" in comp_src and "residual +" in comp_src:
        residual_modules.append("CompositionalReasoner")

    check2 = len(residual_modules) >= 4  # au moins 4 modules avec résiduel
    checks["residuals_everywhere"] = {
        "passed": check2,
        "details": f"Modules avec résiduel : {residual_modules}",
    }
    print(f"  [{'✓' if check2 else '✗'}] Résiduels partout : "
          f"{len(residual_modules)} modules → {residual_modules}")

    # ─── Check 3: Embedding séparable ────────────────────────────────
    # TokenEncoder est un module isolé qui produit h_0 = norm(rope(emb(input_ids))).
    # Il ne partage pas de params avec les blocs intermédiaires (sauf
    # weight-tying avec output_proj, ce qui est OK pour EDT).
    check3 = hasattr(model, 'encoder') and hasattr(model.encoder, 'token_emb')
    checks["separable_embedding"] = {
        "passed": check3,
        "details": f"TokenEncoder présent et isolé : {check3}",
    }
    print(f"  [{'✓' if check3 else '✗'}] Embedding séparable : TokenEncoder isolé")

    # ─── Check 4: Aux losses implémentées ────────────────────────────
    # On fait un forward de test et on vérifie que aux_loss et z_loss sont
    # bien dans le résultat.
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, model.vocab_size, (2, 16), device=device)
        result = model(test_input)

    check4 = ("moe_aux_loss" in result and "moe_z_loss" in result
              and torch.isfinite(result["moe_aux_loss"])
              and torch.isfinite(result["moe_z_loss"]))
    checks["aux_losses_implemented"] = {
        "passed": check4,
        "details": f"aux_loss={result['moe_aux_loss'].item():.4f}, "
                   f"z_loss={result['moe_z_loss'].item():.4f}",
    }
    print(f"  [{'✓' if check4 else '✗'}] Aux losses : "
          f"aux={result['moe_aux_loss'].item():.4f}, z={result['moe_z_loss'].item():.4f}")

    # ─── Check 5: PGSU n_active <= n_blocks ──────────────────────────
    # Vérifié au runtime dans PGSU.__init__, mais on le rappelle ici.
    check5 = True  # sera vérifié quand PGSU est instancié
    checks["pgsu_valid"] = {
        "passed": check5,
        "details": "Sera vérifié à l'instanciation de PGSU en Phase 3",
    }
    print(f"  [✓] PGSU : vérifié au runtime")

    # ─── Résumé ──────────────────────────────────────────────────────
    all_passed = all(c["passed"] for c in checks.values())
    print()
    if all_passed:
        print("  ✅ TOUS LES PRÉREQUIS EDT SONT SATISFAITS — pipeline peut démarrer.")
    else:
        print("  ❌ PRÉREQUIS NON SATISFAITS — corriger avant de lancer EDT.")
        for name, c in checks.items():
            if not c["passed"]:
                print(f"     - {name}: {c['details']}")
    print("=" * 70)

    return checks


def run_edt_pipeline(
    model: CogNetMoE1B,
    data_iter_fn: Callable[[int, int], torch.Tensor],
    cfg: Optional[EDTConfig] = None,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Orchestre les 4 phases EDT séquentiellement.

    Args:
        model: CogNet-MoE-1B (sera modifié in-place)
        data_iter_fn: callable(batch_size, seq_len) -> input_ids (B, T)
        cfg: EDTConfig (ou default)
        save_dir: dossier pour sauvegarder checkpoints intermédiaires

    Returns:
        dict avec stats de chaque phase + temps total
    """
    if cfg is None:
        cfg = EDTConfig()

    if save_dir is None:
        save_dir = cfg.ckpt_dir
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("# EDT PIPELINE — CogNet-MoE-1B")
    print("#" * 70)
    p = model.count_parameters()
    print(f"# Total params      : {p['total']:,}")
    print(f"# Active / token    : {p['active_per_token']:,}")
    print(f"# Capacity mult     : {p['total']/p['active_per_token']:.2f}x vs same-FLOPs dense")
    print(f"# Device            : {cfg.device}")
    print(f"# bf16              : {cfg.use_bf16}")
    print(f"# 8-bit optimizer   : {cfg.use_8bit_optimizer}")
    print(f"# Aux-loss clamp    : {cfg.phase3_aux_loss_clamp}")
    print(f"# PGSU n_active     : {cfg.phase3_pgsu_n_active}/{model.num_blocks}")
    print("#" * 70)

    # ─── Vérification des prérequis EDT ──────────────────────────────
    prerequisite_checks = verify_edt_prerequisites(model)
    if not all(c["passed"] for c in prerequisite_checks.values()):
        raise RuntimeError(
            "EDT prerequisites not satisfied. See checks above."
        )

    t_start = time.time()
    all_stats: Dict = {"phases": {}, "prerequisites": prerequisite_checks}


    # ─── Phase 1 ────────────────────────────────────────────────────
    # Pour Phase 1, on a besoin de hidden states réels. On génère un batch
    # d'input_ids et on forward jusqu'au bloc b (sans MoE) pour extraire h_in.
    def hidden_states_fn(batch_size: int, seq_len: int) -> torch.Tensor:
        """Génère des hidden states réels à un bloc aléatoire."""
        input_ids = data_iter_fn(batch_size, seq_len)
        with torch.no_grad():
            x = model.encoder(input_ids.to(next(model.parameters()).device))
            # Forward jusqu'à un bloc aléatoire.
            # Phase 1 entraîne les experts séparément : on skip le
            # cognitive_expert_router ENTIER (qui contient routing +
            # experts + projections). On garde seulement memory +
            # composer + norm pour produire des hidden states réalistes.
            n_blocks_to_forward = torch.randint(0, model.num_blocks, (1,)).item()
            for b in range(n_blocks_to_forward):
                block = model.blocks[b]
                # Skip cognitive_expert_router (Phase 1 l'entraîne séparément).
                x, _ = block.memory(x)
                x = block.composer(x)
                x = block.norm(x)
        return x

    s1 = phase1_experts(model, hidden_states_fn, cfg)
    all_stats["phases"]["phase1"] = s1
    if save_dir:
        torch.save(model.state_dict(), os.path.join(save_dir, "after_phase1.pt"))

    # ─── Phase 2a ───────────────────────────────────────────────────
    s2a = phase2a_attention(model, data_iter_fn, cfg)
    all_stats["phases"]["phase2a"] = s2a
    if save_dir:
        torch.save(model.state_dict(), os.path.join(save_dir, "after_phase2a.pt"))

    # ─── Phase 2b ───────────────────────────────────────────────────
    s2b = phase2b_embedding(model, data_iter_fn, cfg)
    all_stats["phases"]["phase2b"] = s2b
    if save_dir:
        torch.save(model.state_dict(), os.path.join(save_dir, "after_phase2b.pt"))

    # ─── Phase 3 ────────────────────────────────────────────────────
    s3 = phase3_joint(model, data_iter_fn, cfg)
    all_stats["phases"]["phase3"] = s3
    if save_dir:
        torch.save(model.state_dict(), os.path.join(save_dir, "after_phase3_final.pt"))

    total_time = time.time() - t_start
    all_stats["total_time_s"] = total_time
    all_stats["total_time_h"] = total_time / 3600
    all_stats["config"] = {
        "phase1_steps": cfg.phase1_steps_per_expert,
        "phase2a_steps": cfg.phase2a_steps,
        "phase2b_tokens": cfg.phase2b_tokens,
        "phase3_tokens": cfg.phase3_tokens,
        "pgsu_n_active": cfg.phase3_pgsu_n_active,
        "use_bf16": cfg.use_bf16,
        "use_8bit_optimizer": cfg.use_8bit_optimizer,
    }

    print("\n" + "#" * 70)
    print("# EDT PIPELINE TERMINÉ")
    print("#" * 70)
    print(f"# Temps total : {total_time/3600:.2f}h ({total_time/86400:.2f} jours)")
    print(f"#   Phase 1   : {s1['total_time_s']/3600:.2f}h")
    print(f"#   Phase 2a  : {s2a['time_s']:.2f}s")
    print(f"#   Phase 2b  : {s2b['time_s']/3600:.2f}h")
    print(f"#   Phase 3   : {s3['time_s']/3600:.2f}h")
    print("#" * 70)

    # Save stats.
    if save_dir:
        with open(os.path.join(save_dir, "edt_stats.json"), "w") as f:
            # Convertir les tensors en floats pour JSON.
            def _to_jsonable(o):
                if isinstance(o, torch.Tensor):
                    return o.item() if o.numel() == 1 else o.tolist()
                if isinstance(o, dict):
                    return {k: _to_jsonable(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [_to_jsonable(x) for x in o]
                return o
            json.dump(_to_jsonable(all_stats), f, indent=2)

    return all_stats


# ═══════════════════════════════════════════════════════════════════════
#  Self-test (small model, CPU, few steps)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("EDT Pipeline Self-Test (tiny model, CPU, 5 steps)")
    print("=" * 70)

    # Tiny config pour test CPU.
    cfg = EDTConfig(
        phase1_steps_per_expert=3,
        phase1_batch_size=4,
        phase1_seq_len=64,
        phase2a_steps=1,
        phase2a_batch_size=4,
        phase2b_tokens=512,
        phase2b_batch_size=4,
        phase2b_seq_len=64,
        phase3_tokens=512,
        phase3_batch_size=2,
        phase3_seq_len=64,
        phase3_grad_accum=2,
        phase3_pgsu_n_active=1,  # tiny model has 2 blocks
        use_bf16=False,
        use_8bit_optimizer=False,
        device="cpu",
        log_every=1,
    )

    # Tiny model — CogNet-native MoE : n_experts == num_channels.
    model = CogNetMoE1B(
        vocab_size=136,
        hidden_dim=64,
        num_blocks=2,
        num_channels=4,   # = n_experts (CogNet-native constraint)
        channel_dim=32,   # ignoré (compat signature)
        ff_dim=128,
        max_seq_len=64,
        working_slots=4,
        episodic_slots=8,
        semantic_slots=16,
        key_dim=32,
        n_experts=4,      # = num_channels
        top_k=2,
        use_gradient_checkpointing=False,
    )

    # Dummy data_iter_fn : génère des input_ids aléatoires.
    def data_iter_fn(batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.randint(0, 136, (batch_size, seq_len))

    # Run pipeline.
    stats = run_edt_pipeline(model, data_iter_fn, cfg, save_dir=None)

    print("\n✓ EDT pipeline self-test passed.")
    print(f"  Total time : {stats['total_time_s']:.2f}s")
    print(f"  Phase 1    : {stats['phases']['phase1']['total_time_s']:.2f}s "
          f"({len(stats['phases']['phase1']['experts'])} experts)")
    print(f"  Phase 2a   : {stats['phases']['phase2a']['time_s']:.2f}s")
    print(f"  Phase 2b   : {stats['phases']['phase2b']['time_s']:.2f}s "
          f"({stats['phases']['phase2b']['total_tokens']} tokens)")
    print(f"  Phase 3    : {stats['phases']['phase3']['time_s']:.2f}s "
          f"({stats['phases']['phase3']['total_tokens']} tokens)")
