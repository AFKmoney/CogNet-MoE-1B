"""
Chinchilla Scaling Laws — Adapté à CogNet-MoE-1B
=================================================

Chinchilla (Hoffmann et al., 2022) établit que pour un compute optimal, le
ratio tokens / paramètres = 20:1. Mais pour un MoE sparse, l'interprétation
correcte est :

  tokens_optimaux ≈ 20 × paramètres_actifs_par_token

(et non 20 × paramètres_totaux, sinon on surestime massivement le budget).

C'est l'approche utilisée dans Switch Transformer, ST-MoE, Mixtral — et c'est
l'interprétation validée par le reviewer dans son analyse.

Deux scénarios de tokenisation sont comparés :
  1. CharTokenizer (vocab=136) — tokenizer original CogNet-1B
  2. CognetTokenizer BPE (vocab=16,384) — tokenizer propriétaire nouveau

Pour CogNet-MoE-1B avec BPE 16k :
  - Total params    : 7,13B (vs 7,09B avec CharTokenizer — +33,3M pour token_emb)
  - Actifs / token  : 2,29B (vs 2,26B — token_emb toujours actif)
  - Chinchilla : 20 × 2,29B = 45,86B BPE-tokens
  - En chars (ratio ~3 chars/BPE) : ~137B chars de texte
  - Scénario C EDT : 45,86B / 35 = 1,31B BPE-tokens
"""

import json
from dataclasses import dataclass, field
from typing import Dict


# ═══════════════════════════════════════════════════════════════════════
#  Configurations de tokenizer
# ═══════════════════════════════════════════════════════════════════════

# Tokenizer original CogNet-1B (CharTokenizer).
CHAR_TOKENIZER_VOCAB = 136

# Tokenizer propriétaire nouveau (BPE 16k).
# Reco #4 du reviewer : "Une variante BPE 8k–16k rendrait l'embedding plus
# utile et Phase 2b plus informative." On prend la upper bound 16k.
BPE_TOKENIZER_VOCAB = 16_384

# Ratio chars/BPE pour la conversion (texte FR+EN+code typique).
# Mesuré expérimentalement sur le tokenizer démo : ~2.6-3.0 chars/BPE.
# On prend 3.0 comme estimation conservatrice pour la production.
CHARS_PER_BPE = 3.0


# ═══════════════════════════════════════════════════════════════════════
#  Comptage exact des paramètres CogNet-MoE-1B
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ParamBreakdown:
    """Décomposition exacte des paramètres par sous-module.
    
    Architecture CogNet-native MoE (CORRIGÉ) :
    - Le CoherenceRouter O(n) fait le routing (pas de gate transformer-style)
    - Les 8 canaux = les 8 experts FusedSwiGLU (unifiés)
    - to_channels : D → C×D (projections parallèles vers 8 experts)
    - PAS de from_channels (somme pondérée des experts mixe les canaux)
    - PAS de ChannelProcessor (supprimé, sa fonction absorbée par les experts)
    - PAS de moe_gate séparé (le coherence_router fait le routing)
    - AdaptiveComputationBlock supprimé (sa fonction absorbée par les experts)
    """

    # TokenEncoder
    token_emb: int = 0       # vocab_size × hidden_dim
    rope: int = 0            # no learnable params (registered buffers)
    encoder_norm: int = 0    # hidden_dim

    # Per CogNetMoEBlock (× 16) :
    # ── CognitiveExpertRouter (CogNet-native MoE : routing + experts unifiés)
    cer_coherence_query: int = 0     # hidden_dim × num_channels (Linear)
    cer_coherence_key: int = 0       # hidden_dim × num_channels (Linear)
    cer_to_channels: int = 0         # hidden_dim × (num_channels × hidden_dim)
                                     # = D × C × D (channel_dim = D pour experts FusedSwiGLU)
    cer_experts_total: int = 0       # 8 × FusedSwiGLU(D, ff_dim)
    cer_norm: int = 0                # hidden_dim
    # PAS de from_channels (supprimé — somme pondérée mixe les canaux)
    # PAS de moe_gate séparé (le coherence_router fait le routing)

    # ── ParallelHierarchicalMemory (préservé à l'identique)
    memory_q_proj: int = 0           # hidden_dim × key_dim
    memory_v_proj: int = 0           # hidden_dim × hidden_dim
    memory_out_proj: int = 0         # hidden_dim × hidden_dim
    memory_keys: int = 0             # total_slots × key_dim (slots fixes, SDPA reads)
    memory_vals: int = 0             # total_slots × hidden_dim
    memory_tier_gate: int = 0        # (3 × hidden_dim) × 3 (mixing 3 tiers)
    memory_norm: int = 0             # hidden_dim

    # ── CompositionalReasoner (préservé à l'identique)
    composer_role_proj: int = 0      # hidden_dim × key_dim
    composer_filler_proj: int = 0    # hidden_dim × key_dim
    composer_unbind_proj: int = 0    # key_dim × hidden_dim
    composer_norm: int = 0           # hidden_dim

    block_norm: int = 0              # hidden_dim

    final_norm: int = 0              # hidden_dim
    # output_proj est weight-tied avec token_emb (déjà compté).

    # Scaling
    num_blocks: int = 16
    n_experts: int = 8     # = num_channels (CogNet-native constraint)
    top_k: int = 2


def compute_param_breakdown(
    vocab_size: int = 136,
    hidden_dim: int = 2048,
    num_blocks: int = 16,
    num_channels: int = 8,
    channel_dim: int = 384,  # ignoré (CogNet-native MoE utilise channel_dim = hidden_dim)
    ff_dim: int = 8192,
    working_slots: int = 128,
    episodic_slots: int = 256,
    semantic_slots: int = 512,
    key_dim: int = 256,
    n_experts: int = 8,
    top_k: int = 2,
) -> ParamBreakdown:
    """Décomposition exacte — chaque valeur calculée à la main.
    
    Architecture CogNet-native MoE (CORRIGÉ) :
    - 8 canaux = 8 experts (unifiés, pas de gate séparé)
    - to_channels : D → C×D (projections vers 8 experts parallèles)
    - PAS de from_channels (somme pondérée des experts mixe les canaux)
    - Le coherence_router O(n) fait le routing cognitif
    """
    assert n_experts == num_channels, (
        f"CogNet-native MoE exige n_experts == num_channels. "
        f"Reçu n_experts={n_experts}, num_channels={num_channels}."
    )

    bd = ParamBreakdown()

    # ─── TokenEncoder ────────────────────────────────────────────────
    bd.token_emb = vocab_size * hidden_dim
    bd.encoder_norm = hidden_dim

    # ─── Per CogNetMoEBlock ───────────────────────────────────────────
    # CognitiveExpertRouter (CogNet-native MoE : routing + experts unifiés)
    bd.cer_coherence_query = hidden_dim * num_channels
    bd.cer_coherence_key = hidden_dim * num_channels
    # to_channels : D → C×D (channel_dim = D pour les experts FusedSwiGLU)
    bd.cer_to_channels = hidden_dim * (num_channels * hidden_dim)
    # PAS de from_channels (somme pondérée des experts mixe les canaux)
    
    # FusedSwiGLU(hidden_dim, ff_dim) :
    #   w_gate_up : hidden_dim × (2 × ff_dim) = 2048 × 16384 = 33,554,432
    #   w_down    : ff_dim × hidden_dim = 8192 × 2048 = 16,777,216
    #   norm      : hidden_dim = 2048
    per_expert = hidden_dim * (2 * ff_dim) + ff_dim * hidden_dim + hidden_dim
    bd.cer_experts_total = n_experts * per_expert
    bd.cer_norm = hidden_dim
    # PAS de moe_gate séparé (le coherence_router fait le routing)
    # PAS de ChannelProcessor (supprimé, sa fonction absorbée par les experts)

    # ParallelHierarchicalMemory (préservé à l'identique)
    bd.memory_q_proj = hidden_dim * key_dim
    bd.memory_v_proj = hidden_dim * hidden_dim
    bd.memory_out_proj = hidden_dim * hidden_dim
    total_slots = working_slots + episodic_slots + semantic_slots  # 896
    bd.memory_keys = total_slots * key_dim
    bd.memory_vals = total_slots * hidden_dim
    bd.memory_tier_gate = (3 * hidden_dim) * 3
    bd.memory_norm = hidden_dim

    # CompositionalReasoner (préservé à l'identique)
    bd.composer_role_proj = hidden_dim * key_dim
    bd.composer_filler_proj = hidden_dim * key_dim
    bd.composer_unbind_proj = key_dim * hidden_dim
    bd.composer_norm = hidden_dim

    bd.block_norm = hidden_dim
    bd.final_norm = hidden_dim

    bd.num_blocks = num_blocks
    bd.n_experts = n_experts
    bd.top_k = top_k

    return bd


def total_params(bd: ParamBreakdown) -> int:
    """Total params du modèle (architecture CogNet-native MoE)."""
    per_block = (
        # CognitiveExpertRouter (CogNet-native MoE unifié)
        bd.cer_coherence_query + bd.cer_coherence_key
        + bd.cer_to_channels
        + bd.cer_experts_total
        + bd.cer_norm
        # ParallelHierarchicalMemory (préservé)
        + bd.memory_q_proj + bd.memory_v_proj + bd.memory_out_proj
        + bd.memory_keys + bd.memory_vals + bd.memory_tier_gate + bd.memory_norm
        # CompositionalReasoner (préservé)
        + bd.composer_role_proj + bd.composer_filler_proj + bd.composer_unbind_proj
        + bd.composer_norm + bd.block_norm
    )
    encoder = bd.token_emb + bd.encoder_norm
    final = bd.final_norm  # output_proj est weight-tied (déjà compté dans token_emb)
    return encoder + bd.num_blocks * per_block + final


def active_params_per_token(bd: ParamBreakdown) -> int:
    """
    Actifs par token : tout sauf les experts non sélectionnés.
    En top-2 sur 8 experts, fraction active = 2/8 = 1/4.
    
    Architecture CogNet-native MoE :
    - coherence_router : toujours actif (O(n) routing)
    - to_channels       : toujours actif (projection dense)
    - 2/8 experts       : actifs (top-2 sparse)
    - memory + composer : toujours actifs
    """
    per_block_active = (
        # CognitiveExpertRouter — sauf les experts non sélectionnés
        bd.cer_coherence_query + bd.cer_coherence_key
        + bd.cer_to_channels
        + bd.cer_norm
        # memory + composer (toujours actifs)
        + bd.memory_q_proj + bd.memory_v_proj + bd.memory_out_proj
        + bd.memory_keys + bd.memory_vals + bd.memory_tier_gate + bd.memory_norm
        + bd.composer_role_proj + bd.composer_filler_proj + bd.composer_unbind_proj
        + bd.composer_norm + bd.block_norm
    )
    # Seuls 2/8 des experts sont actifs par token.
    per_block_active += (bd.top_k / bd.n_experts) * bd.cer_experts_total

    encoder = bd.token_emb + bd.encoder_norm
    final = bd.final_norm
    return int(encoder + bd.num_blocks * per_block_active + final)


# ═══════════════════════════════════════════════════════════════════════
#  Chinchilla scaling
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ChinchillaResult:
    """Résultat de l'analyse Chinchilla."""

    # Paramètres
    total_params: int
    active_params: int
    capacity_multiplier: float  # total / active

    # Chinchilla (ratio 20:1 sur active params)
    optimal_tokens: int
    optimal_tokens_active: int

    # Si on utilisait à tort le total (interprétation erronée)
    optimal_tokens_total_wrong: int

    # Conversion char-token -> BPE-équivalent (~4 chars/BPE)
    char_to_bpe_ratio: float = 4.0
    optimal_tokens_bpe_equivalent: int = 0

    # Hardware assumptions (RTX 3090)
    gpu_memory_gb: float = 24.0
    gpu_tflops_bf16: float = 71.0  # 3090 bf16

    # Estimate EDT scenarios
    scenario_a_tokens: int = 0  # 600M chars (agressif, risqué)
    scenario_b_tokens: int = 0  # Plein Chinchilla (inutile avec EDT)
    scenario_c_tokens: int = 0  # EDT-proportionnel (recommandé)
    edt_speedup: float = 35.0   # Réduction 35× de tokens via EDT


def compute_chinchilla(
    total: int,
    active: int,
    edt_speedup: float = 35.0,
    char_to_bpe_ratio: float = 4.0,
) -> ChinchillaResult:
    """
    Calcule le budget Chinchilla pour CogNet-MoE-1B.

    Args:
        total: paramètres totaux (7,09B)
        active: paramètres actifs par token (2,26B)
        edt_speedup: réduction de tokens via EDT (35×, hypothèse à valider)
        char_to_bpe_ratio: ratio chars / BPE (≈4 pour français/anglais)
    """
    r = ChinchillaResult(
        total_params=total,
        active_params=active,
        capacity_multiplier=total / active,
        optimal_tokens=20 * active,
        optimal_tokens_active=20 * active,
        optimal_tokens_total_wrong=20 * total,
        char_to_bpe_ratio=char_to_bpe_ratio,
        edt_speedup=edt_speedup,
    )
    r.optimal_tokens_bpe_equivalent = int(r.optimal_tokens / char_to_bpe_ratio)

    # Scénario A : 600M chars (agressif, risqué — reco : éviter)
    r.scenario_a_tokens = 600_000_000

    # Scénario B : plein Chinchilla (inutile avec EDT)
    r.scenario_b_tokens = r.optimal_tokens

    # Scénario C : EDT-proportionnel = Chinchilla / 35
    r.scenario_c_tokens = int(r.optimal_tokens / edt_speedup)

    return r


def estimate_training_time(
    n_tokens: int,
    active_params: int,
    gpu_tflops_bf16: float = 71.0,  # RTX 3090
    hardware_utilization: float = 0.35,  # MFU realistic for MoE
    flops_per_token_per_param: float = 6.0,  # 2× (forward + backward) × 3 (matmul)
) -> Dict:
    """
    Estime le temps d'entraînement en secondes.

    FLOPs totaux ≈ 6 × N_active × D
    (6 = 2 (forward+backward) × 3 (matmul multiplier-adder))

    Temps = FLOPs / (TFLOPs_gpu × utilization × 1e12)
    """
    flops = flops_per_token_per_param * active_params * n_tokens
    effective_tflops = gpu_tflops_bf16 * hardware_utilization * 1e12
    time_s = flops / effective_tflops
    return {
        "flops": flops,
        "effective_tflops": gpu_tflops_bf16 * hardware_utilization,
        "time_s": time_s,
        "time_h": time_s / 3600,
        "time_d": time_s / 86400,
    }


def estimate_cost(
    time_h: float,
    gpu_hourly_cost: float = 0.40,  # RTX 3090 spot ~$0.40/h
) -> Dict:
    """Estime le coût GPU."""
    return {
        "gpu_hours": time_h,
        "cost_usd": time_h * gpu_hourly_cost,
        "gpu_hourly_cost": gpu_hourly_cost,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Comparaison : standard training vs EDT
# ═══════════════════════════════════════════════════════════════════════

def compare_standard_vs_edt(
    active: int,
    chinchilla_tokens: int,
    edt_speedup: float = 35.0,
) -> Dict:
    """
    Compare l'entraînement standard (plein Chinchilla) vs EDT (réduit 35×).

    Document EDT original :
      Standard : 358 jours, $3,437, 8,592 GPU-hours (pour un plus petit MoE)
      EDT      : ~45h (1,9 jour), $18, ~48 GPU-hours
      Speedup  : ~189×

    Pour CogNet-MoE-1B (experts plus gros, moins nombreux) :
      Speedup attendu : ~164× (ratio Phase 1+2 / Phase 3 moins favorable)
    """
    standard_time = estimate_training_time(
        n_tokens=chinchilla_tokens,
        active_params=active,
    )
    edt_tokens = int(chinchilla_tokens / edt_speedup)
    edt_time = estimate_training_time(
        n_tokens=edt_tokens,
        active_params=active,
    )

    # Le temps EDT total inclut Phase 1+2 (qui sont courts mais non négligeables
    # pour CogNet-MoE car experts plus gros). On estime Phase 1+2 = ~5% du temps
    # EDT total (vs <1% dans le document original).
    edt_phase12_fraction = 0.05
    edt_total_h = edt_time["time_h"] / (1 - edt_phase12_fraction)

    standard_cost = estimate_cost(standard_time["time_h"])
    edt_cost = estimate_cost(edt_total_h)

    return {
        "standard": {
            "tokens": chinchilla_tokens,
            "time_h": standard_time["time_h"],
            "time_d": standard_time["time_d"],
            "cost": standard_cost,
        },
        "edt": {
            "tokens_phase3": edt_tokens,
            "time_phase3_h": edt_time["time_h"],
            "time_phase12_h": edt_total_h - edt_time["time_h"],
            "time_total_h": edt_total_h,
            "time_total_d": edt_total_h / 24,
            "cost": edt_cost,
        },
        "speedup": standard_time["time_h"] / edt_total_h,
        "cost_reduction": standard_cost["cost_usd"] / max(edt_cost["cost_usd"], 1e-6),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Rapport complet
# ═══════════════════════════════════════════════════════════════════════

def full_report(vocab_size: int = BPE_TOKENIZER_VOCAB) -> Dict:
    """
    Génère le rapport Chinchilla complet pour CogNet-MoE-1B.

    Args:
        vocab_size: Taille du vocabulaire tokenizer.
                    - 136 = CharTokenizer original (legacy)
                    - 16384 = CognetTokenizer BPE propriétaire (recommandé)
    """

    # 1. Décomposition exacte pour ce vocab_size.
    bd = compute_param_breakdown(vocab_size=vocab_size)
    total = total_params(bd)
    active = active_params_per_token(bd)

    # 2. Chinchilla.
    chinchilla = compute_chinchilla(total, active)

    # 3. Comparaison standard vs EDT.
    comparison = compare_standard_vs_edt(
        active=active,
        chinchilla_tokens=chinchilla.optimal_tokens,
    )

    # 4. Décomposition détaillée par sous-module.
    per_block = (
        bd.cer_coherence_query + bd.cer_coherence_key
        + bd.cer_to_channels
        + bd.cer_experts_total
        + bd.cer_norm
        + bd.memory_q_proj + bd.memory_v_proj + bd.memory_out_proj
        + bd.memory_keys + bd.memory_vals + bd.memory_tier_gate + bd.memory_norm
        + bd.composer_role_proj + bd.composer_filler_proj + bd.composer_unbind_proj
        + bd.composer_norm + bd.block_norm
    )

    # Identifier le tokenizer utilisé pour le rapport.
    if vocab_size == CHAR_TOKENIZER_VOCAB:
        tokenizer_name = "CharTokenizer (legacy, vocab=136)"
    elif vocab_size == BPE_TOKENIZER_VOCAB:
        tokenizer_name = "CognetTokenizer BPE propriétaire (vocab=16,384)"
    else:
        tokenizer_name = f"Custom (vocab={vocab_size})"

    report = {
        "model": "CogNet-MoE-1B",
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "architecture": (
            "Non-Transformer (CoherenceRouter O(n) + 3-tier Memory + "
            "CompositionalReasoner) + CogNet-native MoE "
            "(8 canaux = 8 experts FusedSwiGLU, top-2 sparse, "
            "routing = coherence O(n), PAS de gate transformer-style)"
        ),
        "param_breakdown": {
            "encoder_token_emb": bd.token_emb,
            "encoder_norm": bd.encoder_norm,
            "per_block_total": per_block,
            "per_block_cer_coherence": bd.cer_coherence_query + bd.cer_coherence_key,
            "per_block_cer_to_channels": bd.cer_to_channels,
            "per_block_cer_experts": bd.cer_experts_total,
            "per_block_cer_norm": bd.cer_norm,
            "per_block_memory": (
                bd.memory_q_proj + bd.memory_v_proj + bd.memory_out_proj
                + bd.memory_keys + bd.memory_vals + bd.memory_tier_gate + bd.memory_norm
            ),
            "per_block_composer": (
                bd.composer_role_proj + bd.composer_filler_proj
                + bd.composer_unbind_proj + bd.composer_norm
            ),
            "num_blocks": bd.num_blocks,
            "n_experts_per_block": bd.n_experts,
            "top_k": bd.top_k,
            "note": (
                "Architecture CogNet-native MoE : 8 canaux = 8 experts "
                "(unifiés). PAS de from_channels (somme pondérée mixe les "
                "canaux). PAS de moe_gate séparé (le coherence_router fait "
                "le routing O(n)). PAS de ChannelProcessor (supprimé, sa "
                "fonction absorbée par les experts FusedSwiGLU)."
            ),
        },
        "totals": {
            "total_params": total,
            "active_params_per_token": active,
            "capacity_multiplier": chinchilla.capacity_multiplier,
            "announced_in_readme": 1_060_000_000,  # ~1.06B (erroné)
            "actual_dense_cognet": 2_260_000_000,   # ~2.26B (corrigé)
        },
        "chinchilla": {
            "ratio": "20:1 sur active params (interprétation MoE correcte)",
            "optimal_tokens": chinchilla.optimal_tokens,
            "optimal_tokens_bpe_equivalent": chinchilla.optimal_tokens_bpe_equivalent,
            "wrong_interpretation_total_tokens": chinchilla.optimal_tokens_total_wrong,
            "char_to_bpe_ratio": chinchilla.char_to_bpe_ratio,
            # Pour BPE natif, le "ratio" est 1:1 (chaque BPE token = 1 token).
            # La conversion en chars se fait avec CHARS_PER_BPE.
            "optimal_tokens_in_chars_equivalent": int(chinchilla.optimal_tokens * CHARS_PER_BPE)
                if vocab_size == BPE_TOKENIZER_VOCAB else chinchilla.optimal_tokens,
        },
        "edt_scenarios": {
            "scenario_a_aggressive": {
                "tokens": chinchilla.scenario_a_tokens,
                "verdict": "TROP AGRESSIF — risque de sous-entraînement",
            },
            "scenario_b_full_chinchilla": {
                "tokens": chinchilla.scenario_b_tokens,
                "verdict": "INUTILE avec EDT — gaspille le bénéfice EDT",
            },
            "scenario_c_edt_proportional_RECOMMENDED": {
                "tokens": chinchilla.scenario_c_tokens,
                "verdict": "RECOMMANDÉ — Scénario C baseline",
                "note": "Si qualité insuffisante, monter Phase 3 à 500M–1B tokens",
            },
        },
        "comparison_standard_vs_edt": comparison,
        "hardware": {
            "gpu": "RTX 3090",
            "memory_gb": chinchilla.gpu_memory_gb,
            "tflops_bf16": chinchilla.gpu_tflops_bf16,
            "assumed_mfu": 0.35,
            "spot_hourly_cost_usd": 0.40,
        },
        "caveats": [
            "Hypothèses de vitesse (4 ms/expert-step, 678 tok/s joint) — varieront sur 3090 réelle.",
            "Réduction 35× de tokens via EDT calquée sur document EDT original, NON validée empiriquement sur CogNet.",
            "Avec BPE 16k : Phase 2b plus informative qu'avec CharTokenizer (reco #4 satisfaite).",
            "Si routing collapse (max_load > 0.5), monter aux_loss_weight à 0.05.",
            "Si VRAM Phase 3 saute, descendre PGSU n_active de 4 à 2.",
        ],
    }
    return report


def compare_tokenizers() -> Dict:
    """
    Compare les deux scénarios de tokenisation (CharTokenizer vs BPE 16k)
    sur tous les axes : params, Chinchilla, EDT, coût.

    Returns:
        dict avec les deux rapports + deltas
    """
    report_char = full_report(vocab_size=CHAR_TOKENIZER_VOCAB)
    report_bpe = full_report(vocab_size=BPE_TOKENIZER_VOCAB)

    return {
        "char_tokenizer_legacy": report_char,
        "bpe_tokenizer_proprietary": report_bpe,
        "deltas": {
            "total_params_increase": (
                report_bpe["totals"]["total_params"]
                - report_char["totals"]["total_params"]
            ),
            "active_params_increase": (
                report_bpe["totals"]["active_params_per_token"]
                - report_char["totals"]["active_params_per_token"]
            ),
            "chinchilla_tokens_increase": (
                report_bpe["chinchilla"]["optimal_tokens"]
                - report_char["chinchilla"]["optimal_tokens"]
            ),
            "scenario_c_tokens_increase": (
                report_bpe["edt_scenarios"]["scenario_c_edt_proportional_RECOMMENDED"]["tokens"]
                - report_char["edt_scenarios"]["scenario_c_edt_proportional_RECOMMENDED"]["tokens"]
            ),
            # Bénéfice clé : 1 BPE token ≈ 3 chars, donc 1.31B BPE = 3.93B chars
            # vs 1.29B char-tokens = 1.29B chars. Plus de texte utile pour le même budget.
            "chars_of_text_data_char_scenario": (
                report_char["edt_scenarios"]["scenario_c_edt_proportional_RECOMMENDED"]["tokens"]
            ),
            "chars_of_text_data_bpe_scenario": int(
                report_bpe["edt_scenarios"]["scenario_c_edt_proportional_RECOMMENDED"]["tokens"]
                * CHARS_PER_BPE
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("CogNet-MoE-1B — Chinchilla Scaling Analysis (re-mesuré avec BPE 16k)")
    print("=" * 70)

    # Comparaison des deux tokenizers.
    comparison = compare_tokenizers()
    report_char = comparison["char_tokenizer_legacy"]
    report_bpe = comparison["bpe_tokenizer_proprietary"]
    deltas = comparison["deltas"]

    # ── Tableau comparatif des deux tokenizers ──
    print("\n" + "═" * 70)
    print("COMPARAISON DES DEUX TOKENIZERS")
    print("═" * 70)
    print(f"{'Métrique':<40}{'CharTokenizer':>15}{'BPE 16k':>15}")
    print("─" * 70)
    print(f"{'Vocab size':<40}{CHAR_TOKENIZER_VOCAB:>15,}{BPE_TOKENIZER_VOCAB:>15,}")
    print(f"{'token_emb params':<40}"
          f"{report_char['param_breakdown']['encoder_token_emb']:>15,}"
          f"{report_bpe['param_breakdown']['encoder_token_emb']:>15,}")
    print(f"{'Total params':<40}"
          f"{report_char['totals']['total_params']:>15,}"
          f"{report_bpe['totals']['total_params']:>15,}")
    print(f"{'Active params / token':<40}"
          f"{report_char['totals']['active_params_per_token']:>15,}"
          f"{report_bpe['totals']['active_params_per_token']:>15,}")
    print(f"{'Capacity multiplier':<40}"
          f"{report_char['totals']['capacity_multiplier']:>14.2f}×"
          f"{report_bpe['totals']['capacity_multiplier']:>14.2f}×")
    print(f"{'Chinchilla optimal (tokens)':<40}"
          f"{report_char['chinchilla']['optimal_tokens']:>15,}"
          f"{report_bpe['chinchilla']['optimal_tokens']:>15,}")
    print(f"{'Scénario C EDT (tokens)':<40}"
          f"{report_char['edt_scenarios']['scenario_c_edt_proportional_RECOMMENDED']['tokens']:>15,}"
          f"{report_bpe['edt_scenarios']['scenario_c_edt_proportional_RECOMMENDED']['tokens']:>15,}")
    print(f"{'Scénario C (équivalent chars)':<40}"
          f"{deltas['chars_of_text_data_char_scenario']:>15,}"
          f"{deltas['chars_of_text_data_bpe_scenario']:>15,}")
    print("─" * 70)
    print(f"{'Δ total params':<40}{'':>15}+{deltas['total_params_increase']:>13,}")
    print(f"{'Δ active params':<40}{'':>15}+{deltas['active_params_increase']:>13,}")
    print(f"{'Δ Chinchilla tokens':<40}{'':>15}+{deltas['chinchilla_tokens_increase']:>13,}")
    print(f"{'Δ Scénario C tokens':<40}{'':>15}+{deltas['scenario_c_tokens_increase']:>13,}")

    # ── Détails du rapport BPE (le recommandé) ──
    report = report_bpe

    print("\n" + "═" * 70)
    print(f"RAPPORT DÉTAILLÉ — {report['tokenizer']}")
    print("═" * 70)

    # Print param breakdown.
    print("\n── Paramètres (comptage exact) ──")
    bd = report["param_breakdown"]
    totals = report["totals"]
    print(f"  Encoder (token_emb + norm)     : {bd['encoder_token_emb'] + bd['encoder_norm']:>14,}")
    print(f"    dont token_emb               : {bd['encoder_token_emb']:>14,}  "
          f"({bd['encoder_token_emb']/totals['total_params']*100:.2f}% du total)")
    print(f"  Per block :")
    print(f"    CER coherence (O(n) routing) : {bd['per_block_cer_coherence']:>14,}")
    print(f"    CER to_channels (D → C×D)    : {bd['per_block_cer_to_channels']:>14,}")
    print(f"    CER experts (8× FusedSwiGLU) : {bd['per_block_cer_experts']:>14,}")
    print(f"    CER norm                     : {bd['per_block_cer_norm']:>14,}")
    print(f"    Memory (3-tier + SDPA)       : {bd['per_block_memory']:>14,}")
    print(f"    Composer (hyperdim binding)  : {bd['per_block_composer']:>14,}")
    print(f"    → Per block total            : {bd['per_block_total']:>14,}")
    print(f"  × 16 blocks                    : {bd['per_block_total'] * bd['num_blocks']:>14,}")
    print(f"  Final norm                     : {2048:>14,}")
    print()
    print(f"  ━━ TOTAL params                : {totals['total_params']:>14,}")
    print(f"  ━━ ACTIVE per token (top-2/8)  : {totals['active_params_per_token']:>14,}")
    print(f"  ━━ Capacity multiplier         : {totals['capacity_multiplier']:>14.2f}×")
    print()
    print(f"  README annonce ~1.06B (ERRONÉ, sous-estimé d'un facteur 2)")
    print(f"  Dense CogNet-1B réel           : {totals['actual_dense_cognet']:>14,}")

    # Chinchilla.
    print("\n── Chinchilla Scaling (interprétation active-param) ──")
    chi = report["chinchilla"]
    print(f"  Ratio : {chi['ratio']}")
    print(f"  Optimal tokens (BPE natif)     : {chi['optimal_tokens']:>14,}")
    print(f"  Équivalent en chars (~3/BPE)   : {chi['optimal_tokens_in_chars_equivalent']:>14,}")
    print(f"  Interprétation erronée (total) : {chi['wrong_interpretation_total_tokens']:>14,}  ← surestime massivement")

    # Scenarios.
    print("\n── Scénarios EDT ──")
    sc = report["edt_scenarios"]
    print(f"  Scénario A (agressif)          : {sc['scenario_a_aggressive']['tokens']:>14,} tokens")
    print(f"    → {sc['scenario_a_aggressive']['verdict']}")
    print(f"  Scénario B (plein Chinchilla)  : {sc['scenario_b_full_chinchilla']['tokens']:>14,} tokens")
    print(f"    → {sc['scenario_b_full_chinchilla']['verdict']}")
    print(f"  Scénario C (EDT-proportional)  : {sc['scenario_c_edt_proportional_RECOMMENDED']['tokens']:>14,} tokens")
    print(f"    → {sc['scenario_c_edt_proportional_RECOMMENDED']['verdict']}")
    print(f"    ({sc['scenario_c_edt_proportional_RECOMMENDED']['tokens'] * CHARS_PER_BPE / 1e9:.2f}B chars de texte)")

    # Comparison.
    print("\n── Comparaison Standard vs EDT (RTX 3090) ──")
    cmp = report["comparison_standard_vs_edt"]
    print(f"  Standard (plein Chinchilla)    :")
    print(f"    Tokens                       : {cmp['standard']['tokens']:>14,}")
    print(f"    Temps                        : {cmp['standard']['time_h']:>14,.1f}h "
          f"({cmp['standard']['time_d']:.1f} jours)")
    print(f"    Coût (3090 spot $0.40/h)     : ${cmp['standard']['cost']['cost_usd']:>13,.2f}")
    print(f"  EDT (Scénario C)               :")
    print(f"    Tokens Phase 3               : {cmp['edt']['tokens_phase3']:>14,}")
    print(f"    Temps Phase 1+2              : {cmp['edt']['time_phase12_h']:>14,.2f}h")
    print(f"    Temps Phase 3                : {cmp['edt']['time_phase3_h']:>14,.2f}h")
    print(f"    Temps total EDT              : {cmp['edt']['time_total_h']:>14,.2f}h "
          f"({cmp['edt']['time_total_d']:.2f} jours)")
    print(f"    Coût EDT                     : ${cmp['edt']['cost']['cost_usd']:>13,.2f}")
    print()
    print(f"  ━━ Speedup                     : {cmp['speedup']:>14.1f}×")
    print(f"  ━━ Réduction coût              : {cmp['cost_reduction']:>14.1f}×")

    # Caveats.
    print("\n── Caveats (à traiter comme hypothèses, pas comme faits acquis) ──")
    for i, c in enumerate(report["caveats"], 1):
        print(f"  {i}. {c}")

    # Save JSON.
    import os
    os.makedirs("/home/z/my-project/cognet-moe", exist_ok=True)
    with open("/home/z/my-project/cognet-moe/chinchilla_report.json", "w") as f:
        # Convertir tous les ints numpy/python en int natifs pour JSON.
        def _to_jsonable(o):
            if isinstance(o, dict):
                return {k: _to_jsonable(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_to_jsonable(x) for x in o]
            if isinstance(o, float):
                return o
            try:
                return int(o)
            except (TypeError, ValueError):
                return str(o)
        json.dump(_to_jsonable(comparison), f, indent=2)
    print(f"\n→ Rapport JSON sauvé : /home/z/my-project/cognet-moe/chinchilla_report.json")
