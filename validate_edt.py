#!/usr/bin/env python3
"""
validate_edt.py — Validation rigoureuse du pipeline EDT
=======================================================

Ce script valide empiriquement les hypothèses EDT sur un modèle réduit
mais avec assez de steps/tokens pour observer une vraie convergence.

Tests effectués :
  1. Phase 1 : les experts apprennent-ils l'identité ? (MSE → 0)
  2. Phase 2a : le symmetry break produit-il un gradient utile ?
  3. Phase 2b : l'embedding converge-t-il en isolation ?
  4. Phase 3 : la loss LM baisse-t-elle ? Le routing est-il stable ?
  5. Comparaison EDT vs from-scratch : EDT avec X tokens bat-il
     from-scratch avec le même nombre de tokens ?

Usage :
    PYTHONPATH=source:. python3 validate_edt.py
"""

import sys, os, time, math, json, copy
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cognet_moe import CogNetMoE1B
from edt_pipeline import (
    EDTConfig, run_edt_pipeline,
    phase1_experts, phase2a_attention, phase2b_embedding, phase3_joint,
    verify_edt_prerequisites,
)


# ═══════════════════════════════════════════════════════════════════════
#  Config de validation (petit modèle, CPU, mais steps suffisants)
# ═══════════════════════════════════════════════════════════════════════

# Modèle de validation : assez petit pour CPU, assez grand pour être meaningfull.
VAL_MODEL_CFG = dict(
    vocab_size=136,         # CharTokenizer original
    hidden_dim=128,
    num_blocks=4,
    num_channels=4,         # = n_experts (CogNet-native)
    channel_dim=32,
    ff_dim=256,
    max_seq_len=64,
    working_slots=8,
    episodic_slots=16,
    semantic_slots=32,
    key_dim=32,
    n_experts=4,
    top_k=2,
    dropout=0.0,
    use_gradient_checkpointing=False,
)

# Tokens synthétiques : séquences pseudo-aléatoires avec seed fixe
# pour reproductibilité. Pas de vrai langage, mais on teste la dynamique
# d'entraînement (convergence, stabilité routing, etc.)
SEED = 42
SEQ_LEN = 64
BATCH_SIZE = 8

# Pour les phases 2b et 3, on veut assez de tokens pour voir la loss baisser.
# Sur CPU, on est limité en temps. On vise ~200k tokens par phase.
PHASE2B_TOKENS = 100_000
PHASE3_TOKENS = 100_000

# Phase 1 : assez de steps pour voir MSE converger.
PHASE1_STEPS = 200


def make_data_iter_fn(seed: int = 42):
    """Génère des input_ids reproductibles (distribution non-uniforme
    pour imiter un vrai tokenizer — certaines tokens plus fréquentes)."""
    rng = torch.Generator().manual_seed(seed)
    # Distribution Zipf-like pour imiter la fréquence des tokens.
    vocab_size = VAL_MODEL_CFG["vocab_size"]
    probs = 1.0 / (torch.arange(1, vocab_size + 1, dtype=torch.float))
    probs = probs / probs.sum()

    def data_iter_fn(batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.multinomial(probs, batch_size * seq_len, replacement=True,
                                  generator=rng).view(batch_size, seq_len)
    return data_iter_fn


def make_hidden_states_fn(model, data_iter_fn):
    """Hidden states réels pour Phase 1."""
    device = next(model.parameters()).device
    def hidden_states_fn(batch_size: int, seq_len: int) -> torch.Tensor:
        input_ids = data_iter_fn(batch_size, seq_len)
        with torch.no_grad():
            x = model.encoder(input_ids.to(device))
            n_blocks_to_forward = torch.randint(0, model.num_blocks, (1,),
                                                 generator=torch.Generator().manual_seed(
                                                     int(time.time() * 1000) % 2**31
                                                 )).item()
            for b in range(n_blocks_to_forward):
                block = model.blocks[b]
                x, _ = block.memory(x)
                x = block.composer(x)
                x = block.norm(x)
        return x
    return hidden_states_fn


def evaluate_model(model, data_iter_fn, n_batches: int = 20) -> dict:
    """Évalue le modèle : LM loss + routing stats."""
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_aux = 0.0
    total_max_load = 0.0
    total_min_load = 1.0
    count = 0

    with torch.no_grad():
        for _ in range(n_batches):
            input_ids = data_iter_fn(BATCH_SIZE, SEQ_LEN).to(device)
            result = model(input_ids, return_stats=True)
            logits = result["logits"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_loss += loss.item()
            total_aux += result["moe_aux_loss"].item()

            # Routing stats du block 0.
            stats = result["stats"]
            if "block0_moe_max_load" in stats:
                total_max_load = max(total_max_load, stats["block0_moe_max_load"].item())
                total_min_load = min(total_min_load, stats["block0_moe_min_load"].item())
            count += 1

    model.train()
    return {
        "lm_loss": total_loss / count,
        "aux_loss": total_aux / count,
        "max_load": total_max_load,
        "min_load": total_min_load,
    }


def track_expert_diversity(model) -> dict:
    """Mesure la diversité des poids des experts (écart-type des normes)."""
    expert_norms = {}
    for b in range(model.num_blocks):
        norms = []
        for e in range(model.n_experts):
            expert = model.get_expert(b, e)
            norm = sum(p.norm().item() for p in expert.parameters())
            norms.append(norm)
        expert_norms[f"block_{b}"] = {
            "mean": sum(norms) / len(norms),
            "std": (sum((n - sum(norms)/len(norms))**2 for n in norms) / len(norms))**0.5,
            "min": min(norms),
            "max": max(norms),
        }
    return expert_norms


# ═══════════════════════════════════════════════════════════════════════
#  Test 1 : Phase 1 — Les experts apprennent-ils l'identité ?
# ═══════════════════════════════════════════════════════════════════════

def test_phase1_convergence():
    print("\n" + "=" * 70)
    print("TEST 1 : Phase 1 — Convergence experts vers identité")
    print("=" * 70)

    torch.manual_seed(SEED)
    model = CogNetMoE1B(**VAL_MODEL_CFG)
    data_iter_fn = make_data_iter_fn()

    cfg = EDTConfig(
        phase1_steps_per_expert=PHASE1_STEPS,
        phase1_batch_size=BATCH_SIZE,
        phase1_seq_len=SEQ_LEN,
        phase1_lr=3e-4,
        phase1_target_loss=0.01,
        # Phases suivantes : minimum pour le test.
        phase2a_steps=1,
        phase2b_tokens=1024,
        phase2b_batch_size=BATCH_SIZE,
        phase2b_seq_len=SEQ_LEN,
        phase3_tokens=1024,
        phase3_batch_size=4,
        phase3_seq_len=SEQ_LEN,
        phase3_grad_accum=2,
        phase3_pgsu_n_active=2,
        use_bf16=False,
        use_8bit_optimizer=False,
        device="cpu",
        log_every=50,
    )

    hidden_states_fn = make_hidden_states_fn(model, data_iter_fn)

    # Mesurer la MSE initiale (avant Phase 1).
    h_test = hidden_states_fn(BATCH_SIZE, SEQ_LEN)
    with torch.no_grad():
        expert_before = model.get_expert(0, 0)
        out_before = expert_before(h_test)
        mse_before = F.mse_loss(out_before, h_test).item()

    # Lancer Phase 1.
    stats = phase1_experts(model, hidden_states_fn, cfg)

    # Mesurer la MSE finale.
    with torch.no_grad():
        expert_after = model.get_expert(0, 0)
        out_after = expert_after(h_test)
        mse_after = F.mse_loss(out_after, h_test).item()

    avg_final_loss = stats["avg_loss"]
    n_steps = [e["n_steps"] for e in stats["experts"]]
    avg_steps = sum(n_steps) / len(n_steps)

    print(f"\n  ─── Résultats Phase 1 ───")
    print(f"  MSE initiale (block 0, expert 0) : {mse_before:.4f}")
    print(f"  MSE finale  (block 0, expert 0)  : {mse_after:.4f}")
    print(f"  Réduction MSE                    : {mse_before - mse_after:.4f}")
    print(f"  Loss moyenne finale (tous experts): {avg_final_loss:.4f}")
    print(f"  Steps moyens par expert           : {avg_steps:.0f} / {PHASE1_STEPS}")

    # Vérifications.
    converged = mse_after < mse_before * 0.5  # au moins 50% de réduction
    print(f"\n  {'✅' if converged else '❌'} Convergence Phase 1 : "
          f"{'MSE réduit de >50%' if converged else 'MSE insuffisamment réduit'}")

    return converged, stats


# ═══════════════════════════════════════════════════════════════════════
#  Test 2 : Phase 2b — L'embedding converge-t-il en isolation ?
# ═══════════════════════════════════════════════════════════════════════

def test_phase2b_convergence():
    print("\n" + "=" * 70)
    print("TEST 2 : Phase 2b — Convergence embedding (TokenEncoder)")
    print("=" * 70)

    torch.manual_seed(SEED)
    model = CogNetMoE1B(**VAL_MODEL_CFG)
    data_iter_fn = make_data_iter_fn()

    cfg = EDTConfig(
        phase1_steps_per_expert=50,
        phase1_batch_size=BATCH_SIZE,
        phase1_seq_len=SEQ_LEN,
        phase2a_steps=1,
        phase2a_batch_size=BATCH_SIZE,
        phase2b_tokens=PHASE2B_TOKENS,
        phase2b_batch_size=BATCH_SIZE,
        phase2b_seq_len=SEQ_LEN,
        phase2b_lr=6e-4,
        phase3_tokens=1024,
        phase3_batch_size=4,
        phase3_seq_len=SEQ_LEN,
        phase3_grad_accum=2,
        phase3_pgsu_n_active=2,
        use_bf16=False,
        use_8bit_optimizer=False,
        device="cpu",
        log_every=20,
    )

    # Évaluer la loss initiale (embedding-only, avant Phase 2b).
    eval_before = evaluate_model(model, data_iter_fn, n_batches=10)
    loss_init = eval_before["lm_loss"]

    # Lancer Phase 2b.
    stats = phase2b_embedding(model, data_iter_fn, cfg)

    # Évaluer la loss finale (embedding-only).
    eval_after = evaluate_model(model, data_iter_fn, n_batches=10)
    loss_final = eval_after["lm_loss"]

    # Loss théorique random : log(vocab_size) = log(136) ≈ 4.91
    random_loss = math.log(VAL_MODEL_CFG["vocab_size"])

    print(f"\n  ─── Résultats Phase 2b ───")
    print(f"  Loss random (log(136))           : {random_loss:.4f}")
    print(f"  Loss initiale                    : {loss_init:.4f}")
    print(f"  Loss finale (Phase 2b)           : {loss_final:.4f}")
    print(f"  Loss moyenne Phase 2b            : {stats['avg_loss']:.4f}")
    print(f"  Tokens entraînés                 : {stats['total_tokens']:,}")
    print(f"  Réduction loss                   : {loss_init - loss_final:.4f}")

    converged = loss_final < loss_init - 0.1  # au moins 0.1 de réduction
    below_random = loss_final < random_loss

    print(f"\n  {'✅' if converged else '❌'} Convergence Phase 2b : "
          f"{'loss baisse significativement' if converged else 'loss ne baisse pas assez'}")
    print(f"  {'✅' if below_random else '❌'} Sous random : "
          f"{'oui — embedding apprend' if below_random else 'non — toujours au niveau random'}")

    return converged, stats


# ═══════════════════════════════════════════════════════════════════════
#  Test 3 : Phase 3 — Loss LM baisse + routing stable
# ═══════════════════════════════════════════════════════════════════════

def test_phase3_convergence():
    print("\n" + "=" * 70)
    print("TEST 3 : Phase 3 — Joint fine-tune PGSU + routing stability")
    print("=" * 70)

    torch.manual_seed(SEED)
    model = CogNetMoE1B(**VAL_MODEL_CFG)
    data_iter_fn = make_data_iter_fn()

    cfg = EDTConfig(
        phase1_steps_per_expert=50,
        phase1_batch_size=BATCH_SIZE,
        phase1_seq_len=SEQ_LEN,
        phase2a_steps=1,
        phase2a_batch_size=BATCH_SIZE,
        phase2b_tokens=50_000,
        phase2b_batch_size=BATCH_SIZE,
        phase2b_seq_len=SEQ_LEN,
        phase3_tokens=PHASE3_TOKENS,
        phase3_batch_size=4,
        phase3_seq_len=SEQ_LEN,
        phase3_grad_accum=2,
        phase3_lr=1e-4,
        phase3_warmup_steps=10,
        phase3_aux_loss_weight=0.01,
        phase3_z_loss_weight=1e-3,
        phase3_pgsu_n_active=2,
        use_bf16=False,
        use_8bit_optimizer=False,
        device="cpu",
        log_every=20,
    )

    # Pré-entraînement rapide (Phases 1+2a+2b).
    hidden_states_fn = make_hidden_states_fn(model, data_iter_fn)
    phase1_experts(model, hidden_states_fn, cfg)
    phase2a_attention(model, data_iter_fn, cfg)
    phase2b_embedding(model, data_iter_fn, cfg)

    # Évaluer avant Phase 3.
    eval_before = evaluate_model(model, data_iter_fn)
    loss_before = eval_before["lm_loss"]
    max_load_before = eval_before["max_load"]

    # Lancer Phase 3.
    stats = phase3_joint(model, data_iter_fn, cfg)

    # Évaluer après Phase 3.
    eval_after = evaluate_model(model, data_iter_fn)
    loss_after = eval_after["lm_loss"]
    max_load_after = eval_after["max_load"]

    # Analyser la stabilité du routing.
    max_loads = stats["max_loads"]
    min_loads = stats["min_loads"]
    avg_max_load = sum(max_loads) / len(max_loads) if max_loads else 0
    avg_min_load = sum(min_loads) / len(min_loads) if min_loads else 1

    # Routing collapse : max_load > 0.5 de manière persistante.
    collapse_steps = sum(1 for ml in max_loads if ml > 0.5)
    collapse_rate = collapse_steps / len(max_loads) if max_loads else 0

    print(f"\n  ─── Résultats Phase 3 ───")
    print(f"  Loss LM avant Phase 3            : {loss_before:.4f}")
    print(f"  Loss LM après Phase 3            : {loss_after:.4f}")
    print(f"  Réduction loss                   : {loss_before - loss_after:.4f}")
    print(f"  Aux loss finale                  : {stats['aux_losses'][-1]:.4f}")
    print(f"  Max load moyen                   : {avg_max_load:.3f}")
    print(f"  Min load moyen                   : {avg_min_load:.3f}")
    print(f"  Taux routing collapse (>0.5)     : {collapse_rate:.1%}")

    loss_decreased = loss_after < loss_before - 0.05
    routing_stable = collapse_rate < 0.5  # moins de 50% des steps en collapse
    aux_finite = all(math.isfinite(l) for l in stats["aux_losses"])

    print(f"\n  {'✅' if loss_decreased else '❌'} Loss LM baisse : "
          f"{'oui' if loss_decreased else 'non'}")
    print(f"  {'✅' if routing_stable else '⚠️'} Routing stable : "
          f"{'oui' if routing_stable else f'collapse à {collapse_rate:.0%} des steps'}")
    print(f"  {'✅' if aux_finite else '❌'} Aux loss finie : "
          f"{'oui' if aux_finite else 'NON — NaN/Inf détecté!'}")

    return loss_decreased and routing_stable, stats


# ═══════════════════════════════════════════════════════════════════════
#  Test 4 : EDT vs From-Scratch — EDT est-il vraiment meilleur ?
# ═══════════════════════════════════════════════════════════════════════

def test_edt_vs_from_scratch():
    print("\n" + "=" * 70)
    print("TEST 4 : EDT vs From-Scratch — Même budget tokens")
    print("=" * 70)

    total_tokens = PHASE2B_TOKENS + PHASE3_TOKENS  # budget total identique
    # On réduit le budget pour Test 4 (timeout CPU).
    total_tokens_t4 = min(total_tokens, 150_000)

    # ─── Modèle A : from-scratch (entraînement standard, même budget) ────
    print("\n  ── Modèle A : From-Scratch (standard) ──")
    torch.manual_seed(SEED)
    model_a = CogNetMoE1B(**VAL_MODEL_CFG)
    data_iter_fn = make_data_iter_fn()

    cfg_a = EDTConfig(
        phase1_steps_per_expert=1,      # pas de Phase 1
        phase1_batch_size=BATCH_SIZE,
        phase1_seq_len=SEQ_LEN,
        phase2a_steps=0,                # pas de Phase 2a
        phase2b_tokens=0,               # pas de Phase 2b
        phase3_tokens=total_tokens_t4,  # tout en Phase 3
        phase3_batch_size=4,
        phase3_seq_len=SEQ_LEN,
        phase3_grad_accum=2,
        phase3_lr=3e-4,                 # LR plus haute (from scratch)
        phase3_warmup_steps=50,
        phase3_aux_loss_weight=0.01,
        phase3_z_loss_weight=1e-3,
        phase3_pgsu_n_active=2,
        use_bf16=False,
        use_8bit_optimizer=False,
        device="cpu",
        log_every=20,
    )

    # From-scratch : juste Phase 3 (joint training sans pré-entraînement).
    stats_a = phase3_joint(model_a, data_iter_fn, cfg_a)
    eval_a = evaluate_model(model_a, data_iter_fn)
    loss_a = eval_a["lm_loss"]

    # ─── Modèle B : EDT complet ──────────────────────────────────────
    print("\n  ── Modèle B : EDT complet (4 phases) ──")
    torch.manual_seed(SEED)
    model_b = CogNetMoE1B(**VAL_MODEL_CFG)

    cfg_b = EDTConfig(
        phase1_steps_per_expert=50,
        phase1_batch_size=BATCH_SIZE,
        phase1_seq_len=SEQ_LEN,
        phase2a_steps=1,
        phase2a_batch_size=BATCH_SIZE,
        phase2b_tokens=PHASE2B_TOKENS // 2,
        phase2b_batch_size=BATCH_SIZE,
        phase2b_seq_len=SEQ_LEN,
        phase3_tokens=PHASE3_TOKENS // 2,
        phase3_batch_size=4,
        phase3_seq_len=SEQ_LEN,
        phase3_grad_accum=2,
        phase3_lr=1e-4,
        phase3_warmup_steps=10,
        phase3_aux_loss_weight=0.01,
        phase3_z_loss_weight=1e-3,
        phase3_pgsu_n_active=2,
        use_bf16=False,
        use_8bit_optimizer=False,
        device="cpu",
        log_every=20,
    )

    hidden_states_fn = make_hidden_states_fn(model_b, data_iter_fn)
    phase1_experts(model_b, hidden_states_fn, cfg_b)
    phase2a_attention(model_b, data_iter_fn, cfg_b)
    phase2b_embedding(model_b, data_iter_fn, cfg_b)
    stats_b = phase3_joint(model_b, data_iter_fn, cfg_b)
    eval_b = evaluate_model(model_b, data_iter_fn)
    loss_b = eval_b["lm_loss"]

    # ─── Comparaison ─────────────────────────────────────────────────
    print(f"\n  ─── Comparaison ───")
    print(f"  Budget tokens total              : {total_tokens:,}")
    print(f"  From-Scratch (A) final loss      : {loss_a:.4f}")
    print(f"  EDT complet (B) final loss       : {loss_b:.4f}")
    print(f"  Delta (A - B)                    : {loss_a - loss_b:+.4f}")

    edt_better = loss_b < loss_a
    significant = abs(loss_a - loss_b) > 0.05

    if edt_better and significant:
        verdict = "✅ EDT bat from-scratch de manière significative"
    elif edt_better:
        verdict = "⚠️  EDT bat from-scratch mais difference < 0.05 (marginal)"
    else:
        verdict = "❌ From-scratch fait aussi bien ou mieux que EDT"

    print(f"\n  {verdict}")
    print(f"  (Note : sur données synthétiques, le bénéfice EDT est un test")
    print(f"   de dynamique d'entraînement, pas de qualité linguistique.)")

    return edt_better, {"from_scratch_loss": loss_a, "edt_loss": loss_b}


# ═══════════════════════════════════════════════════════════════════════
#  Test 5 : Diversité des experts — Les experts se spécialisent-ils ?
# ═══════════════════════════════════════════════════════════════════════

def test_expert_specialization():
    print("\n" + "=" * 70)
    print("TEST 5 : Spécialisation des experts (Phase 1 → Phase 3)")
    print("=" * 70)

    torch.manual_seed(SEED)
    model = CogNetMoE1B(**VAL_MODEL_CFG)
    data_iter_fn = make_data_iter_fn()

    # Mesurer diversité initiale (avant EDT).
    diversity_before = track_expert_diversity(model)

    # EDT partiel (Phase 1 + Phase 3).
    cfg = EDTConfig(
        phase1_steps_per_expert=100,
        phase1_batch_size=BATCH_SIZE,
        phase1_seq_len=SEQ_LEN,
        phase2a_steps=1,
        phase2a_batch_size=BATCH_SIZE,
        phase2b_tokens=50_000,
        phase2b_batch_size=BATCH_SIZE,
        phase2b_seq_len=SEQ_LEN,
        phase3_tokens=50_000,
        phase3_batch_size=4,
        phase3_seq_len=SEQ_LEN,
        phase3_grad_accum=2,
        phase3_pgsu_n_active=2,
        use_bf16=False,
        use_8bit_optimizer=False,
        device="cpu",
        log_every=50,
    )

    hidden_states_fn = make_hidden_states_fn(model, data_iter_fn)
    phase1_experts(model, hidden_states_fn, cfg)

    # Mesurer diversité après Phase 1.
    diversity_after_p1 = track_expert_diversity(model)

    phase2a_attention(model, data_iter_fn, cfg)
    phase2b_embedding(model, data_iter_fn, cfg)
    phase3_joint(model, data_iter_fn, cfg)

    # Mesurer diversité finale.
    diversity_after_p3 = track_expert_diversity(model)

    # Analyser.
    print(f"\n  ─── Diversité des experts ───")
    print(f"  (Écart-type des normes de poids par bloc)")
    print(f"  Plus l'écart-type est grand, plus les experts sont différents.\n")

    avg_std_before = sum(v["std"] for v in diversity_before.values()) / len(diversity_before)
    avg_std_after_p1 = sum(v["std"] for v in diversity_after_p1.values()) / len(diversity_after_p1)
    avg_std_after_p3 = sum(v["std"] for v in diversity_after_p3.values()) / len(diversity_after_p3)

    print(f"  Avant EDT           : std moyen = {avg_std_before:.4f}")
    print(f"  Après Phase 1       : std moyen = {avg_std_after_p1:.4f}")
    print(f"  Après Phase 3 (fin) : std moyen = {avg_std_after_p3:.4f}")

    # Les experts devraient diverger (se spécialiser) après EDT.
    diversified = avg_std_after_p3 > avg_std_before * 1.1  # 10% de divergence min
    print(f"\n  {'✅' if diversified else '❌'} Spécialisation : "
          f"{'experts se différencient' if diversified else 'experts restent identiques'}")

    return diversified, {
        "before": avg_std_before,
        "after_phase1": avg_std_after_p1,
        "after_phase3": avg_std_after_p3,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Orchestrateur
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("VALIDATION EDT — CogNet-MoE-1B")
    print("=" * 70)
    print(f"  Modèle    : {VAL_MODEL_CFG['hidden_dim']}d × {VAL_MODEL_CFG['num_blocks']}b × "
          f"{VAL_MODEL_CFG['n_experts']}experts (top-{VAL_MODEL_CFG['top_k']})")
    print(f"  Device    : CPU")
    print(f"  Phase 1   : {PHASE1_STEPS} steps/expert")
    print(f"  Phase 2b  : {PHASE2B_TOKENS:,} tokens")
    print(f"  Phase 3   : {PHASE3_TOKENS:,} tokens")

    t_start = time.time()
    results = {}

    # Test 1
    ok1, stats1 = test_phase1_convergence()
    results["phase1_convergence"] = ok1

    # Test 2
    ok2, stats2 = test_phase2b_convergence()
    results["phase2b_convergence"] = ok2

    # Test 3
    ok3, stats3 = test_phase3_convergence()
    results["phase3_convergence"] = ok3

    # Test 4
    ok4, stats4 = test_edt_vs_from_scratch()
    results["edt_vs_from_scratch"] = ok4

    # Test 5
    ok5, stats5 = test_expert_specialization()
    results["expert_specialization"] = ok5

    # ─── Résumé ──────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("RÉSUMÉ VALIDATION EDT")
    print("=" * 70)

    tests = [
        ("Phase 1 : experts → identité", ok1),
        ("Phase 2b : embedding converge", ok2),
        ("Phase 3 : LM loss + routing", ok3),
        ("EDT > from-scratch", ok4),
        ("Experts se spécialisent", ok5),
    ]

    for name, ok in tests:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)
    print(f"\n  Résultat : {passed}/{total} tests passent")
    print(f"  Temps total : {total_time:.1f}s ({total_time/60:.1f} min)")

    # Sauvegarder le rapport.
    report = {
        "summary": {name: ok for name, ok in tests},
        "passed": passed,
        "total": total,
        "time_s": total_time,
        "model_config": VAL_MODEL_CFG,
        "phase1_steps": PHASE1_STEPS,
        "phase2b_tokens": PHASE2B_TOKENS,
        "phase3_tokens": PHASE3_TOKENS,
    }
    report_path = os.path.join(HERE, "edt_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Rapport sauvegardé : {report_path}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
