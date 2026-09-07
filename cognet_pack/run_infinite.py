#!/usr/bin/env python3
"""
run_infinite.py — Entraînement INFINI de CogNet-MoE (Phase-Routed MoE)
======================================================================

Le `.pt` final reste entraînable à l'infini : chaque nouvelle phase de données
crée de nouveaux experts à la volée (anciens gelés, savoir préservé).

Pipeline :
  1. Pré-tokeniser chaque phase en .bin (une fois) :
       python3 fast_train.py --build-bin --txt phase0.txt --tokenizer cognet_tokenizer.json --out data/p0
       python3 fast_train.py --build-bin --txt phase1.txt --tokenizer cognet_tokenizer.json --out data/p1
  2. Lancer l'entraînement infini :
       python3 run_infinite.py --bins data/p0.bin,data/p1.bin --tokens-per-phase 1000000000
  3. Ajouter une phase plus tard (reprise + croissance) :
       python3 run_infinite.py --bins data/p0.bin,data/p1.bin,data/p2.bin \\
           --resume infinite_ckpts/final.pt --start-phase 2 --tokens-per-phase 1000000000

Le checkpoint final `infinite_ckpts/final.pt` contient poids + croissance + arch :
il se recharge et se prolonge avec `load_infinite_checkpoint()` (phase_routed_moe.py).

Usage test (CPU, ~1-2 min) :
    python3 run_infinite.py --self-test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from cognet_moe import CogNetMoE1B, create_cognet_moe_1b  # noqa: E402
from phase_routed_moe import (  # noqa: E402
    GrowthConfig, InfiniteConfig, InfiniteTrainer, convert_to_phase_routed,
    load_infinite_checkpoint,
)
from fast_train import MixedPhaseLoader, enable_fast_mode  # noqa: E402


def build_tiny_model(vocab_size: int = 512) -> CogNetMoE1B:
    m = CogNetMoE1B(vocab_size=vocab_size, hidden_dim=128, num_blocks=4,
                    num_channels=8, channel_dim=32, ff_dim=256, max_seq_len=128,
                    working_slots=8, episodic_slots=16, semantic_slots=32,
                    key_dim=32, n_experts=8, top_k=2, use_gradient_checkpointing=False)
    return convert_to_phase_routed(m, max_experts=16, max_phases=16)


def main():
    ap = argparse.ArgumentParser(description="CogNet-MoE entraînement infini (Phase-Routed MoE)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bins", type=str, help="liste .bin séparés par des virgules (1 par phase)")
    ap.add_argument("--tokens-per-phase", type=float, default=1_000_000_000)
    ap.add_argument("--start-phase", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None, help="final.pt à prolonger")
    # Modèle.
    ap.add_argument("--tiny", action="store_true", help="modèle tiny (debug)")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--max-experts", type=int, default=32)
    ap.add_argument("--max-phases", type=int, default=64)
    # Training.
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optimizer", type=str, default="adamw-fused")
    ap.add_argument("--compile", type=str, default=None, help="None|reduce-overhead|max-autotune")
    ap.add_argument("--no-bf16", action="store_true")
    # Lifelong.
    ap.add_argument("--new-experts-per-phase", type=int, default=2)
    ap.add_argument("--no-freeze", action="store_true", help="ne PAS geler les anciens experts")
    ap.add_argument("--replay-ratio", type=float, default=0.03)
    ap.add_argument("--no-mid-growth", action="store_true")
    ap.add_argument("--phase-bias-init", type=float, default=1.0)
    ap.add_argument("--ckpt-dir", type=str, default="./infinite_ckpts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.self_test or not args.bins:
        self_test()
        return

    enable_fast_mode()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RunInfinite] device={device}")
    bin_paths = [b.strip() for b in args.bins.split(",") if b.strip()]
    assert bin_paths, "--bins requis"
    for b in bin_paths:
        assert os.path.exists(b), f"bin introuvable: {b}"

    # ─── Modèle : fresh ou resume ────────────────────────────────────
    if args.resume:
        print(f"[RunInfinite] reprise depuis {args.resume} + prolongation")
        model, meta = load_infinite_checkpoint(args.resume, device="cpu")
        print(f"[RunInfinite] repris : phase={meta['phase_id']}, "
              f"tokens={meta['tokens_seen_total']:,}, C="
              f"{[r.n_experts for r in [b.cognitive_expert_router for b in model.blocks]]}")
        start_phase = args.start_phase or (meta["phase_id"] + 1)
    else:
        loader0 = MixedPhaseLoader([bin_paths[0]], current_idx=0, seed=args.seed)
        vocab = loader0.bins[0].vocab_size
        print(f"[RunInfinite] vocab dataset = {vocab}")
        if args.tiny:
            model = build_tiny_model(vocab)
        else:
            model = create_cognet_moe_1b(vocab_size=vocab, max_seq_len=args.max_seq_len)
            model = convert_to_phase_routed(model, max_experts=args.max_experts,
                                            max_phases=args.max_phases)
        p = model.count_parameters()
        print(f"[RunInfinite] total={p['total']:,} actif/token≈{p['active_per_token']:,}")
        start_phase = args.start_phase

    # ─── Trainer ─────────────────────────────────────────────────────
    growth = GrowthConfig(max_experts=args.max_experts, max_phases=args.max_phases,
                          new_experts_per_phase=args.new_experts_per_phase,
                          enable_mid_phase_growth=not args.no_mid_growth)
    cfg = InfiniteConfig(lr=args.lr, batch_size=args.batch_size, seq_len=args.seq_len,
                         grad_accum=args.grad_accum, use_bf16=not args.no_bf16,
                         compile_mode=args.compile, optimizer=args.optimizer,
                         growth=growth, freeze_old_experts=not args.no_freeze,
                         replay_ratio=args.replay_ratio,
                         new_phase_bias_init=args.phase_bias_init,
                         mid_phase_growth=not args.no_mid_growth, seed=args.seed)
    trainer = InfiniteTrainer(model, cfg, device=device, ckpt_dir=args.ckpt_dir)
    if args.resume:
        trainer.tokens_seen_total = meta["tokens_seen_total"]
        trainer.growth_history = meta.get("growth_history", [])

    # ─── Boucle phases ───────────────────────────────────────────────
    loader = MixedPhaseLoader(bin_paths, current_idx=start_phase,
                              replay_ratio=args.replay_ratio, seed=args.seed)
    last_usages = None
    t_all = time.time()
    for phase in range(start_phase, len(bin_paths)):
        loader.set_current(phase)
        print(f"\n{'#'*70}\n# PHASE {phase} : {bin_paths[phase]}\n{'#'*70}")
        pin = device.startswith("cuda")
        stats = trainer.train_phase(
            batch_fn=lambda: loader.next_batch(args.batch_size, args.seq_len,
                                               device="cpu", pin_memory=False),
            phase_id=phase,
            phase_tokens=int(args.tokens_per_phase),
            last_usage_per_block=last_usages,
        )
        last_usages = stats["last_usages"]
        print(f"[RunInfinite] phase {phase} ✓ {stats['tokens']/1e6:.1f}M tokens "
              f"en {stats['time_s']/3600:.2f}h ({stats['tok_s']:.0f} tok/s)")

    dt = time.time() - t_all
    print(f"\n{'='*70}\n[RunInfinite] TERMINÉ : {trainer.tokens_seen_total/1e9:.3f}B tokens "
          f"en {dt/3600:.2f}h — final.pt prolongeable à l'infini : {args.ckpt_dir}/final.pt\n"
          f"C final={[r.n_experts for r in trainer.routers()]}\n{'='*70}")
    with open(os.path.join(args.ckpt_dir, "growth_history.json"), "w") as f:
        json.dump(trainer.growth_history, f, indent=2, default=str)


def self_test():
    """Test CPU bout-en-bout : 2 phases tiny, croissance, freeze, save/resume."""
    print("=" * 70)
    print("RunInfinite — Self-test bout-en-bout (CPU)")
    print("=" * 70)
    import tempfile
    import numpy as np
    from fast_train import build_bin_dataset
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp()

    # 1. Deux mini-bins (fallback byte-level, pas de tokenizer requis).
    print("\n[1/4] Bins factices...")
    bins = []
    for i, txt_seed in enumerate(["alpha beta gamma delta ", "un deux trois quatre "]):
        txt = os.path.join(tmp, f"p{i}.txt")
        with open(txt, "w") as f:
            f.write(txt_seed * 3000)
        r = build_bin_dataset(txt, os.path.join(tmp, f"p{i}"), tokenizer=None)
        bins.append(r["bin_path"])
    print(f"  ✓ {[os.path.basename(b) for b in bins]}")

    # 2. Trainer tiny, 2 phases.
    print("\n[2/4] Entraînement 2 phases...")
    model = CogNetMoE1B(vocab_size=512, hidden_dim=32, num_blocks=2, num_channels=4,
                        channel_dim=16, ff_dim=64, max_seq_len=32, working_slots=2,
                        episodic_slots=4, semantic_slots=8, key_dim=16,
                        n_experts=4, top_k=2, use_gradient_checkpointing=False)
    model = convert_to_phase_routed(model, max_experts=8, max_phases=8)
    growth = GrowthConfig(max_experts=8, max_phases=8, new_experts_per_phase=1,
                          min_steps_between_growth=0, overload_patience=1000,
                          confidence_low=0.0)
    cfg = InfiniteConfig(lr=3e-4, batch_size=4, seq_len=32, grad_accum=1,
                         use_bf16=False, compile_mode=None, log_every=2,
                         ckpt_every_steps=100, growth=growth, freeze_old_experts=True,
                         replay_ratio=0.25, mid_phase_growth=False)
    trainer = InfiniteTrainer(model, cfg, device="cpu", ckpt_dir=os.path.join(tmp, "ckpts"))
    loader = MixedPhaseLoader(bins, current_idx=0, replay_ratio=0.25, seed=0)

    loader.set_current(0)
    s0 = trainer.train_phase(lambda: loader.next_batch(4, 32, device="cpu"),
                             phase_id=0, phase_tokens=256, phase_total_steps=2)
    c0 = [r.n_experts for r in trainer.routers()]
    loader.set_current(1)
    s1 = trainer.train_phase(lambda: loader.next_batch(4, 32, device="cpu"),
                             phase_id=1, phase_tokens=256, phase_total_steps=2,
                             last_usage_per_block=s0["last_usages"])
    c1 = [r.n_experts for r in trainer.routers()]
    assert c0 == [4, 4] and c1 == [5, 5], f"C inattendu: {c0} → {c1}"
    print(f"  ✓ C: {c0} → {c1}, replay+freeze OK")

    # 3. Reprise + phase 3 (extensibilité du .pt).
    print("\n[3/4] Reprise du final.pt + nouvelle phase...")
    final = os.path.join(tmp, "ckpts", "final.pt")
    assert os.path.exists(final)
    model_r, meta = load_infinite_checkpoint(final, device="cpu")
    assert meta["phase_id"] == 1
    trainer2 = InfiniteTrainer(model_r, cfg, device="cpu", ckpt_dir=os.path.join(tmp, "ckpts2"))
    loader.set_current(1)
    s2 = trainer2.train_phase(lambda: loader.next_batch(4, 32, device="cpu"),
                              phase_id=2, phase_tokens=256, phase_total_steps=2)
    c2 = [r.n_experts for r in trainer2.routers()]
    assert c2 == [6, 6], f"C inattendu après reprise: {c2}"
    print(f"  ✓ reprise OK, C: {c1} → {c2} (infini ✓)")

    # 4. Pas de catastrophic forgetting grossier : les experts gelés n'ont pas bougé.
    print("\n[4/4] Stabilité des experts gelés...")
    w_before = model.blocks[0].cognitive_expert_router.experts[0].w_gate_up.weight.detach().clone()
    # (model a déjà fini phase 1 avec experts 0-3 gelés ; on refait 2 steps phase 1
    #  et on vérifie que l'expert 0 n'a pas bougé.)
    loader.set_current(1)
    for _ in range(2):
        ids, pids = loader.next_batch(4, 32, device="cpu")
        trainer.train_step(ids, pids, 2)
    w_after = model.blocks[0].cognitive_expert_router.experts[0].w_gate_up.weight.detach()
    assert torch.equal(w_before, w_after), "expert gelé a bougé — anti-oubli cassé!"
    print("  ✓ experts gelés bit-identiques après entraînement (anti-oubli ✓)")

    print("\n" + "=" * 70)
    print("✓ Self-test RunInfinite passé : lifelong + croissance + anti-oubli OK.")
    print("=" * 70)


if __name__ == "__main__":
    main()
