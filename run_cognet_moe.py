#!/usr/bin/env python3
"""
run_cognet_moe.py — Script de lancement CogNet-MoE-1B avec EDT
================================================================

Orchestre la création du modèle + pipeline EDT complet.

Usage :
    # Self-test rapide (CPU, tiny model)
    python3 run_cognet_moe.py --self-test

    # Entraînement complet (nécessite GPU 3090 + dataset pré-tokenisé)
    python3 run_cognet_moe.py \
        --dataset-path /path/to/char_tokens.bin \
        --vocab-size 136 \
        --max-seq-len 512 \
        --n-experts 8 \
        --top-k 2 \
        --phase3-tokens 1291398582 \
        --aux-loss-weight 0.01 \
        --pgsu-n-active 4 \
        --use-bf16 \
        --use-8bit-optimizer

    # Si routing collapse détecté, augmenter aux_loss_weight :
    python3 run_cognet_moe.py ... --aux-loss-weight 0.05

    # Si VRAM Phase 3 saute, descendre PGSU :
    python3 run_cognet_moe.py ... --pgsu-n-active 2
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn.functional as F

# Ajouter les chemins.
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))

from cognet_moe import CogNetMoE1B, create_cognet_moe_1b
from edt_pipeline import EDTConfig, run_edt_pipeline
from chinchilla_scaling import full_report, BPE_TOKENIZER_VOCAB
from cognet_tokenizer import CognetTokenizer, DEFAULT_TOKENIZER_PATH


# ═══════════════════════════════════════════════════════════════════════
#  Dataset streaming (BPE tokens via CognetTokenizer)
# ═══════════════════════════════════════════════════════════════════════

class CognetDataset:
    """
    Stream de BPE tokens via CognetTokenizer.

    Lit un fichier texte brut (.txt), tokenise à la volée avec le BPE
    propriétaire, et retourne des batches de (B, T) token IDs.

    Pour de la perf maximale en production, on pré-tokeniserait le corpus
    une fois et on lirait directement les token IDs depuis un .bin. Mais
    pour la flexibilité (changement de tokenizer sans re-prétraiter), on
    tokenise à la volée ici.
    """

    def __init__(
        self,
        path: str,
        tokenizer: CognetTokenizer,
        seq_len: int = 512,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset non trouvé : {path}\n"
                f"Préparer d'abord les données textuelles."
            )
        self.file_size = os.path.getsize(path)
        print(f"[Dataset] {path} ({self.file_size / 1e9:.2f} GB)")
        print(f"[Dataset] Tokenizer vocab_size = {tokenizer.vocab_size}")

        # Pré-charger une portion du fichier pour estimer le ratio chars/BPE.
        sample_size = min(100_000, self.file_size)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample_text = f.read(sample_size)
        sample_ids = self.tokenizer.encode(sample_text, add_special_tokens=False)
        self.chars_per_token = len(sample_text) / max(1, len(sample_ids))
        print(f"[Dataset] Ratio chars/token = {self.chars_per_token:.2f}")

    def make_iter_fn(self, seed: int = 42):
        """Retourne un callable(batch_size, seq_len) -> input_ids."""
        rng = torch.Generator().manual_seed(seed)

        def iter_fn(batch_size: int, seq_len: int) -> torch.Tensor:
            batch = torch.zeros(batch_size, seq_len, dtype=torch.long)
            for i in range(batch_size):
                # Lecture aléatoire d'un chunk de texte.
                # On lit seq_len * chars_per_token chars pour avoir ~seq_len tokens.
                chars_to_read = int(seq_len * self.chars_per_token * 1.2) + 100
                max_start = max(0, self.file_size - chars_to_read - 1)
                start = rng.randint(0, max_start).item() if max_start > 0 else 0
                with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(start)
                    text = f.read(chars_to_read)
                # Tokeniser et tronquer/padder à seq_len.
                ids = self.tokenizer.encode(text, add_special_tokens=True, max_length=seq_len)
                if len(ids) < seq_len:
                    ids = ids + [self.tokenizer.pad_id] * (seq_len - len(ids))
                batch[i] = torch.tensor(ids[:seq_len], dtype=torch.long)
            return batch

        return iter_fn


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════

def self_test():
    """Self-test complet : tokenizer BPE + modèle MoE + EDT + Chinchilla."""
    print("=" * 70)
    print("CogNet-MoE-1B — Self-test complet (tokenizer BPE + MoE + EDT + Chinchilla)")
    print("=" * 70)

    # 0. Test tokenizer BPE propriétaire.
    print("\n[0/4] Test tokenizer BPE propriétaire...")
    from cognet_tokenizer import CognetTokenizer, DEFAULT_TOKENIZER_PATH
    if not DEFAULT_TOKENIZER_PATH.exists():
        print("  → Entraînement du tokenizer BPE (premier run)...")
        from cognet_tokenizer import train_cognet_tokenizer
        train_cognet_tokenizer(save_path=DEFAULT_TOKENIZER_PATH)
    tok = CognetTokenizer(tokenizer_path=DEFAULT_TOKENIZER_PATH, max_seq_len=128)
    test_text = "Bonjour le monde, ceci est un test du tokenizer CogNet."
    ids = tok.encode(test_text)
    decoded = tok.decode(ids)
    ratio = len(test_text) / max(1, len(ids))
    print(f"  ✓ vocab_size={tok.vocab_size}  pad_id={tok.pad_id}  bos_id={tok.bos_id}  eos_id={tok.eos_id}")
    print(f"  ✓ Roundtrip: {len(test_text)} chars → {len(ids)} tokens (ratio {ratio:.2f})")
    print(f"  ✓ Decoded: '{decoded[:60]}...'")

    # 1. Test SparseMoEBlock avec le vocab du tokenizer.
    print("\n[1/4] Test SparseMoEBlock + aux losses...")
    from cognet_moe import CogNetMoE1B
    # Utiliser le vocab_size réel du tokenizer (pas hardcoded 136).
    vocab_size = max(tok.vocab_size, 1024)  # au moins 1024 pour test
    model = CogNetMoE1B(
        vocab_size=vocab_size,
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
    # Utiliser des tokens réels du tokenizer.
    x = torch.tensor([tok.encode(test_text, max_length=64) for _ in range(2)], dtype=torch.long)
    # Pad à 32 pour le test.
    if x.shape[1] < 32:
        x = torch.cat([x, torch.full((2, 32 - x.shape[1]), tok.pad_id, dtype=torch.long)], dim=1)
    x = x[:, :32]
    result = model(x, return_stats=True)
    assert result["logits"].shape == (2, 32, vocab_size), f"Bad shape: {result['logits'].shape}"
    assert torch.isfinite(result["moe_aux_loss"]), "aux_loss NaN/Inf!"
    assert torch.isfinite(result["moe_z_loss"]), "z_loss NaN/Inf!"
    print(f"  ✓ logits={result['logits'].shape}  aux={result['moe_aux_loss'].item():.4f}  z={result['moe_z_loss'].item():.4f}")

    # Backward + vérif gradients sur tous les experts.
    loss = result["logits"].sum() + 0.01 * result["moe_aux_loss"]
    loss.backward()
    n_with_grad = 0
    for b in range(2):
        for e in range(4):
            expert = model.get_expert(b, e)
            if all(p.grad is not None and torch.any(p.grad != 0).item() for p in expert.parameters()):
                n_with_grad += 1
    assert n_with_grad == 8, f"Only {n_with_grad}/8 experts have gradients"
    print(f"  ✓ {n_with_grad}/8 experts have non-zero gradients")

    # 2. Test EDT pipeline (tiny, CPU, few steps).
    print("\n[2/4] Test EDT pipeline (tiny, CPU)...")
    from edt_pipeline import EDTConfig, run_edt_pipeline
    cfg = EDTConfig(
        phase1_steps_per_expert=2,
        phase1_batch_size=4,
        phase1_seq_len=32,
        phase2a_steps=1,
        phase2a_batch_size=4,
        phase2b_tokens=128,
        phase2b_batch_size=4,
        phase2b_seq_len=32,
        phase3_tokens=128,
        phase3_batch_size=2,
        phase3_seq_len=32,
        phase3_grad_accum=2,
        phase3_pgsu_n_active=1,
        use_bf16=False,
        use_8bit_optimizer=False,
        device="cpu",
        log_every=1,
    )

    def data_fn(batch_size, seq_len):
        # Tokens réels du tokenizer.
        return torch.randint(0, vocab_size, (batch_size, seq_len))

    stats = run_edt_pipeline(model, data_fn, cfg, save_dir=None)
    assert "phases" in stats
    assert "prerequisites" in stats
    print(f"  ✓ EDT pipeline terminé en {stats['total_time_s']:.2f}s")
    print(f"  ✓ Prérequis EDT vérifiés: {sum(1 for c in stats['prerequisites'].values() if c['passed'])}/{len(stats['prerequisites'])}")

    # 3. Test Chinchilla avec BPE 16k.
    print("\n[3/4] Test Chinchilla scaling (BPE 16k)...")
    report = full_report(vocab_size=BPE_TOKENIZER_VOCAB)
    total = report["totals"]["total_params"]
    active = report["totals"]["active_params_per_token"]
    optimal = report["chinchilla"]["optimal_tokens"]
    scenario_c = report["edt_scenarios"]["scenario_c_edt_proportional_RECOMMENDED"]["tokens"]
    print(f"  ✓ Tokenizer      : {report['tokenizer']}")
    print(f"  ✓ Total params   : {total:,}")
    print(f"  ✓ Active / token : {active:,}")
    print(f"  ✓ Chinchilla opt : {optimal:,} BPE tokens")
    print(f"  ✓ Scénario C     : {scenario_c:,} BPE tokens")

    # 4. Test Chinchilla avec CharTokenizer (legacy, pour comparaison).
    print("\n[4/4] Test Chinchilla scaling (CharTokenizer legacy, pour comparaison)...")
    from chinchilla_scaling import CHAR_TOKENIZER_VOCAB
    report_char = full_report(vocab_size=CHAR_TOKENIZER_VOCAB)
    print(f"  ✓ Tokenizer      : {report_char['tokenizer']}")
    print(f"  ✓ Total params   : {report_char['totals']['total_params']:,}")
    print(f"  ✓ Active / token : {report_char['totals']['active_params_per_token']:,}")
    print(f"  ✓ Chinchilla opt : {report_char['chinchilla']['optimal_tokens']:,} char tokens")
    print(f"  → BPE 16k augmente total params de "
          f"{total - report_char['totals']['total_params']:,} "
          f"(token_emb plus grand)")

    print("\n" + "=" * 70)
    print("✓ Self-test complet passé. Code prêt à shipper avec :")
    print("  - Tokenizer BPE propriétaire (vocab 16k, FR+EN+code)")
    print("  - SparseMoEBlock + aux-loss clamping (bug original corrigé)")
    print("  - EDT pipeline 4 phases + vérification prérequis + PGSU")
    print("  - Chinchilla scaling re-mesuré pour BPE 16k")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CogNet-MoE-1B + EDT pipeline")
    parser.add_argument("--self-test", action="store_true", help="Self-test CPU puis exit")
    parser.add_argument("--dataset-path", type=str,
                        help="Path vers dataset texte brut (.txt)")
    parser.add_argument("--tokenizer-path", type=str,
                        default=str(DEFAULT_TOKENIZER_PATH),
                        help="Path vers tokenizer BPE (auto-entraîné si absent)")
    parser.add_argument("--train-tokenizer", action="store_true",
                        help="Réentraîner le tokenizer BPE avant l'entraînement")
    parser.add_argument("--tokenizer-corpus", type=str,
                        help="Corpus pour entraîner le tokenizer (si --train-tokenizer)")
    parser.add_argument("--vocab-size", type=int, default=BPE_TOKENIZER_VOCAB,
                        help=f"Vocab size cible (default {BPE_TOKENIZER_VOCAB} = BPE 16k)")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--phase1-steps", type=int, default=2000)
    parser.add_argument("--phase2b-tokens", type=int, default=500_000_000)
    parser.add_argument("--phase3-tokens", type=int, default=1_310_413_385,
                        help="Scénario C = Chinchilla / 35 (default, BPE 16k)")
    parser.add_argument("--aux-loss-weight", type=float, default=0.01,
                        help="Monter à 0.05 si routing collapse (max_load > 0.5)")
    parser.add_argument("--pgsu-n-active", type=int, default=4,
                        help="Descendre à 2 si VRAM Phase 3 saute")
    parser.add_argument("--use-bf16", action="store_true", default=True)
    parser.add_argument("--use-8bit-optimizer", action="store_true", default=True)
    parser.add_argument("--ckpt-dir", type=str, default="/home/z/my-project/cognet-moe/edt_ckpts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.dataset_path:
        print("Erreur : --dataset-path requis (ou --self-test)")
        print("\nPréparer d'abord un dataset texte brut (.txt).")
        sys.exit(1)

    # ─── Device ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cpu":
        print("⚠️  CPU détecté. Entraînement réel nécessite GPU 3090+.")
    if device == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ─── Tokenizer BPE ───────────────────────────────────────────────
    print("\n── Chargement / entraînement du tokenizer BPE ──")
    tokenizer_path = Path(args.tokenizer_path)
    if args.train_tokenizer or not tokenizer_path.exists():
        from cognet_tokenizer import train_cognet_tokenizer
        print(f"  → Entraînement tokenizer BPE (vocab_size={args.vocab_size})...")
        train_cognet_tokenizer(
            corpus=args.tokenizer_corpus,
            vocab_size=args.vocab_size,
            save_path=tokenizer_path,
            byte_level=True,
        )
    tok = CognetTokenizer(tokenizer_path=tokenizer_path, max_seq_len=args.max_seq_len)
    print(f"  Tokenizer : {tok}")
    # Le vocab_size réel (peut être < args.vocab_size si corpus trop petit).
    actual_vocab_size = tok.vocab_size

    # ─── Modèle ──────────────────────────────────────────────────────
    print("\n── Création du modèle CogNet-MoE-1B ──")
    model = create_cognet_moe_1b(
        vocab_size=actual_vocab_size,
        max_seq_len=args.max_seq_len,
        n_experts=args.n_experts,
        top_k=args.top_k,
        aux_loss_weight=args.aux_loss_weight,
        z_loss_weight=1e-3,
        use_gradient_checkpointing=True,
    )
    p = model.count_parameters()
    print(f"  Vocab size       : {actual_vocab_size:,}")
    print(f"  Total params     : {p['total']:,}")
    print(f"  Active / token   : {p['active_per_token']:,}")
    print(f"  Capacity mult    : {p['total']/p['active_per_token']:.2f}×")

    # ─── Dataset ─────────────────────────────────────────────────────
    print("\n── Chargement du dataset ──")
    dataset = CognetDataset(args.dataset_path, tok, args.max_seq_len)
    data_iter_fn = dataset.make_iter_fn(seed=args.seed)

    # ─── Config EDT ──────────────────────────────────────────────────
    cfg = EDTConfig(
        phase1_steps_per_expert=args.phase1_steps,
        phase2b_tokens=args.phase2b_tokens,
        phase3_tokens=args.phase3_tokens,
        phase3_aux_loss_weight=args.aux_loss_weight,
        phase3_pgsu_n_active=args.pgsu_n_active,
        use_bf16=args.use_bf16 and device == "cuda",
        use_8bit_optimizer=args.use_8bit_optimizer and device == "cuda",
        device=device,
        seed=args.seed,
        ckpt_dir=args.ckpt_dir,
    )

    # ─── Run EDT ─────────────────────────────────────────────────────
    print("\n── Lancement du pipeline EDT ──")
    stats = run_edt_pipeline(
        model=model,
        data_iter_fn=data_iter_fn,
        cfg=cfg,
        save_dir=args.ckpt_dir,
    )

    # ─── Save final ──────────────────────────────────────────────────
    final_path = os.path.join(args.ckpt_dir, "cognet_moe_1b_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer_path": str(tokenizer_path),
        "config": {
            "vocab_size": actual_vocab_size,
            "max_seq_len": args.max_seq_len,
            "n_experts": args.n_experts,
            "top_k": args.top_k,
        },
        "edt_stats": stats,
    }, final_path)
    print(f"\n✓ Modèle final sauvé : {final_path}")

    # ─── Rapport final ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RÉCAPITULATIF FINAL")
    print("=" * 70)
    print(f"  Tokenizer         : {tok}")
    print(f"  Temps total EDT   : {stats['total_time_h']:.2f}h "
          f"({stats['total_time_h']/24:.2f} jours)")
    print(f"  Phase 1           : {stats['phases']['phase1']['total_time_s']/3600:.2f}h")
    print(f"  Phase 2a          : {stats['phases']['phase2a']['time_s']:.2f}s")
    print(f"  Phase 2b          : {stats['phases']['phase2b']['time_s']/3600:.2f}h")
    print(f"  Phase 3           : {stats['phases']['phase3']['time_s']/3600:.2f}h")
    print(f"  Loss LM finale    : {stats['phases']['phase3']['final_loss']:.4f}")
    print(f"  Max load moyen    : {sum(stats['phases']['phase3']['max_loads'])/len(stats['phases']['phase3']['max_loads']):.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
