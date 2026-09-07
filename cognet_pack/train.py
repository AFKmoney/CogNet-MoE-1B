#!/usr/bin/env python3
"""CogNet-MoE-1B — point d'entrée unique du pack isolé.

    python3 train.py --demo [--tokens 100000]
        Démo CPU complète : génère un corpus synthétique, le pré-tokenise
        en .bin, entraîne un CogNet-MoE tiny (28M params) et évalue.
        ~2-3 min sur CPU 2 coeurs. Preuve que le modèle apprend vraiment.

    python3 train.py --self-test
        Tous les self-tests du pack (modèle + fast + infini).

    Autres flags : délégation directe à fast_train.py :
    python3 train.py --build-bin --txt corpus.txt --out data/p0 [--tokenizer cognet_tokenizer.json]
    python3 train.py --train --bin data/p0.bin --tokens 1e9 [--tiny] [--batch-size 16 ...]
    python3 train.py --train --bin data/p0.bin --tokens 1e9 --batch-size 16 --grad-accum 4  # 1B réel (GPU)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "source"))
sys.path.insert(0, HERE)


def cmd_demo(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from fast_train import (build_bin_dataset, MMapTokens, FastTrainer,
                            FastTrainerConfig, convert_to_fast, enable_fast_mode)
    from cognet_moe import CogNetMoE1B

    tokens = int(float(args.tokens))
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    txt = os.path.join(outdir, "demo_corpus.txt")
    prefix = os.path.join(outdir, "demo")
    ckpt_dir = os.path.join(outdir, "ckpts")

    # 1. Corpus synthétique (structure répétitive → loss mesurable).
    import random
    random.seed(7)
    sujets = ["le chat", "le chien", "l'oiseau", "le renard", "la lune",
              "le soleil", "la rivière", "le vent"]
    verbes = ["regarde", "traverse", "éclaire", "suit", "réveille",
              "apaise", "caresse", "guide"]
    objets = ["la forêt silencieuse", "le village endormi", "la montagne bleue",
              "le jardin fleuri", "la mer calme", "le chemin poussiéreux",
              "la nuit étoilée", "l'aube naissante"]
    lignes = []
    for i in range(6000):
        s, v, o = random.choice(sujets), random.choice(verbes), random.choice(objets)
        lignes.append(f"{s} {v} {o}.")
        if i % 3 == 0:
            lignes.append(f"Dans {o}, {s} {v} doucement.")
    open(txt, "w").write("\n".join(lignes) + "\n")
    print(f"[Démo] corpus : {txt} ({os.path.getsize(txt)} octets)")

    # 2. Pré-tokenisation .bin (byte fallback, 0 dépendance tokenizer).
    build_bin_dataset(txt, prefix, tokenizer=None)
    ds = MMapTokens(prefix + ".bin")
    print(f"[Démo] dataset : {ds.n_tokens:,} tokens, vocab={ds.vocab_size}")

    # 3. Modèle tiny + entraînement.
    enable_fast_mode()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CogNetMoE1B(vocab_size=ds.vocab_size, hidden_dim=256, num_blocks=4,
                        num_channels=8, channel_dim=64, ff_dim=1024,
                        max_seq_len=64, working_slots=16,
                        episodic_slots=32, semantic_slots=64, key_dim=64,
                        n_experts=8, top_k=2, use_gradient_checkpointing=True)
    model = convert_to_fast(model)
    p = model.count_parameters()
    print(f"[Démo] params total={p['total']:,} actif/token={p['active_per_token']:,} "
          f"device={device}")
    cfg = FastTrainerConfig(batch_size=8, grad_accum=4, lr=3e-4,
                            warmup_steps=20, compile_mode=None,
                            optimizer="adamw", seq_stages=[(0, 64)])
    rng = np.random.default_rng(42)
    FastTrainer(model, cfg, device=device, ckpt_dir=ckpt_dir).train(
        lambda B, T: ds.sample_batch(B, T, rng), total_tokens=tokens)

    # 4. Éval : recharge stricte du checkpoint + loss moyenne.
    sd = torch.load(os.path.join(ckpt_dir, "final.pt"), map_location="cpu",
                    weights_only=False)
    model.load_state_dict(sd["model_state_dict"], strict=True)
    model.eval()
    ls = []
    with torch.no_grad():
        for _ in range(10):
            b = ds.sample_batch(8, 64, rng)
            out = model(b[:, :-1])
            logits = out["logits"] if isinstance(out, dict) else out
            ls.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      b[:, 1:].reshape(-1)).item())
    print(f"[Démo] ✓ loss init ~6.3 → loss_ema train {sd['loss_ema']:.2f} → "
          f"EVAL {float(np.mean(ls)):.2f} — le modèle a appris "
          f"(corpus synthétique : mémorisation, pas généralisation).")
    print(f"[Démo] ✓ checkpoint strict-load OK : {ckpt_dir}/final.pt "
          f"(step={sd['step']}, tokens={sd['tokens_seen']:,})")


def cmd_self_test():
    # Ordre croissant de taille ; chaque sous-process hérite de l'env.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(HERE, "source") + os.pathsep + HERE \
        + os.pathsep + env.get("PYTHONPATH", "")
    for prog in ["cognet_moe.py", "fast_train.py --self-test",
                 "phase_routed_moe.py --self-test", "run_infinite.py --self-test"]:
        print("=" * 70)
        print(f"$ python3 {prog}")
        print("=" * 70)
        r = subprocess.run([sys.executable] + prog.split(), cwd=HERE, env=env)
        if r.returncode != 0:
            print(f"ÉCHEC : {prog}")
            sys.exit(1)
    print("=" * 70)
    print("✓ Tous les self-tests du pack sont passés.")
    print("=" * 70)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="CogNet-MoE-1B — pack isolé")
    ap.add_argument("--demo", action="store_true", help="démo CPU complète")
    ap.add_argument("--tokens", type=str, default="100000", help="budget tokens démo")
    ap.add_argument("--outdir", type=str, default="./demo_out", help="sortie démo")
    ap.add_argument("--self-test", action="store_true", help="tous les self-tests")
    known, rest = ap.parse_known_args()
    if known.self_test:
        cmd_self_test()
    elif known.demo:
        cmd_demo(known)
    else:
        # Délégation : train.py --train ... == fast_train.py --train ...
        from fast_train import main as fast_main
        sys.argv = ["fast_train.py"] + sys.argv[1:]
        fast_main()


if __name__ == "__main__":
    main()
