#!/usr/bin/env python3
"""
hash_experiment.py — Hash routing vs routing appris + faisabilité 100% sans gradient
=====================================================================================

Exp A — HASH-LSH vs SCRATCH (router appris), tâche sonde SANS FUITE (cf. refute_edt.py) :
  budgets strictement égaux (200k tokens), inits appariées, 2 seeds.
  Les experts restent entraînés par gradient — seul le ROUTING est sans gradient.
  Métriques : eval sonde, tok/s, temps wall, balance.

Exp B — 100% SANS GRADIENT (style ELM / random features) :
  trunk hash-routé FIGÉ aléatoire + tête résolue en FORME CLOSE (moindres carrés,
  torch.linalg.lstsq, CPU). Zéro backprop, zéro optimizer. Teste si on peut
  apprendre QUELQUE CHOSE sans aucun gradient.

Exp C — Débit fwd+bwd : hash vs appris (même trunk).

Usage :
    PYTHONPATH=source:. python3 hash_experiment.py   # ~2 min CPU
"""

import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
sys.path.insert(0, str(HERE))

import torch
import torch.nn as nn

from refute_edt import (make_trunk, make_pattern, probe_batch_fn, trunk_hidden,
                        probe_eval, run_probe_training, TokenCounter, VPROBE)
from hash_moe import convert_to_hash


def run_hash_arm(seed, total_tokens, pattern_seed=7, noise=0.1):
    t0 = time.time()
    counter = TokenCounter()
    pattern = make_pattern(pattern_seed)
    torch.manual_seed(9000 + seed)
    head = nn.Linear(64, VPROBE)
    model = convert_to_hash(make_trunk(VPROBE, seed=seed), mode="lsh")
    probe_fn = probe_batch_fn(pattern, noise, seed=500 + seed, counter=counter)
    run_probe_training(model, head, probe_fn, total_tokens, lr=3e-4)
    eval_fn = probe_batch_fn(pattern, noise, seed=9999, counter=None)
    loss = probe_eval(model, head, eval_fn)
    dt = time.time() - t0
    # usage bloc 0
    model.eval()
    with torch.no_grad():
        x, _ = eval_fn(16, 32)
        _, st = model.blocks[0].cognitive_expert_router(model.encoder(x))
        usage = st["moe_expert_usage"]
    return {"eval": loss, "tokens": counter.n, "time_s": dt,
            "tok_s": counter.n / dt, "usage": [round(float(u), 3) for u in usage]}


def run_learned_arm(seed, total_tokens, pattern_seed=7, noise=0.1):
    from refute_edt import run_arm
    r = run_arm("SCRATCH", seed, total_tokens, pattern_seed, noise, edt=None)
    return {"eval": r["eval_probe"], "tokens": r["total_tokens"], "time_s": r["time_s"],
            "tok_s": r["total_tokens"] / r["time_s"], "usage": r["usage"]}


def experiment_A():
    print("=" * 70)
    print("Exp A — HASH-LSH (routing sans gradient) vs SCRATCH (routing appris)")
    print("=" * 70)
    for seed in [0, 1]:
        h = run_hash_arm(seed, 200_000)
        s = run_learned_arm(seed, 200_000)
        print(f"  seed {seed} HASH   : eval={h['eval']:.4f} tok/s={h['tok_s']:.0f} "
              f"usage={h['usage']} ({h['time_s']:.1f}s)")
        print(f"  seed {seed} SCRATCH: eval={s['eval']:.4f} tok/s={s['tok_s']:.0f} "
              f"usage={s['usage']} ({s['time_s']:.1f}s)")
        print(f"  seed {seed} Δ(hash-scratch) = {h['eval'] - s['eval']:+.4f}")
    return h, s


@torch.no_grad()
def experiment_B(n_fit=50_000):
    print("\n" + "=" * 70)
    print("Exp B — 100% SANS GRADIENT : trunk aléatoire figé + tête moindres-carrés")
    print("=" * 70)
    pattern = make_pattern(7)
    model = convert_to_hash(make_trunk(VPROBE, seed=0), mode="lsh")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    fit_fn = probe_batch_fn(pattern, 0.1, seed=500)
    # Collecte features H (N×64) + labels.
    Hs, Ys = [], []
    per = 512
    for _ in range(n_fit // per):
        x, y = fit_fn(per, 32)
        h, _, _ = trunk_hidden(model, x)
        Hs.append(h.mean(dim=1))
        Ys.append(nn.functional.one_hot(y, VPROBE).float())
    H = torch.cat(Hs)  # (N, 64)
    Y = torch.cat(Ys)  # (N, 64)
    t0 = time.time()
    sol = torch.linalg.lstsq(H, Y)
    W = sol.solution  # (64, 64) — forme close, ZÉRO gradient
    dt = time.time() - t0
    print(f"  lstsq résolu en {dt:.1f}s sur CPU ({n_fit:,} échantillons, rang={int(sol.rank.item())})")
    # Eval.
    eval_fn = probe_batch_fn(pattern, 0.1, seed=9999)
    tot, n = 0.0, 0
    for _ in range(40):
        x, y = eval_fn(16, 32)
        h, _, _ = trunk_hidden(model, x)
        tot += nn.functional.cross_entropy(h.mean(dim=1) @ W, y).item()
        n += 1
    print(f"  eval sonde = {tot/n:.4f} (chance = {__import__('math').log(VPROBE):.4f}, "
          f"scratch-gradient ≈ 2.89)")
    return tot / n


def experiment_C():
    print("\n" + "=" * 70)
    print("Exp C — Débit fwd+bwd : hash vs appris (même trunk 64d×2b, B=8 T=32)")
    print("=" * 70)
    import torch.nn.functional as F
    for name, mk in [("appris", lambda: make_trunk(512, 0)),
                     ("hash", lambda: convert_to_hash(make_trunk(512, 0), mode="lsh"))]:
        m = mk().train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        ids = torch.randint(0, 512, (8, 32))
        for _ in range(2):
            opt.zero_grad()
            o = m(ids)
            (o["logits"].sum() + o["moe_aux_loss"]).backward()
            opt.step()
        t0 = time.time()
        it = 15
        for _ in range(it):
            opt.zero_grad()
            o = m(ids)
            (o["logits"].sum() + o["moe_aux_loss"]).backward()
            opt.step()
        tps = it * 8 * 32 / (time.time() - t0)
        ntrain = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:7s}: {tps:.0f} tok/s, {ntrain:,} params entraînables")
    # Pré-sharding : découpe SANS modèle.
    from hash_moe import shard_by_hash
    big = torch.randint(0, 512, (200, 512))  # 102k tokens
    t0 = time.time()
    shards = shard_by_hash(big, 8)
    dt = time.time() - t0
    print(f"  pré-sharding 102k tokens par hash (sans modèle) : "
          f"{[len(v) for v in shards.values()]} en {dt*1000:.1f}ms → chaque CPU prend son shard")


def main():
    t0 = time.time()
    experiment_A()
    experiment_B()
    experiment_C()
    print(f"\nTemps total : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
