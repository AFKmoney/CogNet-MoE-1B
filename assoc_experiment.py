#!/usr/bin/env python3
"""
assoc_experiment.py — Experts associatifs (Hebb, 0 gradient) vs experts denses
================================================================================
Sonde SANS FUITE (cf. refute_edt.py), budgets strictement égaux (200k tokens
d'entrée), inits appariées, seeds 0-1 (+2) :

  ASSOC      : experts associatifs paginés (retrieval + superposition Hebbienne,
               buffers gelés) ; tête + encodeur + mémoire + normes via Adam.
               Experts DÉTACHÉS du graphe (forward seul) — le gradient global
               circule par les résiduels (apprentissage local).
  HASH-TOKEN : experts denses FusedSwiGLU + routing token-hash (même sel),
               tout-gradient. Isole EXACTEMENT la substitution d'expert
               (même routage, même boucle, mêmes données).

Références commitées (même sonde) : HASH-LSH 2.08, SCRATCH 2.89,
ELM figé 3.98, chance 4.16.

Usage :
    PYTHONPATH=source:. python3 assoc_experiment.py   # ~3-5 min CPU
"""

import json
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
sys.path.insert(0, str(HERE))

import torch
import torch.nn as nn
import torch.nn.functional as F

from refute_edt import (make_trunk, make_pattern, probe_batch_fn, trunk_hidden,
                        TokenCounter, VPROBE)
from hash_moe import convert_to_hash
from expert_pager import ExpertPager, PagerConfig, hash_ahead_pages, working_set_hint
from assoc_experts import convert_to_assoc, AttentiveHead, head_credit

SEEDS = [0, 1, 2]
TOKENS = 200_000
VIGILANCE = 0.0  # ART-label dégénère (histogrammes uniformes, cf. rapport §8)
NOVELTY_GAMMA = 0.0  # neutre avec delta (cf. rapport §9)
READOUT = "delta"  # v3 : delta-rule locale, dernier bloc seul (cf. rapport §9)
DELTA_BLOCKS = "last"  # crédit exact au dernier bloc ; amont gelé (cf. §9)
HEBB_ETA = 0.1
B, T = 16, 32


def set_ids(model, ids, labels=None):
    for blk in model.blocks:
        r = blk.cognitive_expert_router
        if hasattr(r, "set_batch_token_ids"):
            r.set_batch_token_ids(ids)
        if labels is not None and hasattr(r, "set_batch_labels"):
            r.set_batch_labels(labels)


@torch.no_grad()
def assoc_eval(model, head, batch_fn, n_batches=40):
    model.eval(); head.eval()
    tot, n = 0.0, 0
    for _ in range(n_batches):
        x, y = batch_fn(16, 32)
        set_ids(model, x)
        h, _, _ = trunk_hidden(model, x)
        logits, _ = head(h)
        tot += F.cross_entropy(logits, y).item(); n += 1
    model.train(); head.train()
    return tot / n


def train_arm(model, head, batch_fn, tokens, pager=None, hebbian_eta=0.0,
              bind_alpha=0.0, vigilance=0.0, lsh_hint=False, novelty_gamma=0.0,
              readout="legacy", delta_blocks="all", tag=""):
    """Boucle partagée (équité stricte) ; Hebb (+binding +ART) distingue ASSOC."""
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=3e-4)
    steps = max(1, tokens // (B * T))
    cur = batch_fn(B, T)
    recent = []
    for s in range(steps):
        nxt = batch_fn(B, T)
        if pager is not None:
            if lsh_hint:  # LSH : hint heuristique (couche-0 exacte + localité)
                pager.prefetch(working_set_hint(recent, nxt[0], model))
            else:  # token : hash-ahead exact (même sel → mêmes pages)
                pager.prefetch(hash_ahead_pages(nxt[0], model))
            pager._install_prefetched()
        x, y = cur
        set_ids(model, x, y if vigilance > 0 else None)
        opt.zero_grad()
        h, aux, z = trunk_hidden(model, x)
        logits, _ = head(h)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        if hebbian_eta > 0:  # superposition directe (sans gradient)
            if lsh_hint:
                pager.prefetch(working_set_hint(recent, x, model))
            else:
                pager.prefetch(hash_ahead_pages(x, model))
            le = None
            with torch.no_grad():
                if bind_alpha > 0:  # embedding du label, détaché (cible)
                    le = model.encoder.token_emb(y).repeat_interleave(T, dim=0)
            re = None
            if readout == "delta":  # crédit exact 1-couche, par formule (0 graphe)
                re, _ = head_credit(head, h.detach(), y)
                if delta_blocks != "hybrid":
                    le = None  # delta subsumes le binding (cibles en conflit sinon)
            nblocks = len(model.blocks)
            for bi, blk in enumerate(model.blocks):
                r = blk.cognitive_expert_router
                ly = y if vigilance > 0 else None
                re_b, le_b = re, le
                if readout == "delta" and delta_blocks in ("last", "hybrid") \
                        and bi < nblocks - 1:
                    re_b = None  # blocs amont : pas de crédit (Jacobien non-identité)
                    if delta_blocks == "last":
                        continue  # experts amont gelés à l'identité (v=A=0)
                    # hybride : amont en legacy stable (Hebb+binding), aval en delta
                r.hebbian_step(eta=hebbian_eta, label_emb=le_b, label_alpha=bind_alpha,
                               label_ids=ly, rho=vigilance if vigilance > 0 else 0.5,
                               novelty_gamma=novelty_gamma, readout_err=re_b)
            recent = sorted(pager.resident.keys())[-8:]
        cur = nxt
    if pager is not None:
        pager.flush()
    return steps


def run_arm(kind, seed, total_tokens, pattern_seed=7, noise=0.1,
            n_mem_slots=8, window=4, bind_alpha=0.0, vigilance=0.0, mode="token",
            novelty_gamma=0.0, readout="legacy", eta=0.05, delta_blocks="all",
            binary=False):
    t0 = time.time()
    counter = TokenCounter()
    pattern = make_pattern(pattern_seed)
    torch.manual_seed(9000 + seed)
    head = AttentiveHead(64, VPROBE)
    trunk = make_trunk(VPROBE, seed=seed)  # init appariée entre bras
    pager = None
    if kind != "ASSOC":
        eta = 0.0  # bras dense : pas de Hebb (le gradient fait tout)
    if kind == "ASSOC":
        pager = ExpertPager(PagerConfig(store_dir=f"/tmp/assoc_exp_{seed}", num_loaders=4))
        model = convert_to_assoc(trunk, n_slots=2, pager=pager, n_mem_slots=n_mem_slots,
                                 window=window, seed=seed, n_labels=VPROBE, mode=mode,
                                 binary_addressing=binary)
    else:
        model = convert_to_hash(trunk, mode="token")
    ntrain = sum(p.numel() for p in list(model.parameters()) + list(head.parameters())
                 if p.requires_grad)
    probe_fn = probe_batch_fn(pattern, noise, seed=500 + seed, counter=counter)
    init_eval = assoc_eval(model, head, probe_batch_fn(pattern, noise, seed=9999))
    train_arm(model, head, probe_fn, total_tokens, pager, eta, bind_alpha,
              vigilance, lsh_hint=(kind == "ASSOC" and mode == "lsh"),
              novelty_gamma=novelty_gamma,
              readout=(readout if kind == "ASSOC" else "legacy"),
              delta_blocks=delta_blocks, tag=f"{kind}{seed}")
    loss = assoc_eval(model, head, probe_batch_fn(pattern, noise, seed=9999))
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        xe, _ = probe_batch_fn(pattern, noise, seed=9999)(16, 32)
        set_ids(model, xe)
        _, st = model.blocks[0].cognitive_expert_router(model.encoder(xe))
        usage = [round(float(u), 3) for u in st["moe_expert_usage"]]
    out = {"eval": loss, "init_eval": init_eval, "tokens": counter.n, "time_s": dt,
           "tok_s": counter.n / dt, "usage": usage, "trainable": ntrain}
    if pager is not None:
        out["pager"] = {k: v for k, v in pager.stats.items()}
        out["art_fallbacks"] = sum(
            getattr(blk.cognitive_expert_router, "_fallbacks", 0) for blk in model.blocks)
    return out


def main():
    print("=" * 70)
    print("ASSOC (Hebb paginé, 0 gradient expert) vs HASH-TOKEN (dense, gradient)")
    print("Sonde leak-free, 200k tokens, inits appariées. Réf : LSH 2.08 / SCR 2.89 / ELM 3.98")
    print("=" * 70)
    results = {}
    for seed in SEEDS:
        a = run_arm("ASSOC", seed, TOKENS, vigilance=VIGILANCE,
                    novelty_gamma=NOVELTY_GAMMA, readout=READOUT,
                    delta_blocks=DELTA_BLOCKS, eta=HEBB_ETA)
        h = run_arm("HASH-TOKEN", seed, TOKENS)
        results[f"seed{seed}"] = {"ASSOC": a, "HASH-TOKEN": h}
        print(f"  seed {seed} ASSOC     : eval={a['eval']:.4f} (init {a['init_eval']:.4f}) "
              f"tok/s={a['tok_s']:.0f} usage={a['usage']} train={a['trainable']:,}")
        if "pager" in a:
            p = a["pager"]
            print(f"           pager: hits={p['hits']} sync={p['faults_sync']} "
                  f"couverts={p['faults_covered']} wb={p['writebacks']} jokers={p['joker_fallbacks']}")
        print(f"  seed {seed} HASH-TOKEN: eval={h['eval']:.4f} (init {h['init_eval']:.4f}) "
              f"tok/s={h['tok_s']:.0f} usage={h['usage']} train={h['trainable']:,}")
        print(f"  seed {seed} Δ(assoc-hash) = {a['eval']-h['eval']:+.4f}")
    ma = sum(results[f"seed{s}"]["ASSOC"]["eval"] for s in SEEDS) / len(SEEDS)
    mh = sum(results[f"seed{s}"]["HASH-TOKEN"]["eval"] for s in SEEDS) / len(SEEDS)
    print(f"\nMoyennes : ASSOC {ma:.4f} vs HASH-TOKEN {mh:.4f} (Δ {ma-mh:+.4f})")
    print("Rappel : HASH-LSH 2.08 | SCRATCH 2.89 | ELM figé 3.98 | chance 4.16")
    with open(HERE / "assoc_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("Résultats bruts → assoc_results.json")


def ablate():
    """Matrice seed-0 : capacité × binding (rapide, oriente le run complet)."""
    print("=" * 70)
    print("Ablation seed-0 : M/W × binding label (200k tokens chacun)")
    print("=" * 70)
    for tag, kw in [("bin-v3", {"readout": "delta", "delta_blocks": "last",
                               "eta": 0.1, "binary": True})]:
        a = run_arm("ASSOC", 0, TOKENS, **kw)
        p = a["pager"]
        print(f"  {tag:12s}: eval={a['eval']:.4f} tok/s={a['tok_s']:.0f} "
              f"hits={p['hits']} sync={p['faults_sync']} couverts={p['faults_covered']} "
              f"ART_replis={a.get('art_fallbacks', 0)}")


if __name__ == "__main__":
    if "--ablate" in sys.argv:
        ablate()
    else:
        main()
