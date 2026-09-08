#!/usr/bin/env python3
"""
refute_edt.py — Tentative de RÉFUTATION d'EDT sur CogNet-MoE (CPU tiny)
=======================================================================

Contexte : EDT a déjà été réfuté sur Fractus. Ce script tente de le réfuter
ici aussi, rigoureusement, avec un protocole corrigé.

Volet 0 — Preuve de fuite (leak) dans le protocole de validation existant :
  Sur données i.i.d. (Zipf-136, comme validate_edt.py), TOUT modèle est borné
  par l'entropie unigramme (~3.90 nats). Si un training shifted-LM descend
  SOUS ce plancher, c'est une preuve information-théorique que le label fuit
  dans l'input — normal : CogNet est BIDIRECTIONNEL (aucun masque causal :
  mean_key sur T, SDPA is_causal=False, composer avec shift futur).
  L'ancien verdict « scratch=1.50 bat EDT=3.60 » comparait donc « qui exploite
  le mieux la fuite », pas « qui apprend le mieux ».

Volet 1 — Comparaison ÉQUITABLE sans fuite (tâche sonde leak-free) :
  Tâche : préfixe bruité de 32 tokens (motif fixe, vocab 64) → prédire le
  33e token (label JAMAIS dans l'input : pas de fuite possible).
  3 bras × 2 seeds, budgets TOKENS-INPUT strictement égaux (y compris les
  tokens consommés par la Phase 1, que les papiers EDT « oublient » souvent) :
    - SCRATCH  : init aléatoire + entraînement joint direct sur la sonde.
    - EDT-V2   : Phase 1 (perturbation ON) + 2a (50 steps) + 2b, puis sonde.
    - EDT-NOFIX: Phase 1 (perturbation OFF, identité pure) + 2a (1 step), puis sonde.
  Init trunk appariée (même seed) entre bras : seule la méthode diffère.

Usage :
    PYTHONPATH=source:. python3 refute_edt.py            # ~8-12 min CPU
    PYTHONPATH=source:. python3 refute_edt.py --quick    # ~2-3 min (1 seed, budgets /2)

Sortie : verdict + tableau + refutation_report.json
"""

import sys, os, time, math, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
sys.path.insert(0, str(HERE))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cognet_moe import CogNetMoE1B
from edt_pipeline import EDTConfig, phase1_experts, phase2a_attention, phase2b_embedding


# ───────────────────────────────────────────────────────────────────
# Config tiny (CPU)
# ───────────────────────────────────────────────────────────────────
def make_trunk(vocab, seed):
    torch.manual_seed(seed)
    return CogNetMoE1B(
        vocab_size=vocab, hidden_dim=64, num_blocks=2, num_channels=4,
        channel_dim=16, ff_dim=128, max_seq_len=64,
        working_slots=4, episodic_slots=8, semantic_slots=16,
        key_dim=16, n_experts=4, top_k=2, dropout=0.0,
        aux_loss_weight=0.05, z_loss_weight=1e-3,
        use_gradient_checkpointing=False,
    )


class TokenCounter:
    def __init__(self): self.n = 0
    def add(self, t): self.n += t


# ───────────────────────────────────────────────────────────────────
# VOLET 0 : preuve de fuite sur bruit i.i.d.
# ───────────────────────────────────────────────────────────────────
def experiment_0_leak(quick=False):
    print("\n" + "=" * 70)
    print("VOLET 0 — La validation existante fuit-elle ? (bruit Zipf i.i.d.)")
    print("=" * 70)
    V = 136
    probs = 1.0 / torch.arange(1, V + 1, dtype=torch.float64)
    probs = probs / probs.sum()
    # Plancher information-théorique : entropie unigramme (données i.i.d.).
    floor = float(-(probs * probs.log()).sum())
    print(f"  Plancher unigramme (borne inf tout modèle sur i.i.d.) : {floor:.4f} nats")
    print(f"  Ancien rapport : scratch=1.50 (soit {floor - 1.50:+.2f} SOUS le plancher → impossible sans fuite)")

    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(1234)
    def data_fn(B, T):
        return torch.multinomial(probs.float(), B * T, replacement=True, generator=rng).view(B, T)

    model = make_trunk(V, seed=0)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    B, T = 8, 64
    steps = 80 if quick else 200
    for s in range(steps):
        ids = data_fn(B, T)
        opt.zero_grad()
        out = model(ids)
        loss = (F.cross_entropy(out["logits"][:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
                + 0.05 * out["moe_aux_loss"] + 1e-3 * out["moe_z_loss"])
        loss.backward()
        opt.step()
        if (s + 1) % max(1, steps // 4) == 0:
            print(f"    step {s+1}/{steps} train_loss={loss.item():.4f} (plancher={floor:.2f})")

    model.eval()
    with torch.no_grad():
        tot, n = 0.0, 0
        for _ in range(20):
            ids = data_fn(B, T)
            l = F.cross_entropy(model(ids)["logits"][:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
            tot += l.item(); n += 1
    eval_loss = tot / n
    print(f"  Eval après {steps} steps shifted-LM sur bruit i.i.d. : {eval_loss:.4f}")
    leaked = eval_loss < floor - 0.1
    print(f"  → {'☠️  FUITE PROUVÉE' if leaked else 'pas de fuite détectée'} "
          f"(écart au plancher : {eval_loss - floor:+.3f})")
    return {"floor": floor, "eval_loss": eval_loss, "leaked": leaked}


# ───────────────────────────────────────────────────────────────────
# VOLET 1 : tâche sonde sans fuite + 3 bras à budget égal
# ───────────────────────────────────────────────────────────────────
VPROBE = 64
PLEN = 256  # longueur du motif à mémoriser

def make_pattern(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VPROBE, (PLEN,), generator=g)

def probe_batch_fn(pattern, noise, seed, counter=None):
    """(B,32) préfixe bruité → (B,) label = token suivant (jamais dans l'input)."""
    rng = torch.Generator().manual_seed(seed)
    def fn(B, T=32):
        s = torch.randint(0, PLEN - T - 1, (B,), generator=rng)
        x = torch.stack([pattern[i:i + T] for i in s]).clone()
        y = torch.stack([pattern[i + T] for i in s])
        mask = torch.rand(x.shape, generator=rng) < noise
        x[mask] = torch.randint(0, VPROBE, (int(mask.sum().item()),), generator=rng)
        if counter is not None:
            counter.add(B * T)
        return x, y
    return fn

def pretrain_stream_fn(pattern, noise, seed, counter=None):
    """Stream pour phases EDT (même distribution que la sonde, BloCS de 32)."""
    rng = torch.Generator().manual_seed(seed)
    def fn(B, T=32):
        s = torch.randint(0, PLEN - T, (B,), generator=rng)
        x = torch.stack([pattern[i:i + T] for i in s]).clone()
        mask = torch.rand(x.shape, generator=rng) < noise
        x[mask] = torch.randint(0, VPROBE, (int(mask.sum().item()),), generator=rng)
        if counter is not None:
            counter.add(B * T)
        return x
    return fn

def trunk_hidden(model, x):
    """Hidden states finaux (sans head LM) + aux losses."""
    h = model.encoder(x)
    aux = torch.tensor(0.0, device=x.device)
    z = torch.tensor(0.0, device=x.device)
    for blk in model.blocks:
        h, st = blk(h)
        aux = aux + st["moe_aux_loss"]
        z = z + st["moe_z_loss"]
    return model.final_norm(h), aux, z

@torch.no_grad()
def probe_eval(model, head, batch_fn, n_batches=40):
    model.eval(); head.eval()
    tot, n = 0.0, 0
    for _ in range(n_batches):
        x, y = batch_fn(16, 32)
        h, _, _ = trunk_hidden(model, x)
        logits = head(h.mean(dim=1))
        tot += F.cross_entropy(logits, y).item(); n += 1
    model.train(); head.train()
    return tot / n

@torch.no_grad()
def expert_usage(model, batch_fn, n_batches=10):
    """Usage moyen par expert (bloc 0) + diversité comportementale."""
    model.eval()
    uses, outs = [], []
    for _ in range(n_batches):
        x, _ = batch_fn(16, 32)
        h = model.encoder(x)
        _, st = model.blocks[0].cognitive_expert_router(h)
        uses.append(st["moe_expert_usage"])
        # diversité comportementale : variance inter-experts des sorties
        with torch.no_grad():
            outs.append(torch.stack([
                model.blocks[0].cognitive_expert_router.experts[e](h[:4])
                for e in range(4)]).var(dim=0).mean().item())
    model.train()
    return torch.stack(uses).mean(0), sum(outs) / len(outs)

def run_probe_training(model, head, batch_fn, tokens, lr=3e-4, log=False, tag=""):
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr)
    B, T = 16, 32
    steps = max(1, tokens // (B * T))
    for s in range(steps):
        x, y = batch_fn(B, T)
        opt.zero_grad()
        h, aux, z = trunk_hidden(model, x)
        loss = (F.cross_entropy(head(h.mean(dim=1)), y)
                + 0.05 * aux.clamp(max=10.0) + 1e-3 * z.clamp(max=10.0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        if log and (s + 1) % max(1, steps // 4) == 0:
            print(f"    [{tag}] step {s+1}/{steps} loss={loss.item():.4f}")
    return steps

def run_arm(name, seed, total_tokens, pattern_seed=7, noise=0.1,
            edt=None, p1_steps=20, p2a_steps=50, p2b_tokens=20000):
    """
    edt: None (scratch) | 'v2' (P1 perturb ON + 2a 50 + 2b) | 'nofix' (P1 ident pure + 2a 1 + 2b)
         | 'p1only' (Phase 1 seule) | 'p2only' (Phase 2b seule)
    Retourne métriques + tokens comptés par phase.
    """
    t0 = time.time()
    counter = TokenCounter()
    pattern = make_pattern(pattern_seed)
    model = make_trunk(VPROBE, seed=seed)
    torch.manual_seed(9000 + seed)
    head = nn.Linear(64, VPROBE)
    spent = {}

    do_p1 = edt in ("v2", "nofix", "p1only")
    do_p2a = edt in ("v2", "nofix")
    do_p2b = edt in ("v2", "nofix", "p2only")
    if edt is not None:
        # ── Phase 1 (tokens comptés via le stream caché) ──
        stream = pretrain_stream_fn(pattern, noise, seed=100 + seed, counter=counter)
        def hidden_fn(B, T):
            ids = stream(B, T)
            with torch.no_grad():
                h = model.encoder(ids)
                for blk in model.blocks:  # profondeur FIXE (reproductible)
                    h, _ = blk.memory(h); h = blk.composer(h); h = blk.norm(h)
            return h
        cfg = EDTConfig(
            phase1_steps_per_expert=p1_steps, phase1_batch_size=8, phase1_seq_len=32,
            phase1_lr=3e-4, phase1_target_loss=1e-9,  # pas d'arrêt anticipé : budget exact
            phase1_use_perturbation=(edt in ("v2", "p1only")),
            phase2a_steps=p2a_steps if edt == "v2" else 1,
            phase2a_batch_size=8, phase2a_lr=3e-4,
            phase2b_tokens=0, phase2b_batch_size=8, phase2b_seq_len=32, phase2b_lr=6e-4,
            phase3_tokens=0, use_bf16=False, use_8bit_optimizer=False,
            device="cpu", log_every=10**9)
        import io, contextlib
        if do_p1:
            with contextlib.redirect_stdout(io.StringIO()):
                phase1_experts(model, hidden_fn, cfg)
        spent["phase1"] = counter.n
        # ── Phase 2a ──
        if do_p2a:
            with contextlib.redirect_stdout(io.StringIO()):
                phase2a_attention(model, stream, cfg)
        spent["phase2a"] = counter.n - sum(spent.values())
        # ── Phase 2b (budget fixe) ──
        cfg.phase2b_tokens = p2b_tokens
        if do_p2b:
            with contextlib.redirect_stdout(io.StringIO()):
                phase2b_embedding(model, stream, cfg)
        spent["phase2b"] = counter.n - sum(spent.values())
    else:
        spent = {"phase1": 0, "phase2a": 0, "phase2b": 0}

    # ── Sonde : le RESTE du budget (égalité stricte des totaux) ──
    probe_tokens = total_tokens - counter.n
    assert probe_tokens > 0, f"budget épuisé par EDT ({counter.n} > {total_tokens})"
    probe_fn = probe_batch_fn(pattern, noise, seed=500 + seed, counter=counter)
    run_probe_training(model, head, probe_fn, probe_tokens, lr=3e-4)
    spent["probe"] = counter.n - sum(spent.values())

    # ── Évals ──
    eval_fn = probe_batch_fn(pattern, noise, seed=9999, counter=None)  # bruit frais
    loss_main = probe_eval(model, head, eval_fn)
    # Contrôle : motif inédit (doit être ~chance si le modèle a mémorisé le motif 1)
    pattern2 = make_pattern(4242)
    eval_fn2 = probe_batch_fn(pattern2, noise, seed=9999, counter=None)
    loss_new = probe_eval(model, head, eval_fn2)
    usage, behav_div = expert_usage(model, lambda B, T: (eval_fn(B, T)[0], None))
    dead = int((usage < 0.05).sum().item())
    dt = time.time() - t0
    return {
        "arm": name, "seed": seed, "eval_probe": loss_main, "eval_new_pattern": loss_new,
        "usage": [round(float(u), 3) for u in usage], "dead_experts": dead,
        "behav_diversity": behav_div, "tokens": spent, "total_tokens": counter.n,
        "time_s": round(dt, 1),
    }

def experiment_1_fair(quick=False):
    print("\n" + "=" * 70)
    print("VOLET 1 — Comparaison équitable SANS fuite (sonde : préfixe → token suivant)")
    print("=" * 70)
    total = 100_000 if quick else 200_000
    p2b = 5_000 if quick else 20_000
    seeds = [0] if quick else [0, 1]
    print(f"  Budget TOTAL par bras : {total:,} tokens-input (Phases EDT incluses)")
    print(f"  Modèle : 64d × 2 blocs × 4 experts top-2 | motif 256 tokens, vocab 64, bruit 10%")
    results = []
    for seed in seeds:
        for name, edt in [("SCRATCH", None), ("EDT-V2", "v2"), ("EDT-NOFIX", "nofix"),
                          ("P1-ONLY", "p1only"), ("P2B-ONLY", "p2only")]:
            print(f"\n  ── Bras {name} (seed {seed})… ──", flush=True)
            r = run_arm(name, seed, total, p2b_tokens=p2b, edt=edt)
            results.append(r)
            print(f"     eval sonde={r['eval_probe']:.4f} | motif inédit={r['eval_new_pattern']:.4f} "
                  f"| usage={r['usage']} | morts={r['dead_experts']} "
                  f"| div={r['behav_diversity']:.4f} | tokens={r['tokens']} | {r['time_s']}s")
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    e0 = experiment_0_leak(quick=args.quick)
    e1 = experiment_1_fair(quick=args.quick)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Volet 0 : {'☠️ fuite prouvée — validation existante INVALIDE' if e0['leaked'] else 'pas de fuite'}")
    # Moyennes par bras
    from collections import defaultdict
    agg = defaultdict(list)
    for r in e1:
        agg[r["arm"]].append(r["eval_probe"])
    for k, v in agg.items():
        print(f"  {k:10s} eval sonde = {sum(v)/len(v):.4f}  (runs: {[f'{x:.4f}' for x in v]})")
    s = sum(agg["SCRATCH"]) / len(agg["SCRATCH"])
    edt_arms = {k: sum(v) / len(v) for k, v in agg.items() if k != "SCRATCH"}
    best_name = min(edt_arms, key=edt_arms.get)
    best = edt_arms[best_name]
    print()
    if s <= best - 0.05:
        print(f"  ❌ EDT RÉFUTÉ : SCRATCH ({s:.4f}) bat le meilleur bras EDT ({best_name}={best:.4f}) à budget égal.")
    elif best < s - 0.05:
        print(f"  ✅ EDT NON réfuté : {best_name} ({best:.4f}) bat SCRATCH ({s:.4f}).")
    else:
        print(f"  ⚠️  MATCH NUL (|Δ| < 0.05) : EDT n'apporte rien à budget égal → bénéfice 35× NON observé.")
    p1 = edt_arms.get("P1-ONLY", float("nan"))
    p2 = edt_arms.get("P2B-ONLY", float("nan"))
    print(f"  Ablations : P1-ONLY={p1:.4f} (Phase 1 seule), P2B-ONLY={p2:.4f} (Phase 2b seule) vs SCRATCH={s:.4f}")
    print(f"  Temps total : {(time.time()-t0)/60:.1f} min")

    with open(HERE / "refutation_report.json", "w") as f:
        json.dump({"volet0": e0, "volet1": e1}, f, indent=2)
    print("  Rapport : refutation_report.json")


if __name__ == "__main__":
    main()
