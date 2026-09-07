# Training rapide + infini — CogNet-MoE-1B

> **Objectif** : entraîner CogNet sur des **milliards de tokens sans les jours de GPU**,
> et produire un `.pt` final **entraînable à l'infini** grâce au **Phase-Routed MoE**
> (création d'experts à la volée). Le tout **100% CogNet-native** : pas d'attention,
> pas de gate transformer-style, O(n) strict.

---

## TL;DR

```bash
# Dépendances (en plus de torch) : numpy ; tokenizers (déjà requis par le repo)
pip install numpy tokenizers

# 1. Pré-tokeniser UNE fois chaque corpus en .bin (tue le goulot CPU)
python3 fast_train.py --build-bin --txt corpus_phase0.txt \
    --tokenizer cognet_tokenizer.json --out data/p0
python3 fast_train.py --build-bin --txt corpus_phase1.txt \
    --tokenizer cognet_tokenizer.json --out data/p1

# 2a. Training joint rapide (1 phase, remplace phase3 lente)
python3 fast_train.py --train --bin data/p0.bin --tokens 1361634450 \
    --batch-size 16 --grad-accum 4 --compile reduce-overhead

# 2b. OU training INFINI multi-phases (experts créés à la volée)
python3 run_infinite.py --bins data/p0.bin,data/p1.bin \
    --tokens-per-phase 1000000000 --new-experts-per-phase 2 --compile reduce-overhead

# 3. Prolonger le .pt final plus tard avec de nouvelles données (infini ♾️)
python3 fast_train.py --build-bin --txt corpus_phase2.txt \
    --tokenizer cognet_tokenizer.json --out data/p2
python3 run_infinite.py --bins data/p0.bin,data/p1.bin,data/p2.bin \
    --resume infinite_ckpts/final.pt --start-phase 2 --tokens-per-phase 1000000000

# Vérifier que tout marche (CPU, ~1 min)
python3 phase_routed_moe.py --self-test && python3 fast_train.py --self-test \
    && python3 run_infinite.py --self-test
```

---

## 1. Pourquoi l'entraînement prenait des jours (7 goulots)

| # | Goulot | Coût | Fix (fichier) |
|---|---|---|---|
| 1 | Dataset : `open()+seek()+BPE-encode` **par sample** (des millions d'appels Python) | CPU-bound, GPU idle — potentiellement le facteur dominant en pratique | `.bin` memmap pré-tokenisé, fenêtres contiguës (`fast_train.py`) |
| 2 | Cohérence O(n) calculée **2×** par forward (softmax + re-calcul logits) | 2× matmuls D→C gaspillés | Single-pass (`FastCognitiveExpertRouter`, `GrowableCoherenceRouter`) |
| 3 | Dispatch MoE : `one_hot (N,K,C)` + passes multiples + boucle naïve | Kernels redondants, mémoire (N,K,C) | 1 `scatter_add_` + boucle actifs, compile-friendly |
| 4 | Micro-batch B=4, T=512 (2k tokens) | GPU sous-alimenté, MFU ~18% | Gros batch + grad accum + 0 padding (fenêtres contiguës) |
| 5 | Pas de `torch.compile`, AdamW non-fusé, pas de TF32 | MFU bas | `enable_fast_mode()` + compile + `adamw-fused`/`adamw8bit` |
| 6 | Phase 1 EDT : 128 experts × 2000 steps **séquentiels** (1 optimizer/expert) | ~heures, parallélisme 0 | `fast_phase1_parallel()` : 8 experts/bloc en batch partagé (~4-6×) |
| 7 | Pas de curriculum, pas de resume, pas de DDP | Redémarrages coûteux, 1 GPU max | Curriculum seq-len, ckpt resumables, DDP `torchrun`-ready |

## 2. Gains attendus (honnêtes, Phase 3 = 1.36B tokens Scénario C)

Rappel physique : 3090 = 142 TFLOPS bf16 ; coût ≈ 6 × 2.38B = 14.3 GFLOP/token
→ plafond absolu ≈ 9 900 tok/s à MFU 100% (inatteignable).

| GPU | Avant (PGSU, MFU~18%) | Après fast stack (MFU 35-45% + PGSU) | Avec infini (par phase suivante) |
|---|---|---|---|
| RTX 3090 | ~10 jours (~1 800 tok/s) | **~2.5-4 jours** (~4 000-5 500 tok/s) | **~0.5-1 jour / milliard** (experts gelés) |
| RTX 4090 | ~3.2 jours | **< 1 jour** (~10-12k tok/s) | **~quelques heures / milliard** |
| H100 | ~6 h | **~2-3 h** (~30-35k tok/s) | **< 1 h / milliard** |

Le **vrai déblocage « milliards de tokens »** vient du combo avec le Phase-Routed MoE :
chaque nouveau milliard (nouvelle phase) n'entraîne que les **nouveaux experts + router**
(~15-25% des FLOPs d'un full-train) au lieu de tout ré-entraîner.

> Note de méthode : sur CPU tiny, legacy et fast sont à parité (~16k tok/s) —
> c'est la **preuve que les maths sont identiques**. Les gains apparaissent sur GPU
> (compile, fused, dataloader, gros batch). Mesurez sur votre GPU :
> `python3 fast_train.py --benchmark`.

---

## 3. `fast_train.py` — training rapide (1 phase)

### 3.1 Dataset binaire (le fix #1)

```python
from fast_train import build_bin_dataset, MMapTokens
build_bin_dataset("corpus.txt", "data/train", tokenizer="cognet_tokenizer.json")
# → data/train.bin (uint16 : 1.36B tokens = 2.7 GB) + data/train.meta.json
ds = MMapTokens("data/train.bin")
batch = ds.sample_batch(B=16, T=512, rng)   # fenêtres contiguës, 0 padding
```

- Streaming, RAM O(1) à la construction ; lecture memmap (le corpus peut dépasser la RAM).
- `uint16` car vocab 16k < 65 536 (2 octets/token).

### 3.2 Router fast (poids-identique)

`FastCognitiveExpertRouter` est une sous-classe du legacy **sans nouveau paramètre** :
`convert_to_fast(model)` + `load_state_dict` → checkpoints interchangeables dans les 2 sens.
Équivalence numérique vérifiée au self-test (écart max `0.00e+00`).

### 3.3 FastTrainer

```python
from fast_train import FastTrainer, FastTrainerConfig, convert_to_fast
model = convert_to_fast(create_cognet_moe_1b(vocab_size=16384))
cfg = FastTrainerConfig(batch_size=16, grad_accum=4, lr=1e-4,
                        compile_mode="reduce-overhead", optimizer="adamw-fused",
                        pgsu_n_active=4,               # 0 = full
                        seq_stages=[(0, 128), (200_000_000, 256), (600_000_000, 512)])
FastTrainer(model, cfg, device="cuda").train(lambda B, T: ds.sample_batch(B, T, rng),
                                             total_tokens=1_361_634_450)
```

- Curriculum seq-len (T=128 rapide au début → 512), LR cosine manuelle,
  PGSU inline, checkpoints resumables (`save_optimizer=True` = resume exact),
  DDP via `torchrun --nproc_per_node=8 fast_train.py --train ...` (sampling par rang).

---

## 4. `phase_routed_moe.py` — Phase-Routed MoE (infini ♾️)

### 4.1 Idée

Le routing est conditionné par la **phase de vie** du modèle :

```
logits(b, t, e) = cohérence_O(n)(x)[b, t, e] + phase_bias[phase_b, e]
```

- `cohérence_O(n)` = la brique CogNet d'origine (query × mean_key). **Pas de gate.**
- `phase_bias` = simple **biais scalaire** (Embedding phases × experts), pas une projection.
- Les nouveaux experts sont de **nouveaux canaux cognitifs** : l'invariant
  « canaux == experts » reste vrai à tout instant, O(n) préservé.

### 4.2 Cycle de vie d'une phase (p ≥ 1)

1. **Croissance** : `+k` experts/bloc (défaut k=2), initialisés par **clone de l'expert
   le plus chargé + bruit** (split de charge) ; `to_channels` étendu (copie + bruit).
2. **Freeze** : experts `[0, C_old)` gelés — `requires_grad=False` sur les experts +
   **hooks de masquage** sur les tranches partagées (`to_channels`, `query`, `key`) +
   `phase_bias` des anciennes phases figé → **anti catastrophic forgetting**
   (vérifié bit-identique au self-test).
3. **Biais** : `phase_bias[p, nouveaux] = +1.0` → la nouvelle phase route d'abord vers
   les nouveaux experts, puis le training ajuste.
4. **Chirurgie optimizer** : reconstruction en préservant les moments Adam des params
   inchangés (cache par nom) ; fallback fresh + re-warmup si 8-bit.
5. **Rehearsal** : 3% d'anciennes données (replay) pour ancrer le router.

### 4.3 Croissance mid-phase automatique (`GrowthController`)

| Déclencheur | Règle (défauts) | Action |
|---|---|---|
| Surcharge | `usage_ema[i] > 1.5 × (top_k/C)` pendant 20 updates | Split : clone l'expert surchargé |
| Incertitude | confiance moyenne `< 0.35` pendant 20 updates | +1 expert (clone busiest) dans le bloc le plus chargé |
| Nouvelle phase | systématique | +k experts/bloc (défaut 2) |
| Garde-fous | `max_experts=32`, cooldown 500 steps, pruning optionnel des morts | jamais de croissance explosive |

### 4.4 Checkpoint infini

`infinite_ckpts/final.pt` = `{model_state_dict, growth, phase_id, arch, tokens_seen}`.
- `load_infinite_checkpoint()` **re-croit automatiquement** avant chargement → un `.pt`
  sauvegardé à C=10 se recharge à C=10 puis peut grandir à C=12, 14… **à l'infini**
  (borné par `max_experts`, extensible en le relevant + `set_growth_state`).
- `load_legacy_checkpoint_into_phase_routed()` : un `.pt` legacy (C fixe, ex.
  `after_phase3_final.pt`) se charge tel quel (vérifié écart `0.00e+00`), puis grandit.

---

## 5. `run_infinite.py` — recettes

### 5.1 Continuer à l'infini (le cas d'usage demandé)

Chaque `--bins` supplémentaire = une phase = de nouveaux experts. Coût marginal
d'une phase ≈ 15-25% d'un full-train (le reste est gelé) :

```bash
# Mois M : phases 0-1 (2B tokens). Mois M+1 : nouvelles données → phase 2 :
python3 run_infinite.py --bins data/p0.bin,data/p1.bin,data/p2.bin \
    --resume infinite_ckpts/final.pt --start-phase 2 \
    --tokens-per-phase 1000000000 --new-experts-per-phase 2
```

### 5.2 Sur une seule 3090 (budget serré)

```bash
python3 run_infinite.py --bins data/p0.bin --tokens-per-phase 1361634450 \
    --batch-size 8 --grad-accum 8 --new-experts-per-phase 0 --compile reduce-overhead
```

(`--new-experts-per-phase 0` = C constant, équivalent fast-train mais avec le protocole infini.)

### 5.3 Multi-GPU

```bash
torchrun --nproc_per_node=8 run_infinite.py --bins data/p0.bin,data/p1.bin \
    --tokens-per-phase 4000000000 --batch-size 16 --compile max-autotune
```

---

## 6. Audit CogNet-native (contraintes préservées)

| Contrainte | Statut |
|---|---|
| Aucune attention inter-token / O(n²) | ✅ routing = cohérence O(n) + biais scalaire |
| Aucun gate transformer-style `Linear(D→N)` | ✅ `phase_bias` = biais, pas une projection du hidden |
| Canaux == experts | ✅ croissance = nouveaux canaux (query/key/to_channels étendus ensemble) |
| Mémoire 3-tier slots fixes | ✅ inchangée |
| Résiduels partout (prérequis EDT) | ✅ inchangés |
| Équivalence numérique legacy | ✅ écart `0.00e+00` (self-tests [3b/7] fast [1/5]) |

## 7. Limites honnêtes / TODO

- Les chiffres GPU du §2 sont des **projections MFU** (pas encore mesurés sur 3090/H100 :
  pas de GPU dans cet environnement). Le `--benchmark` permet de les valider.
- `max_experts=32` par défaut : 16 blocs × 32 experts × 50M ≈ 25B params totaux si saturé
  (actifs/token inchangés : top-2). Surveiller le poids du `.pt` (~2 octets/param en bf16).
  Relever `max_experts` = changer 1 argument, les ckpt restent compatibles.
- DDP + experts gelés : `find_unused_parameters` géré côté FastTrainer ; côté infini,
  les experts gelés ont `requires_grad=False` (exclus du graphe DDP — pas d'erreur).
- Pistes futures : grouped-GEMM (`torch._grouped_mm`) pour le dispatch, offload CPU
  des experts inactifs, distillation des vieux experts vers les nouveaux.
