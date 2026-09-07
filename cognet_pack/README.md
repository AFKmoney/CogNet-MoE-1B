# CogNet-MoE-1B — Pack isolé, fonctionnel et entraînable

> **Oui, CogNet est fonctionnel et entraînable.** Ce dossier contient tout ce qu'il
> faut — modèle, tokenizer, boucle d'entraînement, routage, pagination — vérifié
> en live sur CPU le 2026-09-07 (voir §2). Aucune dépendance au reste du repo.

## 0. Récap du projet (où on en est)

| Couche | Fichiers | Statut |
|---|---|---|
| Modèle CogNet-MoE-1B (7.21B MoE / 2.38B actifs, 100% non-transformer, O(n)) | `cognet_moe.py` + `source/cognet_1b_optimized.py` | ✅ self-test vert, fwd+bwd+audit |
| Tokenizer BPE 16k (FR+EN+code) | `cognet_tokenizer.py` + `.json` | ✅ entraîné, prêt |
| Training rapide (memmap, fast router, compile, PGSU, curriculum, resume) | `fast_train.py` | ✅ **prouvé live sur CPU** (loss 6.27 → 2.39, ckpt strict-load) |
| Training infini (nouveaux experts/phase, freeze anti-oubli) | `phase_routed_moe.py` + `run_infinite.py` | ✅ self-tests verts |
| Hash routing (0 param, bat le routing appris de ~0.8 nats) | `hash_moe.py` | ✅ validé 3/3 seeds (travaux repo) |
| Pagination experts disque (LRU + prefetch async, bit-exact) | `expert_pager.py` | ✅ validé (travaux repo) |
| Experts associatifs sans gradient (Hebb, binaire, MLP local) | *hors pack (recherche)* | v1→v4 : plafond mappé, voir repo principal |

**Réponse honnête sur « CPU rapide »** : ce qui est rapide sur CPU, c'est le
**tiny** (28M params, ~900 tok/s sur 2 cœurs — la démo ci-dessous). Le **vrai 1B**
(2.38B actifs/token) utilise exactement le même code mais exige un **GPU**
(§5). Il n'existe pas de training 1B rapide sur CPU — quiconque prétend le
contraire ment sur les FLOPs.

## 1. Ce que fait chaque fichier

| Fichier | Rôle | Entrée → Sortie |
|---|---|---|
| `source/cognet_1b_optimized.py` | Tronc CogNet d'origine (routing cohérence O(n), mémoire 3-tier, composer) | — (brique de base) |
| `cognet_moe.py` | `CogNetMoE1B` : 16 blocs × 8 experts-canaux top-2, aux/z-loss | self-test via `train.py --self-test` |
| `cognet_tokenizer.{py,json}` | BPE 16k propriétaire + entraînement sur corpus | `corpus.txt` → token ids |
| `fast_train.py` | Dataset `.bin` memmap, `FastCognitiveExpertRouter` (poids-identique), `FastTrainer`, PGSU, curriculum, ckpt resumables, DDP-ready | `.txt` → `.bin` → `.pt` |
| `phase_routed_moe.py` | MoE à phases : `+k` experts/bloc/phase (clone-busiest), freeze + hooks, `phase_bias`, chirurgie optimizer, `GrowthController` | `.pt` → `.pt` grandi |
| `run_infinite.py` | Orchestrateur multi-phases (lifelong, rehearsal 3%) | `p0.bin,p1.bin…` → `final.pt` infini |
| `hash_moe.py` | `convert_to_hash(model)` : remplace le routing appris par hash token/LSH (0 param) | modèle → modèle + stable |
| `expert_pager.py` | Experts paginés sur SSD (safetensors), LRU + prefetch async, hash-ahead exact | E experts « infinis », RAM bornée |
| `train.py` | **Point d'entrée unique** : `--demo`, `--self-test`, délégation `--build-bin`/`--train`/`--benchmark` | voir §3-5 |

## 2. Preuves live (2026-09-07, CPU 2 cœurs, torch 2.14)

```
$ python3 train.py --self-test   → 4/4 verts (modèle, fast, phase-routed, infini)
$ python3 train.py --demo        → tiny 28M, 100k tokens, ~2 min CPU :
  loss 6.30 → ema 3.87 → EVAL 0.43, checkpoint strict-load OK
  (run équivalent 200k : 6.27 → 4.91 step 50 → EVAL 2.39)
```

## 3. Installation

```bash
pip install -r requirements.txt   # torch, numpy, tokenizers, safetensors
# GPU (optionnel, sinon fallback AdamW auto) : pip install bitsandbytes
```

## 4. Recette CPU — tiny (fonctionne partout, minutes)

```bash
# Preuve complète en UNE commande (~2-3 min, 100k tokens) :
python3 train.py --demo

# Avec votre corpus :
python3 train.py --build-bin --txt corpus.txt --out data/mono --tokenizer cognet_tokenizer.json
python3 train.py --train --tiny --bin data/mono.bin --tokens 500000 \
    --batch-size 8 --seq-len 128 --lr 3e-4
# → demo_out/ckpts/final.pt (rechargeable, resumable)
```

Le tiny : D=256, 4 blocs, 8 experts top-2 (28M params, 9.5M actifs/token).
Débit mesuré : ~900 tok/s sur 2 cœurs CPU. Parfait pour valider un corpus,
un tokenizer, un hyperparamètre — pas pour un vrai LLM.

## 5. Recette GPU — le vrai 1B (heures/jours, pas minutes)

```bash
# 1. Pré-tokeniser UNE fois (tue le goulot CPU) :
python3 train.py --build-bin --txt corpus.txt --out data/p0 \
    --tokenizer cognet_tokenizer.json

# 2. Training joint (1.36B tokens = Scénario C Chinchilla-EDT) :
python3 train.py --train --bin data/p0.bin --tokens 1361634450 \
    --batch-size 16 --grad-accum 4 --compile reduce-overhead
# RTX 3090 : ~2.5-4 jours | RTX 4090 : < 1 jour | H100 : ~2-3 h (projections MFU)

# 3. OU multi-GPU :
torchrun --nproc_per_node=8 train.py --train --bin data/p0.bin \
    --tokens 4000000000 --batch-size 16 --compile max-autotune
```

Le 1B : D=2048, 16 blocs, 8 experts top-2 (7.21B params, **2.38B actifs/token**).
`--tiny` enlevé = exactement ce modèle. Mêmes checkpoints, même code.

## 6. Entraînement infini (lifelong, phases)

Chaque nouveau milliard de tokens = une phase = `+k` experts/bloc, anciens gelés
(bit-identique vérifié, anti catastrophic forgetting), coût marginal ~15-25%.

```bash
python3 train.py --build-bin --txt corpus_p1.txt --out data/p1 --tokenizer cognet_tokenizer.json
python3 run_infinite.py --bins data/p0.bin,data/p1.bin \
    --tokens-per-phase 1000000000 --new-experts-per-phase 2 --compile reduce-overhead
# Plus tard, avec de nouvelles données :
python3 run_infinite.py --bins data/p0.bin,data/p1.bin,data/p2.bin \
    --resume infinite_ckpts/final.pt --start-phase 2 --tokens-per-phase 1000000000
```

## 7. Options avancées (toutes compatibles entre elles)

```python
import sys; sys.path.insert(0, "source")     # une fois par session
from cognet_moe import create_cognet_moe_1b
from hash_moe import convert_to_hash          # routing hash : 0 param, + stable
from fast_train import convert_to_fast        # router fast : poids-identique
from expert_pager import ExpertPager          # experts sur disque (E illimité)

model = create_cognet_moe_1b(vocab_size=16384)
model = convert_to_hash(model, mode="token")  # ou "lsh" ; validé 3/3 seeds
model = convert_to_fast(model)                # maths identiques, kernels rapides
# + ExpertPager pour dépasser la RAM/GPU : préfetch async masqué (~4× anti-stall)
```

## 8. Formats

- `.bin` + `.meta.json` : tokens pré-tokenisés (uint16, 2 o/token, lecture memmap).
- `final.pt` : `{model_state_dict, step, tokens_seen, loss_ema, …}` (+ état de
  croissance pour l'infini). `save_optimizer=True` = resume bit-exact.
- Pages pager : `blkXX_expYYY.safetensors` (1 fichier/page, LRU + prefetch).

## 9. Limites honnêtes

- Les débits GPU du §5 sont des **projections MFU**, pas encore mesurés (pas de
  GPU dans l'env de dev). Validez avec `python3 train.py --benchmark`.
- L'infini sature à `max_experts=32` par défaut (relevable, ckpt compatibles).
- Le chemin EDT historique (`run_cognet_moe.py`, 4 phases) est **hors pack** :
  les chiffres rapides ci-dessus viennent du fast stack qui le remplace.
  Voir le repo principal pour l'historique et la réfutation EDT.
- La recherche « experts associatifs sans gradient » (Hebb, Hamming ÷32, MLP
  local) est aussi hors pack : prototypes validés sur tronc tiny uniquement,
  pas encore substituables au 1B. Conclusion v4 : la superposition directe
  apprend, la backprop locale profonde ne mord pas.

## 10. Vérifier que tout marche

```bash
python3 train.py --self-test   # modèle + fast + phase-routed + infini (CPU, ~1 min)
python3 train.py --demo        # entraînement tiny réel + éval (CPU, ~3 min)
```
