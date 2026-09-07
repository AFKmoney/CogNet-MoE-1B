# Routing par hashage (sans gradient) — Rapport

**Date** : 2026-09-07 · **CPU tiny** (64d × 2 blocs × 4 experts) · **Scripts** : `hash_moe.py`, `hash_experiment.py`

Idée testée : remplacer le router appris par une **fonction de hashage déterministe**
qui assigne chaque token à ses experts — zéro gradient de routing — pour couper le
goulot GPU et entraîner sur CPU (cf. « Hash Layers », Roller et al., 2021).

---

## Verdicts

| Claim | Verdict |
|---|---|
| Hash-LSH ≈ routing appris (à budget égal, tâche sans fuite) | ✅ **DÉPASSÉ : hash GAGNE de 0.8 nat, 3/3 seeds** |
| Zéro param de routing, zéro aux-loss, zéro collapse | ✅ validé (usage 0.40-0.56, 0 mort) |
| Pré-découpage données SANS modèle (sharding CPU sans communication) | ✅ 102k tokens shardés en 5.2 ms, shards équilibrés |
| 100% sans gradient (random features + moindres carrés) | ⚠️ **faisable mais faible** : 3.98 vs chance 4.16 (vs 2.89 avec gradients) |

**Conclusion** : le sweet spot = **routing par hash (sans gradient) + experts entraînés par
gradient**. Le hash ne supprime pas les matmuls des experts, mais il supprime le router,
toutes ses pathologies, et — point clé — permet le **sharding expert-parallèle sur CPU
avec ZÉRO communication** (l'assignation est connue avant tout calcul). Le 100% sans
gradient reste un pari recherche (écart qualité énorme).

---

## Exp A — HASH-LSH vs SCRATCH (sonde sans fuite, 200k tokens, inits appariées)

| Bras | seed 0 | seed 1 | seed 2 | Moyenne |
|---|---|---|---|---|
| SCRATCH (cohérence apprise) | 2.890 | 2.896 | 2.852 | 2.88 |
| **HASH-LSH** (projection fixe, 0 gradient) | 2.058 | 2.103 | 2.087 | **2.08** |

Mieux même que P1-ONLY (2.39, cf. `EDT_REFUTATION_REPORT.md`). Débit identique (~23k tok/s :
le routing est une fraction mineure du compute à cette échelle — le gain est qualité +
simplicité + parallélisme, pas tok/s single-core).

**Pourquoi le hash gagne ici** : le router appris souffre du dilemme poule-œuf (router stable →
experts spécialisés → router stable) + bruit du noisy top-k ; le hash donne à chaque expert
un sous-ensemble **stable et cohérent** (localité LSH : états similaires → mêmes experts) dès
le step 0 → spécialisation immédiate. Conforme à la littérature Hash Layers (fixe ≈ appris),
ici supérieur car le router cohérence tiny est peu expressif.

**Note** : le LSH est « semi-fixe » (projection aléatoire fixe, mais appliquée à un hidden
state qui évolue) — toujours sans gradient (detach), avec un bonus d'adaptativité gratuit.

## Exp B — 100% sans gradient (trunk aléatoire figé + tête en forme close)

`W = lstsq(H, Y)` sur 50k échantillons (0.1s CPU) → eval **3.98** vs chance 4.16.
Ça apprend *un peu* (+0.18 nat), très loin des gradients (2.89). Honnêteté : remplacer les
gradients partout n'est pas viable pour de la qualité LM aujourd'hui — le hash doit rester
cantonné au **routing**, où il excelle.

## Exp C — Débit + pré-sharding

- fwd+bwd : hash 16.7k tok/s vs appris 15.4k (+8%, router supprimé), −1024 params à tiny scale
  (query/key), et surtout : **0 param routing à toute échelle, 0 aux/z-loss, 0 collapse possible**.
- Pré-sharding : 102 400 tokens → 8 shards en **5.2 ms sans aucun modèle**
  ([13880, 12181, 14648, 10115, 12744, 15395, 10866, 12571]).
  Recette CPU-scale : `shard_by_hash()` → 1 worker CPU/shard → concaténer. Zéro sync.

## Recette CPU (sans GPU)

```python
from hash_moe import convert_to_hash, shard_by_hash
model = convert_to_hash(model, mode="lsh")   # ou "token" (balance parfaite, ids requis)
shards = shard_by_hash(all_token_ids, n_experts=8)  # découpe avant tout training
# worker e : entraîne experts[e] (+to_channels partagé synchro rare) sur shards[e]
```

## Limites

- Tiny CPU + sonde : l'avantage hash doit être re-validé à grande échelle et sur vraie LM
  (avec objectif sans fuite — cf. chantier causalité dans `EDT_REFUTATION_REPORT.md`).
- LSH suppose des hidden states à peu près isotropes pour la balance — monitorer `usage`
  (ici sain : 0.40-0.56) ; sinon mode `token` (balance parfaite).

## Reproduction

```bash
PYTHONPATH=source:. python3 hash_moe.py                  # self-test balance/déterminisme
PYTHONPATH=source:. python3 hash_experiment.py           # ~1 min : A + B + C
```
