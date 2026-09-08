# Entraîner CogNet sur CPU — Rapport (FR)

**Date** : 2026-09-08 · **Module livré** : [`cpu_stack.py`](cpu_stack.py)
**Question** : trouver une façon d'entraîner CogNet **sans GPU**, en exploitant des
techniques de pointe et les propriétés mathématiques propres à l'architecture.
**Réponse courte** : oui — mais pas le 1B naïf. La physique des FLOPs condamne
le 1B brut sur CPU (~300+ ans), **et** l'architecture CogNet possède des propriétés
mathématiques inédites (absentes des transformers) qui changent le coût au premier
ordre. Ce rapport les démontre, les implémente (8/8 self-tests verts, sonde 3 seeds,
e2e complet validés), et trace 3 routes honnêtes : **(A)** un CogNet natif-CPU
entraîné from scratch en **jours/semaines** sur desktop, **(B)** la croissance
d'experts **sans gradient** paginés sur disque (frontière recherche, déjà amorcée
par `assoc_experts.py`), **(C)** le **sharding zéro-communication** multi-CPU
permis par le hash routage (Exp C du repo) + location d'un serveur CPU à 64-128 cœurs
(quelques euros/heure, pas de GPU du tout).

---

## 1. La physique d'abord — pourquoi le 1B naïf est mort sur CPU

Cible EDT Scénario C du repo : 1.36B tokens × coût 6·N_actif ≈ 14.3 GFLOP/token
≈ **2×10¹⁹ FLOPs**. Un CPU délivre ~3 GFLOP/s (cette VM 2 cœurs) à ~800 GFLOP/s
(gros serveur 128 cœurs, eager fp32 réaliste) :

| Machine | tok/s 1B (mesuré/projeté) | Temps pour 1.36B tokens |
|---|---|---|
| Cette VM 2 cœurs | ~0.4-1 | ~50-100 ans |
| Desktop 16-32 cœurs | ~10-30 | ~1.5-4 ans |
| Serveur 128 cœurs | ~30-70 | ~7 mois-1.5 an |

Le constat du README (« CPU only ~10-30 tok/s — useless ») est **physiquement
correct** : aucune astuce logicielle ne crée des FLOP/s. Il faut donc changer le
**compte de FLOPs par token**, pas seulement l'implémentation. C'est exactement ce
que permettent les propriétés ci-dessous — et pour l'échelle 1B, il faut en plus
changer de métrique : la **capacité** (params totaux, sur disque) plutôt que
l'activation, cf. route B.

## 2. Les propriétés mathématiques exploitées (cœur du travail)

### P1 — Associativité du produit matriciel : la projection `to_channels` est repliable (inédit, EXACT)

Un bloc CogNet-MoE calcule, pour l'expert `e` sélectionné :

```
out_e = W_e·x + Norm( W_down · SwiGLU( U_e · (W_e·x) ) )
         └─résiduel─┘      └── w_gate_up·(to_channels) ──┘
```

`to_channels = Linear(D → C·D)` ≡ C matrices empilées `W_e (D×D)`. Par
**associativité** du produit matriciel : `U_e·(W_e·x) = (U_e·W_e)·x = A_e·x`.
On **plie** `A_e = U_e·W_e` à l'init :

- Ce n'est **pas une approximation** : rang(A_e) ≤ D = rang max de U_e ∘ id — la
  classe de fonctions est **identique** ; vérifié bit-par-bit (self-test [2] :
  écart max 2.3e-6 fp32, arrondi pur).
- `fold_residual=True` : on garde le résiduel `W_e·x` mais on ne le calcule que
  pour les **K experts sélectionnés** : `K·D²` MACs au lieu de `C·D²` → **÷4** sur
  la projection (C=8, K=2).
- `fold_residual=False` (« simplify », from-scratch) : le résiduel intra-expert
  est **redondant** avec le résiduel de bloc `out = x + Norm(Σ w_e·out_e)` — on
  supprime `W_e` entièrement. L'invariant « canaux == experts » survit : chaque
  expert garde sa propre vue de `x` (sa matrice A_e pliée = sa projection-canal).
- **Mesure (MACs/token des blocs, 1B)** : legacy 2148M → fold_res 1745M (−19%) →
  simplify 1611M (−25%) — *gratuits*, aucune approximation.
- Bonus params : 7.21B → 6.68B totaux ; les experts « simplify » ont **exactement
  la forme state_dict de FusedSwiGLU** → compatibilité `expert_pager.py` immédiate.

### P2 — Le routage est une fonction PURE du token (déjà prouvé au repo, généralisé ici)

`token_hash_experts` assigne K=2 experts par token par avalanche entière — **le
résultat est connu avant tout calcul**. Conséquences sous-exploitées jusqu'ici :

1. Zéro param/gradient de routing, zéro aux/z-loss, zéro collapse (déjà documenté) ;
2. **Le dispatch ne calcule que les experts sélectionnés** : combiné à P1, il n'y a
   plus AUCUNE matrice dense toujours-active dans le chemin expert (le legacy
   calculait les 8 canaux pour n'en garder que 2) ;
3. **Pré-sharding** : l'assignation données→experts est décidable hors ligne →
   workers CPU expert-parallèles **sans communication** (Exp C du repo :
   102k tokens shardés en 5.2 ms) ; pour les phases à experts découplés (P1/EDT-like,
   route B) c'est du linéaire-machine ; pour le joint, le motif d'échange devient
   statique et pré-fetchable (même propriété qui rend le pager exact) ;
4. Le hash-ahead de batch N+1 rend le **prefetch disque exact** (pager).

Re-mesuré ici dans MON protocole (inits appariées, 3 seeds) : fold+hash
**2.72** vs legacy cohérence **3.47** → **−0.75 nats** en supprimant le router.
Le repo avait mesuré −0.8 (HASH 2.08 vs SCRATCH 2.89) : reproduction indépendante.

### P3 — Parcours sous-déterminé du graphe — LISA/PGSU avec détachement strict

Observation NeurIPS 2024 : **LISA** (Pan et al.) égalise/bat le full-parameter en
n'entraînant que γ couches échantillonnées par période. Le repo possède déjà
**PGSU** (rotation de blocs actifs). J'ajoute le mode **« lisa-detach »** : le
préfixe gelé est exécuté **sans graphe puis détaché** — le backward est
*strictement coupé* à la frontière (self-test [7] : 0 grad sous le préfixe, API
prouvée). Coût : fwd_total + 2×(γ/B·MAC_blocs + MAC_tronc+tête) au lieu de ×3 —
à γ/B=1/4, ~**−45% de FLOPs d'entraînement**. Taxe qualité mesurée (γ=1/2, tiny,
3 seeds) : **+0.25 nats pour ×1.6-2 tok/s** (2.98 vs 2.73) — point de Pareto,
pas défaut de mécanisme. À B=12-16, γ=2-4 : cut 60-85% du backward (à valider par
échelle, c'est la même famille que PGSU déjà en production ici).

### P4 — Softmax échantillonné à partition exacte (tête liée V=16384)

Logits pleins = `h·Eᵀ` = V·D MACs/token aller **et** scatter dense au retour.
Tête d'entraînement : cíbles + S négatifs uniformes **avec remise, multiplicités
conservées**, correction `log(S/V)`, masquage des vrais-accidentels **par
échantillon**. Théorème vérifié (self-test [5]) : l'estimateur de partition est
**exactement sans biais** (|E[Ẑ]−Z|/Z = 0.0003, 128 essais) — le piège classique
(dédupliquer les négatifs ou exclure les cibles du batch) brise la correction
(|E[Ẑ]−Z|/Z = 0.39 mesuré avant fix ; documenté dans le code). Le log (Jensen)
garde un biais ∝1/S, ~0 à S=V/2, faible à S≥√V. Taxe qualité mesurée (V=64/S=16,
régime défavorable) : +0.37 nats — à réserver **aux gros vocabulaires** où la tête
pèse >10% des MACs (à V=16k, S=512 : ratio MACs ÷32, bruit bien moindre).

### P5 — Smart-init P1 **sans données** (validé précédemment 4/4 seeds, ici : inutile sous hash)

P1 d'EDT (experts → identité sur bruit synthétique, **0 token de corpus**) fut la
seule phase EDT validée (+0.50 nat vs scratch router-appris). Re-mesure honnête sur
routeur **hash** : simplify 2.73 vs simplify+P1 2.73 — **gain nul, 3/3 seeds**.
Interprétation : le bénéfice P1 stabilisait le couplage poule-œuf router↔experts ;
le hash est stable des le step 0, donc P1 n'a plus rien à réparer. **Conclusion
pratique** : sous hash, sauter P1 — économie nette. La fonction
`p1_identity_init` reste livrée (self-test [8] : MSE 1.87→0.31 sans données).

### P6 — États d'optimiseur 8-bit **sur CPU** (bitsandbytes est GPU-only)

`AdamW8bitCPU` : m,v en int8 + échelles fp32 par bloc de 2048 (style bnb), pur torch.
Mémoire optim ≈ **2.0 o/param** vs 8 (÷4). Piège trouvé et fixé : `v` par blocs
s'effondre à 0 pour les petites entrées (g² disperse sur 4 ordres de grandeur →
div/0 dans l'update) → **plancher demi-LSB** `v̂=(|q|+0.5)/127·amax`, borne
mathématique ≤ +50% du LSB, dénominateur ≥ 6% de l'échelle du bloc. Mesure
(self-test [6]) : MSE test 0.021 vs 0.011 fp32 (tâche structurée 120 steps) —
adéquat, pas bit-égal ; à réserver aux runs où la RAM optimiseur contraint.

### P7 — Cœurs bas-rang / ternaire b1.58 (pistes noyau, mesurées en qualité)

- **Bas-rang** (SwiGLU factorisé D→r→ff) : `simplify+lr32` = **2.99** vs 2.73 →
  +0.26 nat pour −33% MACs experts (tiny, r=D/2) — vraie taxe, à doser ; livré.
- **Ternaire b1.58** (poids {-1,0,+1}·s, STE) : BitNet a prouvé le from-scratch à
  2.4B params/4T tokens à parité pleine précision, et bitnet.cpp mesure
  **×2.37-6.17 sur CPU x86** en inférence (÷8 bande passante). En eager PyTorch on
  ne touche pas ce gain (pas de kernel) : option livrée en mode **qualité-seule**
  (TernaryLinear), speedrun = travail kernel documenté (§7).

### P8 — Mur mémoire → mur disque (déjà prouvé au repo, et ratifié par l'industrie)

`expert_pager.py` = LRU + prefetch hash-ahead exact + writeback, **bit-exact**,
anti-stall 4.4×, pages int8 ×3.72. La capacité d'un MoE vit sur NVMe ; la RAM ne
voit que S slots. Signe que la vision était juste : **llama.cpp a RFC exactement
cette idée en 2026** (« MoE offload to disk with on-demand paging », mai 2026) et
HOBBIT (2024) fait pareil en inférence — le repo l'implémentait **pour
l'entraînement** avant. Les experts « simplify » (forme FusedSwiGLU) sont
page-compatibles tels quels.

## 3. Ce qui a été construit — `cpu_stack.py` (~900 lignes)

| Composant | Rôle | Preuve |
|---|---|---|
| `FoldedExpert` / `FoldedHashRouter` | Pliage P1 + hash P2, drop-in `CogNetMoEBlock` | test [2][3][4] |
| `FoldedHashRouter.from_legacy` | Pliage EXACT d'un checkpoint existant | test [2] (2.3e-6) |
| `convert_to_cpu(model, ...)` | Swap routers (init neuve ou `from_trained`) | test [4] |
| `sampled_ce_loss` | P4, partition sans biais | test [5] |
| `AdamW8bitCPU` | P6, mémoire ÷4 + plancher demi-LSB | test [6] |
| `CPUTrainer` | Plomberie ids, LISA-detach, LR cosine, ckpt | test [7] + e2e |
| `p1_identity_init` | P5 synthétique (recherche/ablation) | test [8] |
| `probe_run` | **Protocole EXACT `refute_edt.py` volet 1** | §4 |
| `benchmark` / `projection_tables` | tok/s mesurés + tables MACs | §4-5 |

Invariants CogNet vérifiés : pas d'attention, pas de gate appris `Linear(D→N)`,
canaux == experts, résiduels de bloc partout, mémoire 3-tier et composer intacts,
O(n), drop-in pager.

## 4. Résultats mesurés (cette VM 2 cœurs, torch 2.14 CPU)

### 4.1 Self-tests : **8/8 verts** (`--self-test`, ~6 s)

Hash déterministe+balancé [0.123,0.129] · pliage exact 2.26e-6 · MACs −19%/−25% ·
grads == experts assignés · partition sans biais 0.0003 · AdamW8bit ×0.25 mémoire ·
LISA-detach 0-grad préfixe · P1 MSE 1.87→0.31 sans données.

### 4.2 Sonde sans fuite (protocole EXACT du repo, 200k tokens/bras, 3 seeds) — `cpu_probe_results.json`

| Bras | seed 0 | seed 1 | seed 2 | Moyenne | Params | vs legacy |
|---|---|---|---|---|---|---|
| legacy (cohérence apprise) | 3.491 | 3.502 | 3.425 | **3.47** | 1.68M | — |
| **fold_res+hash (pliage EXACT)** | 2.720 | 2.772 | 2.673 | **2.72** | 1.68M | −0.75 |
| **simplify+hash (−25% MACs)** | 2.697 | 2.792 | 2.687 | **2.73** | 1.60M | −0.74 |
| simplify+P1 (0-token init) | 2.746 | 2.764 | 2.672 | 2.73 | 1.60M | −0.74 |
| simplify+lowrank32 | 2.984 | 3.033 | 2.952 | 2.99 | 0.93M | −0.48 |
| simplify+LISA(γ=1/2) | 2.974 | 3.033 | 2.942 | 2.98 | 1.60M | −0.49 |
| simplify+sampled16 (tête éch.) | 3.089 | 3.108 | 3.108 | 3.10 | 1.67M | −0.37 |

Contrôle motif-inédit ≈ 4.20-4.28 partout (= chance ✓ la tâche mesure la
mémorisation). Références repo à 200k : SCRATCH 2.88 · P1-ONLY 2.39 · HASH-TOKEN
1.87 · ASSOC-v3 2.88 (protocole identique, campagnes distinctes — comparaison dure
= au sein d'une même campagne).

**Lecture** : (1) le pliage EXACT ne coûte rien (2.72≈2.73, bruit) — ses −19% de
MACs sont gratuits ; (2) **la suppression complète de W_e/to_channels (−25% MACs,
−5% params) est gratuite en qualité** (2.73 vs 2.72, Δ=0.01 ≪ bruit inter-seed
~0.05) — la « vue canal » est bien absorbable par les experts ; (3) les options de
frugalité suivantes (low-rank, LISA, sampled) sont des points de Pareto
qualité↔vitesse, toutes sous le legacy malgré leur taxe.

### 4.3 Débit tiny fwd+bwd (relatif machine) — `cpu_bench.json`

legacy 5.1k tok/s → fold_res 6.5k (**×1.26**) → simplify 6.5k (×1.27) →
**LISA 13.1k (×2.56)** → lowrank(D/4) 7.5k (×1.46). À l'échelle 1B les ratios
MACs (−19…−56%) dominent ce que le tiny sous-estime (le routeur y est sur-compté).

### 4.4 E2E complet (`--e2e`, 40k tokens)

Stream 4 motifs, simplify+hash+sampled16+AdamW8bit+LISA(1/2→2/2) : loss
4.16→3.17, 9-12k tok/s, **checkpoint `strict=True` reload OK + reprise** (14.3k
tokens repris, tok/s corrigé). La plomberie entière est démontrée.

## 5. Projections honnêtes (`--tables` ; params tronc mesurés par instanciation)

| Config | Params tot. | Actifs/tok | MAC fwd simplify | Desktop 8c full-train | 16-32c LISA+échant. |
|---|---|---|---|---|---|
| CogNet-MoE-1B (réf) | 6.68B | 1.85B | 1845M | 3-7 tok/s | 17-41 tok/s |
| cognet-cpu-S (512d×8b) | 0.22B | 68M | 68M | 73-196 tok/s | 478-1196 tok/s |
| cognet-cpu-M (768d×12b) | 0.72B | 209M | 209M | 24-64 | 148-371 |
| cognet-cpu-L (1024d×16b) | 1.69B | 478M | 478M | 10-28 | 64-160 |
| cpu-M/lr256 | 0.30B | 106M | 106M | 47-126 | 275-689 |

**Temps pour 300M tokens** (LISA γ=B/4 + tête S=512, fourchettes efficacité eager) :

| Machine | cpu-S | cpu-M | cpu-L | cpu-M/lr256 |
|---|---|---|---|---|
| Desktop 8c AVX2 | 9-24 j | 29-78 j | 68-181 j | 16-42 j |
| Desktop 16-32c AVX512 | **3-7 j** | 9-23 j | 22-54 j | **5-13 j** |
| Serveur 64-128c (location, pas de GPU) | **1-2 j** | 3-8 j | 7-18 j | 2-4 j |

## 6. Les 3 routes CPU (le plan complet)

### Route A — qualité d'abord : **CogNet natif-CPU from scratch** (la réponse principale)

Ne pas porter le 1B sur CPU ; entraîner un CogNet **dimensionné pour le CPU** avec
la stack complète. Recette desktop (16-32 cœurs, ~2 semaines) :

```bash
pip install torch tokenizers numpy   # pas de GPU, pas de bitsandbytes
PYTHONPATH=source:. python3 cpu_stack.py --self-test      # 8/8 ~10 s
# 1) corpus → .bin (infra existante)
python3 fast_train.py --build-bin --txt corpus.txt --tokenizer cognet_tokenizer.json --out data/cpu_p0
# 2) modèle cpu-M/lr256 + stack (API python ci-dessous), training CPUTrainer
python3 - <<'EOF'
import sys; sys.path.insert(0, "source"); sys.path.insert(0, ".")
import torch
from cognet_moe import CogNetMoE1B
from cpu_stack import convert_to_cpu, CPUTrainer, CPUTrainerConfig
from fast_train import MMapTokens
torch.set_num_threads(24)   # vos cœurs
model = CogNetMoE1B(vocab_size=16384, hidden_dim=768, num_blocks=12, num_channels=8,
                    channel_dim=768, ff_dim=3072, max_seq_len=512, key_dim=96,
                    n_experts=8, top_k=2, use_gradient_checkpointing=False)
convert_to_cpu(model, mode="token", fold_residual=False, low_rank=256)
ds = MMapTokens("data/cpu_p0.bin")
cfg = CPUTrainerConfig(batch_size=8, seq_len=256, lr=4e-4, warmup_tokens=2_000_000,
                       sampled_softmax_negs=512,
                       lisa_schedule=[(0, 3), (100_000_000, 6), (200_000_000, 12)],
                       optimizer="adamw8bit-cpu")
import numpy as np
rng = np.random.default_rng(0)
trainer = CPUTrainer(model, cfg)
trainer.train(lambda B, T: ds.sample_batch(B, T, rng), total_tokens=300_000_000)
trainer.save("cognet-cpu-M-final.pt")   # ~0.3B params MoE, ~100M actifs/token
EOF
```

- **Capacité vs activation** : 0.30B params totaux (les 8 experts comptent), ~106M
  MACs/token actifs — les experts inactifs n'existent qu'en RAM (ou disque, route B).
- Checkpoint `strict`-compatible ; génération/inference identiques au modèle legacy
  (`model.generate` marche tel quel — le routage token-hash n'a besoin que des ids).
- **Montée en échelle sans douleur** : croissance par phases (infra
  `phase_routed_moe` — le Phase-Routed MoE accepte des routeurs hash : `+k`
  experts/bloc par nouveau corpus, coût marginal 15-25%, anciens gelés — **le
  lifelong learning du repo devient le lifelong learning du CPU**).

### Route B — échelle d'abord : capacité **infinie paginée + apprentissage local forward-only** (frontière recherche du repo)

Le vrai « 1B+ sur CPU » n'est pas un modèle dense-activé, c'est un **tronc actif
modeste + une ferme d'experts sur disque** qui grossit sans fin :

- **Machine déjà construite** : `expert_pager.py` (E experts sur NVMe, int8,
  prefetch exact — les pages « simplify » sont FusedSwiGLU-forme-compatibles) +
  `assoc_experts.py` v1→v4 (Hebb/delta-rule, zéro backprop dans les pages, zéro
  moments Adam paginés — le gap mesuré vs experts-gradient est passé de +1.68 à
  **+1.02 nats**, plafond de la ligne lookup-prototype mappé).
- **Le point dur identifié par le repo** : l'expressivité du readout, pas le
  mécanisme. La voie v3/v4 (crédit exact 1-couche + MLP local zéro-init) est le
  front à pousser ; chaque progrès ici = le CPU gagne de la capacité **sans payer
  de backprop** (coût marginal d'un expert ≈ 2×fwd au lieu de 3×).
- Hypervecteurs binaires (clés Hamming ÷32, CPU-natif ALS) : infra prête
  (assoc v3 binaire bit-exact).

### Route C — cluster d'abord : **sharding zéro-communication + serveur CPU loué**

- Le hash rend l'assignation pré-computable → `shard_by_hash` (Exp C) : phases à
  experts découplés (P1-like, assoc, pager writeback) = **parallélisme machine
  quasi-parfait**. Joint : motif d'échange statique/prefetchable, pas de
  re-sharding appris — travail d'ingénierie borné.
- Le hack économique : **un serveur CPU 64-128 cœurs se loue pour une fraction du
  prix GPU** → cpu-S en 1-2 jours, cpu-M/lr256 en 2-4 jours (table §5). Zéro GPU,
  zéro CUDA, zéro disponibilité GPU requise.

## 7. Limites honnêtes / chantier

1. **Leak volet 0 rappel** : CogNet est bidirectionnel ; les losses shifted-LM
   absolues du repo (et de l'e2e smoke) restent non-autorégressives. Les
   comparaisons relatives à budget égal (toutes ici) sont valides ; le chantier de
   fond (cohérence causale par cumsum, composer causal) reste ouvert — et il est
   *orthogonal* à la stack CPU (drop-in).
2. **LISA-γ/B petit (1/4-1/8) à profondeur réelle** : mécanisme prouvé, qualité à
   valider par run pilote (taxe mesurée +0.25 à γ=1/2, tiny).
3. **Ternaire b1.58** : validité from-scratch prouvée par Microsoft à l'échelle ;
   ici livré en option qualité — le gain ×2.4-6 demande des kernels C/AVX-512
   (travail identifié, non faké).
4. **AdamW8bitCPU** : pas bit-égal à fp32 (MSE ×2 sur tâche 120-steps) ; pour les
   runs longs critiques, valider sur 1% puis basculer.
5. **Projections** : les tok/s §5 sont des fourchettes eager-honnêtes (calibrer :
   `--bench`) ; MFU des gros GEMM lisa/low-rank peut dépasser les hypothèses
   (surtout low-rank : GEMMs plus petites = efficacité moindre — compté
   conservateur).
6. **DDP CPU multi-nœuds** non implémenté (gloo marche mais non testé ici) ; le
   sharding hash est la voie recommandée (route C).

## 8. Reproduction

```bash
PYTHONPATH=source:. python3 cpu_stack.py --self-test    # 8/8, ~10 s
PYTHONPATH=source:. python3 cpu_stack.py --probe --seeds 3   # sonde repo-exacte, ~6 min
PYTHONPATH=source:. python3 cpu_stack.py --bench        # tok/s tiny, ~1 min
PYTHONPATH=source:. python3 cpu_stack.py --tables       # projections 1B/cpu-*
PYTHONPATH=source:. python3 cpu_stack.py --e2e --tokens 40000   # e2e + ckpt, ~1 min
```

Artefacts : `cpu_probe_results.json`, `cpu_bench.json`.

---

*Verdict : « entraîner CogNet sans GPU » a une réponse constructive — pas en
déplaçant le 1B sur CPU, mais en laissant l'architecture payer ce qu'elle utilise :
le pliage associatif (exact) supprime la projection dense, le hash rend le routage
et le sharding gratuits et pré-computables, LISA coupe le backward, la tête s'échantillonne
sans biais, l'optimizer se quantifie ÷4, et la capacité s'exile sur disque avec des
experts qui apprennent sans gradient. Addition : un vrai LM-MoE CogNet-native,
entraîné from scratch sur un desktop CPU en jours-semaines, extensible à l'infini,
0 GPU. Self-tests 8/8, sonde 3 seeds, e2e validé — aucun chiffre de ce rapport
n'est une promesse non mesurée à l'échelle où elle est revendiquée.*

**Références** : Hash Layers (Roller et al., 2021) · LISA (Pan et al., NeurIPS
2024) · BitNet b1.58 (Ma et al., 2024) + bitnet.cpp (Wang et al., 2024, ×2.37-6.17
x86) · Sampled softmax (Jean et al., 2015) · Switch/ST-MoE (aux/z-loss) · llama.cpp
RFC « MoE offload to disk » (2026) · HOBBIT (2024) · et travaux internes du repo :
`HASH_ROUTING_REPORT.md`, `EXPERT_PAGER_REPORT.md`, `ASSOC_EXPERTS_REPORT.md`,
`EDT_REFUTATION_REPORT.md` (protocole sonde repris à l'identique).
