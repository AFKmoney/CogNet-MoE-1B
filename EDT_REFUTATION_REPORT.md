# EDT : tentative de réfutation sur CogNet-MoE — Rapport

**Date** : 2026-09-07 · **Modèle** : 64d × 2 blocs × 4 experts top-2, CPU · **Script** : `refute_edt.py`

EDT ayant déjà été réfuté sur Fractus, ce rapport tente la réfutation ici avec un protocole
corrigé (budgets strictement égaux, tâche sans fuite, 2-4 seeds, inits appariées).

---

## Verdicts (résumé)

| Claim testé | Verdict |
|---|---|
| L'ancienne validation (EDT=3.60 vs scratch=1.50) mesure un vrai apprentissage | ☠️ **INVALIDE — fuite prouvée** (volet 0) |
| Le pipeline EDT complet (4 phases) bat le joint-training à budget égal | ❌ **RÉFUTÉ** : 3.50 vs 2.89 (volet 1) |
| Le gain 35× tokens est observable | ❌ **NON observé** (EDT consomme 37% du budget en pré-phases et perd quand même) |
| La Phase 1 (experts → identité + perturbation) aide | ✅ **VALIDÉ** : P1-seule 2.39 vs 2.89, 4/4 seeds |
| Les Phases 2a/2b (shifted-LM) aident | ❌ **NUISENT** : P2B-seule 3.24 ; full EDT pire que P1-seule de 1.1 nat |
| Les fixes V2 (perturbation, 2a×50, aux 0.05) sauvent EDT | ❌ **NON** : EDT-V2 (3.53) ≈ EDT-NOFIX (3.47), tous deux < scratch |

**Conclusion nette** : le pipeline EDT complet est réfuté comme méthode d'entraînement principal,
mais la Phase 1 est validée comme *smart init*. Le coupable est identifié : les phases 2a/2b,
dont l'objectif shifted-LM **fuit** sur une architecture bidirectionnelle (volet 0).

---

## Volet 0 — ☠️ Fuite prouvée : l'ancienne validation est invalide

**Protocole** : shifted-LM standard sur bruit i.i.d. Zipf-136 (mêmes données que `validate_edt.py`).
Sur données i.i.d., TOUT modèle est borné par l'entropie unigramme : **plancher = 3.8902 nats**.

**Résultat** :

| Steps | Train loss | Eval loss (bruit frais) | Écart au plancher |
|---|---|---|---|
| 80 | 3.73 | 3.44 | **−0.45** |
| 200 | — | **1.78** | **−2.11** |

Une loss sous le plancher information-théorique est **impossible sans fuite du label dans l'input**.
On reproduit le régime de l'ancien rapport (scratch=1.50) : le modèle apprend à **copier le futur
visible**, pas à modéliser la langue.

**Mécanisme** (vérifié dans `source/cognet_1b_optimized.py`) : CogNet est entièrement bidirectionnel,
aucun masque causal nulle part — `mean_key` sur tout T, `SDPA(is_causal=False)`, composer avec
shift futur (`bound[t+1]` visible en position t). En shifted-LM, `logits[:, :-1]` prédit `input[:, 1:]`
**en ayant vu `input[:, 1:]`**.

**Conséquence honnête** : TOUTES les losses shifted-LM du repo sont affectées (ancien rapport,
`run_cognet_moe.py` phases 2a/2b/3, et mon propre smoke test CPU). Les comparaisons *relatives*
restent informatives, mais les valeurs absolues ne mesurent pas un apprentissage autorégressif réel.
Seule la tâche sonde du volet 1 (label jamais dans l'input) est sans fuite.

---

## Volet 1 — Comparaison équitable sans fuite

**Tâche sonde** : motif fixe (256 tokens, vocab 64), préfixe 32 bruité à 10% → prédire le 33e token
(label **jamais** dans l'input). Chance = log(64) = 4.16. Contrôle « motif inédit » ≈ 4.2-4.3 partout
(= chance → la tâche mesure bien la mémorisation, pareil pour tous les bras).

**Équité** : budget **total tokens-input strictement égal** (200k/bras), **y compris les tokens
consommés par la Phase 1** (41k) que les papiers EDT « oublient ». Init trunk appariée (même seed),
même head, même LR, même stream de données.

**Résultats** (eval sonde, ↓ mieux) :

| Bras | seed 0 | seed 1 | seed 2 | seed 3 | Moyenne | Tokens (P1/2a/2b/sonde) |
|---|---|---|---|---|---|---|
| SCRATCH | 2.890 | 2.896 | 2.852 | 2.924 | **2.89** | 0/0/0/199.7k |
| **P1-ONLY** | 2.367 | 2.415 | 2.354 | 2.423 | **2.39** | 41.2k/0/0/158.7k |
| P2B-ONLY | 3.221 | 3.264 | — | — | 3.24 | 0/0/20.2k/179.7k |
| EDT-NOFIX (full) | 3.451 | 3.495 | — | — | 3.47 | 41.2k/0.3k/20.2k/138.2k |
| EDT-V2 (full) | 3.479 | 3.586 | — | — | 3.53 | 41.2k/12.8k/20.2k/125.4k |

- **Full EDT réfuté** : pire que scratch de ~0.6 nat, avec 37% du budget brûlé en pré-phases.
  Les fixes V2 ne changent rien (V2 ≈ NOFIX) : le problème n'est pas la diversité, c'est l'objectif.
- **P1-ONLY gagne de 0.50 nat, 4/4 seeds**, avec MOINS de tokens-sonde (158.7k vs 199.7k)
  et MOINS de FLOPs (les 41k tokens P1 n'entraînent que des petits experts) → bénéfice d'init réel.
- **P2B-ONLY perd** (3.24 vs 2.89) malgré 180k tokens-sonde → la phase 2b nuit activement.
- Routing sain partout (0 expert mort, usage ~0.42-0.61) : l'écart ne vient pas d'un collapse.

---

## Analyse mécaniste : pourquoi

1. **Phase 1 = bonne init résiduelle** (cf. Fixup/ReZero/SkipInit) : des experts pré-conditionnés
   à `f(x) ≈ x + δ_e` (δ_e = perturbation V2 qui préserve la diversité) perturbent moins le flux
   résiduel au démarrage que des experts aléatoires à forte variance. Le joint-training part de
   plus haut et converge mieux. La perturbation V2 est utile *à ce niveau*.
2. **Phases 2a/2b = entraînement à l'exploitation de fuite** : leur objectif shifted-LM fuit
   (volet 0). L'encoder et le router apprennent le « copy-trick » bidirectionnel au lieu de
   représentations transférables, puis la sonde (sans fuite) doit *désapprendre* — d'où le déficit.
   Elles brûlent aussi 16-37% du budget.
3. **Full EDT = P1 utile + 2a/2b toxiques + moins de tokens utiles** → perd. CQFD.
4. Le diagnostic historique (« P1 rend les experts indiscernables ») était un **faux diagnostic
   sur des données fuitantes** : avec une tâche sans fuite, P1 *aide* (+0.5 nat). La diversité
   comportementale mesurée (P1-ONLY div 0.90 vs scratch 0.95) confirme que les experts restent
   différenciés.

---

## Implications pour le repo (recommandations)

1. **Chemin par défaut** : `fast_phase1_parallel()` (P1 = smart init, ~4-6× plus rapide que P1
   séquentielle) → joint training `fast_train.py` → lifelong `run_infinite.py`.
   Autrement dit : **la stack fast+infini livrée sur cette branche est déjà le bon design**.
2. **Ne PAS utiliser le pipeline EDT complet** comme méthode principale (phases 2a/2b shifted-LM
   à proscrire sur cette architecture tant qu'elles fuitent).
3. **Chantier de fond** (hors scope ici) : CogNet a besoin d'un training LM principiel —
   soit masquage causal (cohérence causale par cumsum, composer causal, …), soit entraînement
   prefix-LM / sonde-style sans fuite. Sans ça, toutes les losses LM affichées restent suspectes.
4. Garder `validate_edt.py` pour l'historique, mais son verdict est caduc (protocole fuitant).

## Limites

- Échelle tiny CPU (64d×2b), tâche sonde ≠ LM plein : la réfutation vaut pour ce régime ;
  un bénéfice EDT à grande échelle exigerait des phases 2a/2b sans fuite + une preuve nouvelle.
- 2 seeds (5 bras) + 2 seeds (paire clé) : l'effet P1 (0.50 nat, 4/4) est très au-delà du bruit
  inter-seed (~0.05), mais un n plus grand serait idéal avec un GPU.

## Reproduction

```bash
PYTHONPATH=source:. python3 refute_edt.py --quick   # ~1 min : fuite + 1 seed
PYTHONPATH=source:. python3 refute_edt.py           # ~2 min : full (volet 0 + 5 bras × 2 seeds)
```

Données brutes : `refutation_report.json`.
