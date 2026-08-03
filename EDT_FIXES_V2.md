# EDT Fixes V2 — Perturbation Multiplicative + Phase 2a 50 steps

**Date** : 2026-08-03
**Branche** : `arena/019fc5c5-cognet-moe-1b` (suite de PR #1 mergée)
**Issue** : EDT 2.4× pire que from-scratch à budget tokens identique (cf. EDT_VALIDATION_REPORT.md)

---

## Diagnostic (rappel)

PR #1 a montré :

| Test | Verdict |
|------|---------|
| Phase1 MSE 0.984→0.018 | ✅ |
| Phase2b loss 4.93→4.70 | ✅ |
| Phase3 loss baisse 4.45→2.58 | ✅ nuancé |
| Routing stable | ⚠️ faux positif (seuil 0.5 inadapté C=4/top2) |
| **EDT vs from-scratch** | **❌ FAIL EDT=3.60 vs scratch=1.50 (×2.4 pire)** |

**Cause racine** : Phase 1 entraîne tous les experts vers la même identité `f(x)≈x`. 
Comportement identique → CoherenceRouter O(n) ne peut pas différencier les experts → 
en Phase 3 il choisit arbitrairement 2 experts sur 4 → 50% capacité MoE morte.

```
Avant Phase1 (random)  : cos_sim poids ≈ 0.002-0.003 (différents)
Après Phase1 (identité) : cos_sim poids ≈ 0.002-0.004 (pareil) MAIS comportement identique
Tous les experts : f(x) ≈ x, MSE < 0.02 → router voit sorties identiques
```

---

## Fix #1 — Perturbation Multiplicative Unique par Expert

### Problème
Sans perturbation, la target Phase 1 est identique pour tous les experts :
`h_target = h_in` → tous les experts convergent vers même fonction.

### Solution implémentée dans `edt_pipeline.py`

```python
# EDTConfig nouveaux champs
phase1_perturbation_scale = 0.02   # 2% perturbation
phase1_perturbation_mode = "both"  # multiplicative + additive
phase1_use_perturbation = True

# Génération déterministe par expert
def _get_expert_perturbation(block, expert, D, scale, mode, seed):
    gen.manual_seed(seed + block*100 + expert*17)
    mult = U(-scale, scale, shape=(D,))   # vectoriel D
    add  = U(-scale*0.5, scale*0.5, shape=(D,))
    return mult, add

# Training
h_target = h_in * (1 + mult_e) + add_e
loss = MSE(h_out, h_target)
```

**Effet** :
- Chaque expert apprend `h_out ≈ h_in * (1+δ_e) + ε_e` où δ_e, ε_e uniques par expert.
- Diversité comportementale préservée : `var(sorties experts)` mesurée et loggée.
- Router reçoit signal différenciable dès Phase 2a.
- Inspiré recommandation rapport : `target_i = identity + delta_i * 0.1`

**Type** : multiplicative (demandé) + additive (complément, 50% scale) = mode "both" par défaut.
- `multiplicative` seul : `h_target = h_in * (1 + mult)` → préserve échelle
- `additive` seul : `h_target = h_in + add`
- `both` : combine les deux pour diversité maximale

**Log** :
- Norme perturbation par expert : `mult_norm, add_norm`
- Diversité comportementale : `var_across_experts = mean((out_e - mean)^2)` sur block 0

---

## Fix #2 — Phase 2a Étendue 1→50 Steps

### Problème
EDT original : "1 step suffit pour symmetry break". 
Sur CogNet-MoE :
- CoherenceRouter O(n) très peu expressif (2×Linear D→C) vs gate MoE classique (D→N)
- Experts 50M chacun (vs 0.43M Switch Transformer) → attracteur identité plus fort
- Avec perturbation Fix #1, router a besoin de temps pour apprendre différenciation

1 step → router n'apprend rien → Phase 3 démarre avec routing arbitraire.

### Solution

```python
# EDTConfig
phase2a_steps = 50  # was 1

# Phase2a : 50 steps × 32 batch ≈ 1-2 min (vs <1s avant)
# Freeze experts (déjà diversifiés), unfreeze router+memory+composer
```

**Effet** :
- Router voit 50× plus d'exemples des experts diversifiés
- Apprend à router avant joint fine-tune
- Coût négligeable : 50 steps << 2000 steps/expert Phase 1 et 100M tokens Phase 3

**Validation** :
- Dans `validate_edt.py`, test Phase 3 passe de `phase2a_steps=1` à `50`
- Monitoring loss Phase 2a : devrait baisser sur 50 steps si router apprend

---

## Fix #3 — Seuil Collapse Dynamique + Aux Loss Weight 0.01→0.05

### Problème rapporté

| Config | max_load uniforme | Seuil 0.5 fixe | Verdict |
|--------|------------------|----------------|---------|
| C=4 top2 | 0.50 (=2/4) | 0.50 | Toujours ≥ seuil → 100% faux positifs |
| C=8 top2 | 0.25 | 0.50 | OK |

Avec test C=4/top2, taux collapse = 100% même avec routing optimal.

### Solution

```python
uniform_load = top_k / n_experts
collapse_threshold = 1.5 * uniform_load  # 50% au-dessus uniforme
# C=4/top2 → thr=0.75 (vs 0.5 avant)
# C=8/top2 → thr=0.375 (vs 0.5 avant, plus strict mais juste)

# Phase3
phase3_aux_loss_weight = 0.05  # was 0.01, force utilisation tous experts
```

**Dans validate_edt.py** : même seuil dynamique + logs comparatifs ancien/nouveau.

---

## Fix #4 — Logging Diversité Comportementale

Au lieu de logger seulement `std des normes de poids` (diversité structurelle), on loggue `var des sorties`.

```python
expert_outputs = [expert_e(h_test) for e in range(C)]
var = mean((stack - mean)^2)  # diversité comportementale réelle
```

Plus var est grande, plus router peut différencier.

---

## Fichiers modifiés

| Fichier | Changement |
|---------|------------|
| `edt_pipeline.py` | Fix #1 perturbation, #2 phase2a=50, #3 aux=0.05 + seuil dynamique, #4 diversity logging |
| `validate_edt.py` | Seuil dynamique, phase2a=50 dans tests, aux=0.05 |
| `EDT_FIXES_V2.md` | Ce document |
| `EDT_VALIDATION_REPORT.md` | Inchangé (historique), voir V2 pour fixes |

---

## Prochaines étapes validation

1. **Relancer `validate_edt.py` CPU** (~15 min) avec fixes :
   - Attendu : diversity var > 0 après Phase1 (vs ≈0 avant)
   - Attendu : Phase3 collapse_rate < 50% avec nouveau seuil
   - Attendu : EDT vs from-scratch gap réduit (voire EDT meilleur)

2. **Si EDT toujours pire** :
   - Augmenter `phase1_perturbation_scale` 0.02→0.05
   - Mode `multiplicative` seul (moins destructif que `both`?)
   - Tester C=8 (vrai modèle) pas C=4 (test réduit)
   - Augmenter `aux_loss_weight` 0.05→0.1

3. **Sur 3090** : lancer pipeline complet EDT avec fixes et comparer from-scratch 150k tokens.

---

## Commandes

```bash
# Push fixes (cette session)
git add -A
git commit -m "Fix EDT: perturbation multiplicative + phase2a 50 steps + seuil dynamique"
git push origin arena/019fc5c5-cognet-moe-1b
gh pr create --base main --title "Fix EDT: perturbation + phase2a 50 steps" --body "..."

# Valider
PYTHONPATH=source:. python3 validate_edt.py
```

---

## Références

- EDT_VALIDATION_REPORT.md § Problème 1 (CRITIQUE)
- Recommandations §4 : "Modifier Phase 1 : target_i = identity + delta_i"
- Recommandation §2 : "Phase 2a étendue : 1 step ne suffit pas. Étendre à 50-100"
