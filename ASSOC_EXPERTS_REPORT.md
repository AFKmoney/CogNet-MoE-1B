# Experts associatifs sans gradient — Rapport (FR)

**Verdict** : la substitution modulaire est validée de bout en bout — routage par hash,
pré-chargement asynchrone, retrieval associatif et mise à jour Hebbienne, **zéro
backprop dans les experts, zéro moment Adam à paginer** (le piège architectural est
évité : le problème des états d'optimiseur s'évapore au lieu d'être durci). Sur la
sonde leak-free (200k tokens, 3 seeds) : **ASSOC 3.56** — bat la tête ELM figée (3.98)
de 0.42 nats avec **7× moins de paramètres entraînables** (39k vs 269k). Écart assumé
vs experts gradient (+1.68), expliqué et borné par ablations (§4).

## 1. Ce qui a été construit

**`assoc_experts.py`** (~550 lignes) — contenu de page substitué, machine inchangée :
- Page disque = `{keys (M,D), values (M,D), counts (M,), proj (D,D)}`. Init directe
  sur disque (jamais E pages en RAM). Codec polymorphe : le pager ne connaît que
  `snapshot/install/read`, le router possède son contenu (refactor `expert_pager.py`,
  régression 5/5 verte).
- `assoc_retrieve` : `z = q@projᵀ`, fenêtre circulaire de W slots depuis le hash,
  `out = softmax(z·keys)@values`. Coût O(W·D) ≈ 3× moins que le matmul dense.
- `hebbian_update` : WTA-EMA vectorisée, in-place, `torch.no_grad` : les gagnants
  absorbent leurs requêtes (clés) et leurs cibles (valeurs). **Binding HDC requête⊕label**
  sur les valeurs : supervision locale, toujours 0 gradient.
- `PagedAssocExpertRouter` : S slots pour E pages, vagues + `ensure()` du pager,
  experts **détachés du graphe** (forward seul) — le gradient global (tête, encodeur,
  mémoire, composer, normes) circule par les résiduels : apprentissage local pur.
- `frozen_pages` devient la protection palimpseste : les vieilles pages sautent
  le Hebb (souvenirs protégés, gel par âge = gel de page).

**Alignement d'adressage O(1) (défi posé — résolu)** : un SEUL hash par token sert les
deux niveaux — `pages = token_hash_experts(...)` (source unique de vérité, donc
alignement **exact par construction**, prouvé en test [1]) + fenêtre slot par 1
multiply (0 passe mémoire suppl.). Page O(1) + fenêtre O(1), zéro double overhead.
Diagnostic : fenêtre pleine ≡ fenêtre hash (3.560 vs 3.554) → **la contrainte O(1)
ne coûte rien**.

## 2. Résultats sonde (200k tokens, inits appariées, 3 seeds)

| Bras | s0 | s1 | s2 | Moy | Trainables |
|---|---|---|---|---|---|
| ASSOC (Hebb + binding, paginé S=2) | 3.554 | 3.550 | 3.564 | **3.556** | 39 104 |
| HASH-TOKEN (dense, gradient) | 1.890 | 1.907 | 1.820 | **1.872** | 268 992 |
| Réf. commitées | HASH-LSH 2.08 · SCRATCH 2.89 · ELM figé 3.98 · chance 4.16 |

- ASSOC apprend : 4.18 (chance) → 3.56, bat ELM figé de **0.42 nats** 3/3 seeds.
- Le pager en boucle associative : hits 3120, sync 644, **couverts 3120 (83 %)**, 0 joker.
- Bonus : HASH-TOKEN (1.87) bat HASH-LSH (2.08) — 2e confirmation que RoPE disperse LSH.

Ablations seed-0 : M8→M16 ≈ +0 (pas un manque de capacité) · binding −0.05~−0.08
(cohérent, adopté) · **FA-erreur sur clés : 3.55 → 3.59 (négatif)** — erreur broadcastée
identique pour tous les tokens d'une séquence = signal trop fruste ; documenté, pas tuné.

## 3. Self-test (`python3 assoc_experts.py --self-test`, ~5 s CPU)

| # | Test | Résultat |
|---|---|---|
| 1 | Alignement pages assoc == `token_hash_experts` | **exact**, déterministe, dédup K |
| 2 | Hebb sur 3 clusters (M=6, 0 grad) | erreur angulaire **0.812 → 0.057** |
| 3 | Boucle paginée (E=4, S=2) | déterminisme ✓, writeback reload **0.00e+00**, 0 param expert |
| 4 | Pages int8 | cos **0.999974** |
| 5 | Anti-stall (15 ms/faute) | **2.90×**, couverts 88/112 |

## 4. Pourquoi +1.68 ? (analyse honnête, pas une excuse)

Les experts Hebbiens v1 mémorisent des **prototypes non supervisés** (lissage adaptatif
du flux résiduel) ; seul le binding donne aux valeurs un écho du label. Les experts
gradient, eux, sculptent des **features discriminatives** par backprop. Le binding aide
(−0.06) mais ne transporte pas de contraste d'erreur ; la FA-broadcastée échoue
(signal identique par séquence). Conclusion : v1 = mémoire adaptative (bat le figé),
pas encore machine discriminative. Le gap est un **manque de signal d'erreur localisé**,
pas de capacité (M16 ≈ M8) ni d'adressage (fenêtre pleine ≡ hash).

## 5. Correctifs infra au passage (dans `expert_pager.py`)

- **Codec polymorphe** (`snapshot/install/read` sur le router) : la substitution
  associative n'a touché AUCUNE ligne du chemin chaud du pager. Preuve de modularité.
- **`ensure()` résidentes-d'abord** : corrige la pathologie du balayage (hits 0 → 3120,
  sync ÷6). Bénéficie aussi au dense (4.37× maintenu).
- **int8 : vecteurs exacts** (seules les matrices passent en int8) : compteurs Hebbiens
  et biais préservés ; ratio dense inchangé (~3.7×).
- **`hash_ahead_pages` polymorphe** (`supports_id_prefetch`) : prefetch exact partagé
  dense/associatif (même sel → mêmes pages).

## 6. Limites + v2

- v1 auto-associative + binding : adressée au signal, pas à l'erreur contrastive.
  v2 candidate : erreur **par token** (tête locale ou décodage positionnel) + FA, ou
  iteration Hopfield (retrieval multi-pas attracteur) — coût nul en paramètres.
- Hypervecteurs binaires (le vrai HDC, ÷32 mémoire, Hamming au lieu de dot) : non
  testé, pages 2× plus grosses que dense-fp32 à (M=8,D=64) vs FusedSwiGLU — int8
  compense (÷4), binaire exploserait le ratio.
- Tête + tronc non-expert encore en Adam : le 100 % sans-gradient reste un horizon
  (tête lstsq online = piste, cf. Exp B hash).

## 8. v2 — Campagne erreur-par-token (résultat : plafond mappé, loi établie)

Question : signal discriminant par token SANS graphe global. 20+ configs seed-0
+ run 3-seed. Réponse honnête : **+0.00** — le plafond du lookup-prototype est
3.55±0.02, et la campagne révèle la loi de l'architecture.

| Voie | Mécanisme (tous 0 gradient) | seed-0 | Verdict |
|---|---|---|---|
| FA broadcast (v1) | erreur séquence → clés | 3.59 | ✗ dégrade (aucun contraste) |
| FA centrée β=1 / β=2 | sonde fixe + résidu token | 3.59 / 3.55 | ✗ neutre/négatif — **retirée du code** |
| ART-label (token-hash) ρ=.3/.5 | vigilance catégorielle | **bit-identique** | ✗ dégénère (preuve ci-dessous) |
| LSH + ART | routage corrélé + vigilance | 3.63 / identique | ✗ LSH pire + ART dégénère encore |
| Binding α=1.0 / 2.0 | readout supervisé fort | 3.61 / 3.56 | ~ sweet spot α=0.5 |
| Nouveauté γ=2 / γ=4 | match-tracking (plasticité) | 3.53 / 3.60 | ~ γ=2 aide, γ=4 oublie |
| **v2 fige (α.5+γ2, 3 seeds)** | | **3.549 vs 3.556** | ~ −0.008 : plat |

**Preuve de dégénérescence ART** : le repli retombe sur l'argmax ; sous routage
indépendant des labels, les histogrammes sont uniformes → soit commit (argmax),
soit repli (argmax) → gagnants **bit-identiques** à vigilance 0 (vérifié au chiffre).
Sous LSH : scatter RoPE + labels many-to-one sur les contextes (64 valeurs pour
224 positions — même label ⟺ contextes DIFFÉRENTS) → même uniformité. ART exige
une géométrie label-corrélée que ni le hash ni cette tâche ne fournissent
(synthèse corrélée : pureté 0.97 — le mécanisme est sain, son domaine absent).

**Loi empirique** : *l'adressage n'accepte que du non-supervisé* (Hebb ✓, FA-clés ✗,
ART ✗) ; *seul le readout accepte le signal tâche* (binding ✓). Le gap +1.68 n'est
ni capacité (M16≈M8), ni adressage (W8≡W4), ni signal (20 configs plates) : c'est
**l'expressivité du readout** — lookup de prototypes vs fonction apprise.

**v3 proposée** : tête à pooling ATTENTIF (1 couche → crédit par token gratuit,
pas de graphe global) + slots **localement-linéaires** (mini-readout par slot
appris en delta-rule locale sur ce crédit). Ou axe matériel : hypervecteurs
binaires (÷32, Hamming). L'infra v2 reste intacte : O(1), prefetch 83 %, writeback
bit-à-bit, 0 joker — la campagne n'a rien cassé (régressions vertes).

## 7. Usage

```bash
PYTHONPATH=source:. python3 assoc_experts.py --self-test      # 5 tests, ~5 s
PYTHONPATH=source:. python3 assoc_experiment.py --ablate      # matrice seed-0
PYTHONPATH=source:. python3 assoc_experiment.py               # 3 seeds, ~70 s
```

---

*Verdict : le moteur de pensée continu tourne — hash → prefetch → retrieval → Hebb →
writeback — sans aucun gradient dans les experts et sans état d'optimiseur. Les experts
apprennent (3.56 < 3.98), la machine est modulaire (codec substitué, pager intact),
l'adressage est O(1) sans surcoût. Reste à donner aux mémoires un signal d'erreur
localisé : c'est la frontière v2, pas un défaut d'infrastructure.*
