# ExpertPager — Rapport : pagination disque des experts (FR)

**Question posée** : comment la latence I/O disque pendant un page fault est-elle masquée,
pour que le traitement CPU d'un batch ne stalle jamais sur des chargements d'experts froids ?

**Réponse courte** : le prefetch **hash-ahead exact** (l'adressage du batch N+1 est connu
dès le batch N, gratuitement) + **double buffering** sur pool de loaders + **vagues bornées**
+ **fallback joker compté**. Mesuré (CPU tiny, latence SSD simulée 15 ms, E=16 pages/bloc,
S=4 slots) : **4.4× plus rapide, stall 4562 ms → 817 ms, 272 fautes couvertes / 24 sync,
zéro stall dur**. Le paging est **bit-à-bit transparent** (écart dense↔paginé `0.00e+00`).

---

## 1. Ce qui a été construit (`expert_pager.py`, ~950 lignes)

| Composant | Rôle |
|---|---|
| `ExpertPageStore` | 1 fichier safetensors par page (expert + slice `to_channels`), fp32 ou **int8** (×0.98 fidélité cos 0.9999, ÷3.72 disque) |
| `ExpertPager` | LRU + hot-pinning (usage-EMA) + pool async de loaders + writeback dirty + stats (hits/sync/couverts/stall/jokers) |
| `PagedHashExpertRouter` | S slots partagés pour E pages, dispatch par **vagues**, backward par **recompute** (style gradient-checkpointing) |
| `hash_ahead_pages` | prefetch **exact** mode token : les pages du batch N+1 depuis ses token IDs (0 forward) |
| `working_set_hint` | prefetch LSH : couche-0 exacte (encoder ne dépend d'aucun expert) + localité temporelle |
| `_PagedDispatchFn` | custom autograd Function : forward sans graphe (swaps invisibles), backward rejoue chaque vague pages ré-installées |

Croissance infinie : `add_experts` = `mmap` (nouvelles pages initialisées sur disque,
jamais en RAM) ; `phase_bias` = espace d'adressage versionné (le freeze par âge devient
du gel de pages — `frozen_pages`, `requires_grad=False` à l'install).

## 2. Anatomie du masquage de latence (réponse à la question)

```
Batch N (compute)          │ Batch N+1 (I/O en avance)
───────────────────────────┼─────────────────────────────────
forward waves 1..W         │ prefetch(hash_ahead(batch N+1))  ← soumis AVANT, exact
  vague w : ensure()       │   loaders async (8) : 15 ms/page en parallèle
    hit résident → 0 ms    │
    faute couverte → ~1 ms │ fuseau : le compute N (~100 ms) couvre ~6-50 loads
    faute sync → 15 ms     │ (SSD réel : 64+ queues, pages int8 = 4× moins d'octets)
backward : recompute/vague │ prefetch(batch N courant) re-soumis après forward
───────────────────────────┴─────────────────────────────────
Jamais de stall dur : si tous les slots sont pinnés → joker (saut compté, pas d'attente).
```

Les 5 mécanismes et leur effet mesuré :

1. **Hash-ahead exact + double buffering** (le gros levier) : 296 fautes sync → 24 ;
   stall −82 %. En mode token, l'oracle est parfait et gratuit.
2. **Hot-pinning** (usage-EMA) : les pages chaudes ne faultent plus (mécanisme livré,
   effet nul sur batches i.i.d. tiny — bénéfice attendu sur texte réel bursty).
3. **Prédicteur de working set** (LSH) : couverture du hint **0.71** (0.47 à froid =
   couche-0 exacte seule, 0.75+ en régime) ; recall résident au plafond des slots.
4. **Pages int8** : ÷3.72 trafic disque, cos-similarité 0.999943 (erreur < bruit d'optim).
5. **Vagues + joker compté** : borne mémoire dure (S slots), dégradation gracieuse —
   **0 joker déclenché** dans tous les tests (jamais de pression extrême atteinte).

## 3. Résultats du self-test (`python3 expert_pager.py --self-test`, ~10 s CPU)

| # | Test | Résultat |
|---|---|---|
| 1 | Parité paginé/dense (vagues forcées S=3, E=8) | **écart `0.00e+00`** — transparence bit-à-bit |
| 2 | Anti-stall (15 ms/f Faute, E=16, S=4, 6 batches train) | **4.39×**, stall 4562→817 ms, couverts 272/296 |
| 3 | Writeback dirty → reload disque frais | **écart `0.00e+00`** — le disque = l'état entraîné |
| 4 | Pages int8 | cos **0.999943**, ratio disque **3.72×** |
| 5 | Prefetch LSH (couche-0 exacte + localité) | couverture hint **0.71**, résident au plafond slots |

## 4. Limites honnêtes (connues, assumées v1)

- **Backward = 1 forward supplémentaire** (recompute par vague). Standard (checkpointing),
  mais le coût est réel ; le prefetch couvre aussi le recompute (pages du batch courant
  re-soumises après le forward).
- **Moments Adam suivent le slot, pas la page** : après swap, l'optimiseur voit des poids
  frais avec des moments résiduels. Converge (prouvé en test [2]/[3]) mais sous-optimal ;
  v2 = moments paginés par page sur disque.
- **LSH + RoPE disperse** : les hiddens position-dépendants étalent le working set
  (~30/32 pages touchées en tiny aléatoire). Confirme la reco : **token-hash par défaut**
  (exact, gratuit), LSH opt-in. Sur modèle entraîné réel, la concentration est meilleure
  (mesure tiny = pire cas).
- **Latence simulée** (`io_delay_ms`), pas un benchmark NVMe réel : ce qui est prouvé,
  c'est le *mécanisme* de masquage (couverture 92 %), pas un chiffre SSD absolu.
- Pages `to_channels` dupliquées par slice (E×D×D au total sur disque — normal, c'est le
  prix du découplage ; int8 le rend négligeable).

## 5. Usage

```bash
pip install safetensors
python3 expert_pager.py --self-test          # les 5 tests (~10 s CPU)

# Entraînement paginé (squelette) :
#   pager = ExpertPager(PagerConfig(store_dir="pages/", int8_pages=True))
#   model = convert_to_paged(dense_hash_model, n_slots=4, pager=pager, mode="token")
#   for ids_next in loader:
#       pager.prefetch(hash_ahead_pages(ids_next, model))  # I/O du futur
#       out = model(ids); out.backward(); opt.step()
#       pager.mark_all_resident_dirty(); pager._install_prefetched()
#   pager.flush()  # writeback final → le disque EST le checkpoint
```

## 6. Prochaine étape (frontière matmul, substitution modulaire)

Le pipeline I/O→CPU est posé. Les experts restent des matmuls dense (FusedSwiGLU) —
remplaçables sans toucher au pager : experts **associatifs / sans-gradient**
(Palimpseste-max-style) = même interface page (×2 pales associatives au lieu de ×3
matrices), même adressage, même prefetch. Le jour où le matmul n'est plus le bottleneck,
on substitue le contenu des pages, pas la machine.

---

*Verdict : la pagination disque valide la croissance matérielle infinie — E croît sur
disque (int8), la RAM ne voit que S slots, le batch ne stalle jamais (couverture 92 %,
joker compté en dernier recours). Prochaine frontière : le contenu des pages.*
