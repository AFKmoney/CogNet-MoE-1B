# CogNet-MoE-1B — MoE + EDT + Chinchilla + Tokenizer BPE propriétaire

> Modification **CogNet-native** de [CogNet-1B](https://github.com/AFKmoney/CogNet-1B) :
> - On respecte **strictement** la structure **100% non-transformer** de CogNet
> - **Aucune attention inter-token**, aucune opération O(n²), aucun KV-cache
> - Les **8 canaux du CognitiveRouter = les 8 experts MoE** (unification, pas de gate séparé)
> - Pipeline **EDT** (Expert Decoupled Training) adapté aux canaux cognitifs
> - **Chinchilla** avec interprétation active-param
> - **Tokenizer BPE propriétaire** (vocab 16k, FR+EN+code)

---

## ⚠ Avertissement architectural

CogNet-1B est **100% non-transformer**. Tout ce qui suit l'est aussi.

| Pattern transformer | Statut dans CogNet-MoE-1B |
|---|---|
| Self-attention, multi-head attention | ❌ **INAPPLICABLE** — remplacé par **cognitive routing O(n)** |
| FlashAttention / FlashAttention-2 | ❌ **INAPPLICABLE** — pas d'attention inter-token |
| GQA / MQA | ❌ **INAPPLICABLE** — pas de multi-head |
| KV-cache | ❌ **INAPPLICABLE** — mémoire hiérarchique 3-tier avec **slots SDPA fixes** (128+256+512) |
| Sliding-window attention | ❌ **INAPPLICABLE** — pas d'attention du tout |
| Mixtral-style MoE gate (Linear D→N) | ❌ **INAPPLICABLE** — le routing est fait par le **CoherenceRouter O(n)**, pas par un gate séparé |
| O(n²) complexity | ❌ **INAPPLICABLE** — strictement O(n) par layer |

Le seul SDPA présent est sur les **slots mémoire fixes** (128+256+512 = 896 slots), ce qui est **O(1) par token en seq_len**.

---

## TL;DR

| Métrique                      | CharTokenizer (legacy) | **BPE 16k (actuel)**    |
|-------------------------------|-----------------------:|------------------------:|
| Vocab size                    | 136                    | **16,384**              |
| token_emb params              | 278,528                | **33,554,432** (+33,3M) |
| Total params (MoE)            | 7,09B                  | **7,21B**               |
| Actifs / token                | 2,26B                  | **2,38B**               |
| Capacity multiplier           | 3,14×                  | **3,03×**               |
| Chinchilla optimal (tokens)   | 45,2B char-tokens      | **47,66B BPE-tokens**   |
| Chinchilla (équivalent chars) | 45,2B chars            | **143B chars** (~3×)    |
| Scénario C EDT (tokens)       | 1,29B char-tokens      | **1,36B BPE-tokens**    |
| Scénario C (équivalent chars) | 1,29B chars            | **4,08B chars** (~3×)   |
| Temps EDT (3090, Scénario C, PGSU) | ~10 jours        | **~10-12 jours**        |
| Coût EDT (3090 spot)          | ~$95                   | **~$110**               |

**Bénéfice clé du BPE 16k** : pour le même budget compute (~1,36B tokens), on consomme **~3× plus de texte réel** (4,08B chars vs 1,29B chars), donc le modèle voit beaucoup plus de données linguistiques. Phase 2b (embedding-only) devient également plus informative.

---

## Architecture CogNet-native (CORRIGÉ)

Tous les composants **non-transformer** de CogNet sont préservés. Le MoE est **unifié** au CognitiveRouter : **les 8 canaux = les 8 experts**.

```
TokenEncoder (RoPE + RMSNorm, séparable pour EDT Phase 2b)
  → 16× CogNetMoEBlock
      ├─ CognitiveExpertRouter              ← NOUVEAU (unifie Router + MoE)
      │     ├─ CoherenceRouter O(n)         (query × mean_key, softmax sur 8 canaux)
      │     │     → produit routing_weights (B,T,8) — PAS de gate séparé
      │     ├─ to_channels                  Linear(D → 8×D)
      │     ├─ Top-2 sparse sur routing_weights (noisy top-k, Shazeer 2017)
      │     ├─ 8 experts = 8 canaux FusedSwiGLU(D=2048, ff=8192)
      │     │     → seuls les 2 experts sélectionnés calculent par token
      │     ├─ aux_loss : N · Σ f_i · P_i   (Switch Transformer load-balancing)
      │     ├─ z_loss   : mean(router_logits²)  (ST-MoE)
      │     └─ clamp(max=10)                (anti-explosion, EDT)
      ├─ ParallelHierarchicalMemory         (3-tier Working/Episodic/Semantic, SDPA reads)
      │     ├─ Working  : 128 slots (court terme)
      │     ├─ Episodic : 256 slots (moyen terme)
      │     └─ Semantic : 512 slots (long terme)
      │     → O(1) par token en seq_len (slots fixes, SDPA bornée)
      └─ CompositionalReasoner              (hyperdim role-filler binding)
  → final RMSNorm
  → output_proj (weight-tied avec token_emb)
```

**Complexité : STRICTEMENT O(n) par layer.** Aucune attention inter-token. Aucun gate transformer-style. Le routing cognitif et le MoE sont unifiés — un seul mécanisme décide l'activation cognitive ET l'expert.

### Pourquoi cette conception est CogNet-native

1. **Le routing est TOUJOURS la coherence O(n)** du CognitiveRouter original. On ne réintroduit aucun gate transformer-style.
2. **Les experts SONT les canaux** — un seul mécanisme de routing décide l'activation cognitive ET l'expert. Pas de doublon.
3. **La structure résiduelle est préservée** (to_channels comme dans l'original, mais chacun des 8 canaux est un FusedSwiGLU au lieu d'un ChannelProcessor conv+SwiGLU).
4. **La complexité reste O(n) par layer** — le top-2 sparse ne change rien à la complexité (juste le coût constant).
5. **Aucune opération O(n²)** n'est introduite. Le seul SDPA est sur les slots mémoire fixes.

---

## Fichiers livrés

| Fichier | Rôle |
|---|---|
| `cognet_tokenizer.py` | **Tokenizer BPE propriétaire** (vocab 16k, FR+EN+code, ByteLevel) |
| `cognet_tokenizer.json` | Tokenizer entraîné sauvegardé (HuggingFace format) |
| `cognet_moe.py` | `CognitiveExpertRouter` (CogNet-native MoE) + `CogNetMoE1B` (modèle complet) |
| `edt_pipeline.py` | Pipeline EDT 4 phases + `PGSU` + vérification prérequis + aux-loss clamping |
| `chinchilla_scaling.py` | Décomposition params + Chinchilla + comparaison CharTokenizer vs BPE 16k |
| `run_cognet_moe.py` | Script de lancement orchestrant le tout |
| `fast_train.py` | Stack d'entraînement rapide (dataset memmap, router fast, FastTrainer) |
| `phase_routed_moe.py` | Phase-Routed MoE (experts à la volée, entraînement infini) |
| `run_infinite.py` | Orchestrateur d'entraînement lifelong multi-phases |
| `FAST_TRAINING.md` | Guide entraînement rapide + infini |
| `hash_moe.py` + `hash_experiment.py` | Routage par hash (0 params, bat le learned de ~0.8 nats) |
| `HASH_ROUTING_REPORT.md` | Verdict routage hash |
| `expert_pager.py` | Pagination disque des experts (LRU + prefetch async, bit-exact) |
| `EXPERT_PAGER_REPORT.md` | Verdict pager + anatomie du masquage de latence |
| `assoc_experts.py` + `assoc_experiment.py` | Experts associatifs sans gradient (Hebbiens, paginés) |
| `ASSOC_EXPERTS_REPORT.md` | Verdict substitution associative |
| `training_time_estimate.json` | Estimation temps d'entraînement (RTX 3090/4090, A100, H100, H200) |
| `source/` | Code original CogNet-1B (cloné depuis GitHub) |
| `CogNet-MoE-1B_Whitepaper.pdf` | Whitepaper technique (français, ~30 pages) |

---

## Tokenizer BPE propriétaire

**Pourquoi remplacer le CharTokenizer (vocab=136) ?**

Reco #4 du reviewer :
> « CharTokenizer (vocab=136) handicape Phase 2b. Une variante BPE 8k–16k rendrait l'embedding plus utile et Phase 2b plus informative. »

**Design choices :**
- Vocab size : **16,384** (upper bound de la reco 8k-16k)
- Algorithme : BPE **byte-level** (style GPT-2/LLaMA, gère tout Unicode, pas d'UNK)
- Special tokens : `<pad>` (0), `<bos>` (1), `<eos>` (2), `<unk>` (3) + 4 réservés (`<think>`, `<code>`, `<fr>`, `<en>`)
- Pre-tokenizer : ByteLevel (regex GPT-2)
- Post-processing : ajout automatique `<bos> $A <eos>`
- Padding : dynamique par batch

**Entraînement :**
```python
from cognet_tokenizer import train_cognet_tokenizer, CognetTokenizer

# Entraîner sur un corpus réel (ici corpus synthétique démo FR+EN+code).
tokenizer = train_cognet_tokenizer(
    corpus="/path/to/corpus.txt",  # ou None pour corpus démo
    vocab_size=16384,
    save_path="cognet_tokenizer.json",
    byte_level=True,
)

# Utiliser.
tok = CognetTokenizer(tokenizer_path="cognet_tokenizer.json")
ids = tok.encode("Bonjour le monde")  # → [1, 432, 89, ...]
text = tok.decode(ids)                # → "<bos> Bonjour le monde<eos>"
```

**Ratio chars/BPE mesuré** : ~2,6-3,0 sur texte FR+EN+code (vs 1,0 pour CharTokenizer). Le BPE compresse ~3× le texte.

---

## Pipeline EDT (4 phases) — adapté à CogNet-native MoE

**Vérification automatique des prérequis EDT au démarrage** :

```
======================================================================
EDT — Vérification des prérequis
======================================================================
  [✓] top_k < n_experts : 2 < 8
  [✓] Résiduels partout : FusedSwiGLU (experts), CognitiveExpertRouter,
      ParallelHierarchicalMemory, CompositionalReasoner
  [✓] Embedding séparable : TokenEncoder isolé
  [✓] Aux losses : aux=4.37, z=0.15
  [✓] PGSU : vérifié au runtime
  [✓] CogNet-native : aucun gate transformer-style, routing = coherence O(n)

  ✅ TOUS LES PRÉREQUIS EDT SONT SATISFAITS — pipeline peut démarrer.
```

Si un prérequis manque (ou si un gate transformer-style est détecté), le pipeline lève une `RuntimeError` avant de démarrer.

**4 phases** :

| Phase | Ce qu'elle entraîne | Tokens / steps | Temps estimé (3090) |
|---|---|---|---|
| **1** | 128 experts indépendants (16 blocs × 8 canaux) — MSE identity | 2000 steps/expert | ~1,2 h |
| **2a** | CoherenceRouter + to_channels + Memory + Composer (1 step symmetry break) | 1 step × 16 blocs | < 6 min |
| **2b** | TokenEncoder (séparable, sans blocs MoE) | 125M BPE tokens | ~20 min |
| **3** | Joint fine-tune avec PGSU (n_active=4) + aux-loss clamping | 1,36B BPE tokens (Scénario C) | ~175 h |
| **Total** | | | **~10 jours (RTX 3090, MFU 18%)** |

### Adaptations EDT vs document original

| Adaptation | Raison |
|---|---|
| Phase 1 : 128 experts (16×8) au lieu de 2048 | Experts plus gros (50,33M vs 0,43M), moins nombreux |
| Phase 2a : étendue à CoherenceRouter + Memory + Composer | CogNet n'a pas d'attention classique ; on pré-entraîne tous les modules cognitifs |
| Phase 2a : le MoE gate est le CoherenceRouter lui-même | **CogNet-native** — pas de gate séparé, le routing cognitif O(n) fait tout |
| Phase 2b : compatible BPE 16k | Reco #4 satisfaite — embedding plus riche qu'avec CharTokenizer |
| Phase 3 : aux-loss clamping (max=10) ajouté | Anti-explosion en début d'entraînement quand le router n'est pas équilibré |
| Phase 3 : monitoring routing collapse auto | Alerte si `max_load > 0.5` → reco monter `aux_loss_weight` à 0.05 |

### Aux-loss clamping

Le document EDT original mentionne le clamping des aux losses mais le snippet de code ne l'implémentait pas. On l'ajoute explicitement :

```python
# Dans phase3_joint() :
aux_loss = result["moe_aux_loss"]
z_loss = result["moe_z_loss"]

# ─── Aux-loss clamping (mentionné dans le doc EDT original) ───
aux_loss = aux_loss.clamp(max=cfg.phase3_aux_loss_clamp)  # default 10.0
z_loss = z_loss.clamp(max=cfg.phase3_z_loss_clamp)        # default 10.0

total_loss = lm_loss + 0.01 * aux_loss + 1e-3 * z_loss
```

---

## Chinchilla scaling re-mesuré

**Deux scénarios comparés** :

```
══════════════════════════════════════════════════════════════════════
COMPARAISON DES DEUX TOKENIZERS
══════════════════════════════════════════════════════════════════════
Métrique                                  CharTokenizer        BPE 16k
──────────────────────────────────────────────────────────────────────
Vocab size                                          136         16,384
token_emb params                                278,528     33,554,432
Total params (CogNet-native MoE)           7,091,982,336  7,214,895,104
Active params / token                      2,259,947,520  2,382,860,288
Capacity multiplier                               3.14×          3.03×
Chinchilla optimal (tokens)              45,198,950,400 47,657,205,760
Scénario C EDT (tokens)                   1,291,398,582 1,361,634,450
Scénario C (équivalent chars)             1,291,398,582 4,084,903,350
──────────────────────────────────────────────────────────────────────
```

**Interprétation** :
- Le passage au BPE 16k n'augmente les params que de ~0,5% (token_emb plus grand)
- Mais pour le même budget compute, on consomme **~3× plus de texte réel**
- Le Chinchilla optimal en tokens augmente légèrement (45,2B → 47,7B) car les params actifs augmentent aussi

---

## Self-test CogNet-native

Le self-test dans `cognet_moe.py` vérifie explicitement :

1. ✅ Forward + backward pass fonctionnels
2. ✅ Tous les 16 experts (8 canaux × 2 blocs en test) reçoivent du gradient
3. ✅ Aux loss et z loss finis (pas de NaN/Inf)
4. ✅ **Aucun gate transformer-style détecté** (audit nom des modules)
5. ✅ Routing = coherence O(n) uniquement (vérification architecturale)
6. ✅ Memory SDPA sur slots fixes (128+256+512 = 896) → O(1)/token en seq_len

```bash
python3 cognet_moe.py
# → "All self-tests passed! CogNet-native MoE is correct."
```

---

## Bug corrigé dans le calcul des aux losses

Le snippet MoE original avait un bug cassant le calcul des aux losses :

```python
# ❌ Buggy — one_hot, n_tokens, K non définis
with torch.no_grad():
    f = one_hot.sum(0) / (n_tokens * K)
```

Version corrigée dans `cognet_moe.py` :

```python
# ✅ Construit explicitement one_hot à partir de topk_indices
B, T, D = x.shape
n_tokens = B * T                    # ← défini explicitement
K = self.top_k                      # ← défini explicitement
C = self.num_channels               # = n_experts

topk_weights, topk_indices = torch.topk(routing_logits_noisy, K, dim=-1)
topk_weights = F.softmax(topk_weights, dim=-1)

one_hot = F.one_hot(topk_indices, num_classes=C).float()  # (n_tokens, K, C)
expert_mask = one_hot.sum(dim=1)                          # (n_tokens, C)
f = expert_mask.mean(dim=0)                               # (C,)
P = routing_weights.reshape(n_tokens, C).mean(dim=0)      # (C,)

aux_loss = C * (f * P).sum()                              # Switch formulation
z_loss = router_logits.square().mean()                    # ST-MoE
```

---

## Estimation temps d'entraînement (vrai 1B)

Le script `scripts/training_time_estimate.py` répond à la question « combien de temps pour entraîner mon vrai 1B ? ».

| GPU | MFU CogNet-native | Phase 3 seul | Total EDT |
|---|---:|---:|---:|
| RTX 3090 (24GB) | 18% | 7,3 jours | 7,7 jours |
| RTX 4090 (24GB) | 22% | 2,4 jours | 2,6 jours |
| A100 80GB | 42% | 22,4 h | 23,2 h |
| H100 SXM | 48% | 6,0 h | 6,4 h |
| H200 | 52% | 5,1 h | 5,5 h |

**Optimisations CogNet-native applicables** (à la place de flash-attn qui est INAPPLICABLE) :

1. Fused channel matmul (cohérence + to_channels en un seul Linear)
2. Memory tier batching (déjà fait dans le source original)
3. Working memory SRAM caching (H100/A100 only)
4. Triton kernel pour CompositionalReasoner
5. Sparse dispatch optimisé pour le MoE
6. Gradient checkpointing (déjà fait)
7. PGSU (n_active=4/16)
8. bf16 + bitsandbytes 8-bit Adam
9. torch.compile (mode=reduce-overhead)
10. Batch packing

---

## Quickstart

```bash
# 1. Installer dépendances
pip install torch tokenizers bitsandbytes

# 2. Self-test complet (vérifie tokenizer + MoE + EDT + Chinchilla)
python3 run_cognet_moe.py --self-test

# 3. Rapport Chinchilla complet (comparaison CharTokenizer vs BPE 16k)
python3 chinchilla_scaling.py

# 4. Entraîner le tokenizer BPE sur un vrai corpus
python3 -c "from cognet_tokenizer import train_cognet_tokenizer; \
  train_cognet_tokenizer(corpus='/path/to/corpus.txt', vocab_size=16384)"

# 5. Lancer l'entraînement complet (nécessite GPU 3090 + dataset texte)
python3 run_cognet_moe.py \
    --dataset-path /path/to/text_corpus.txt \
    --max-seq-len 512 \
    --n-experts 8 \
    --top-k 2 \
    --phase3-tokens 1361634450 \
    --aux-loss-weight 0.01 \
    --pgsu-n-active 4 \
    --use-bf16 \
    --use-8bit-optimizer
```

---

## Recommandations cash (issues de la review)

1. **Scénario C comme baseline** : 1,36B BPE-tokens. Si qualité insuffisante, monter Phase 3 à 500M–1B.
2. **Routing collapse** : surveiller `moe_max_load` / `moe_min_load` dès les premiers runs. Si `max_load > 0.5`, passer `aux_loss_weight` à 0.05.
3. **PGSU** : `n_active=4` est bon. Si VRAM Phase 3 saute, descendre à 2.
4. **Tokenizer BPE 16k** : reco #4 satisfaite. Pour un vocab complet 16k, fournir un corpus réel d'au moins 100k documents variés.
5. **À compute égal**, CogNet-MoE-1B **doit** battre la version dense. Si ce n'est pas le cas, le problème est dans le routing ou le load-balancing.
6. **Ne JAMAIS** réintroduire un gate transformer-style. Le routing est le CoherenceRouter O(n), un point c'est tout.

---

## Verdict reviewer

> La modification est propre, le diagnostic des paramètres réels est exact,
> et le plan d'entraînement (CogNet-native MoE + EDT + Chinchilla active) est
> le plus intelligent possible avec une seule 3090.
>
> L'architecture est strictement non-transformer : aucune attention inter-token,
> aucun gate MoE séparé, mémoire hiérarchique 3-tier avec slots SDPA fixes.
>
> Le seul vrai risque restant est empirique : la réduction 35× de tokens via
> EDT doit être validée sur CogNet, et le code aux-loss a été nettoyé. ✓ Fait.
>
> Reco #4 (BPE 8k-16k) : ✓ Fait — tokenizer BPE 16k propriétaire ajouté.
>
> Correction architecturale CogNet-native : ✓ Fait — les 8 canaux du
> CognitiveRouter = les 8 experts MoE, plus de gate séparé.

---

## Training rapide + infini

- **Stack rapide** (`fast_train.py`) : dataset memmap pré-tokenisé, router fast
  (identique numériquement), `torch.compile`, optimizers fusés/8-bit, PGSU,
  curriculum seq-len, checkpoints resumables, DDP-ready.
  Projection : 1,36B tokens en ~2,5-4 jours sur RTX 3090 (vs ~10 jours), ~2-3h sur H100.
- **Phase-Routed MoE** (`phase_routed_moe.py` + `run_infinite.py`) : routing CogNet-native
  conditionné par phase, création d'experts à la volée (clone-busiest + bruit),
  anciens experts gelés (pas de catastrophic forgetting), checkpoints `.pt` infinis.
  Chaque nouveau milliard de tokens coûte ~15-25% d'un full-train.

Voir **[FAST_TRAINING.md](FAST_TRAINING.md)** pour le guide complet.

```bash
python3 fast_train.py --build-bin --txt corpus.txt --tokenizer cognet_tokenizer.json --out data/p0
python3 run_infinite.py --bins data/p0.bin --tokens-per-phase 1000000000 --compile reduce-overhead
```

## License

Code original CogNet-1B : voir `source/` (license originale préservée).
Modifications MoE + EDT + Chinchilla + BPE : MIT.
