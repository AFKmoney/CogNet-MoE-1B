# 🌙 Plan « Aurore » — entraînement de ce soir (Phase 0, infini)

> **Aurore** = le premier run 1B réel : ce soir Phase 0, puis le `.pt` grandit
> à l'infini (chaque phase ≈ 15-25% d'un full-train). Nom du checkpoint :
> `aurore-p0-final.pt`.

## 1. Les chiffres (calculés depuis `chinchilla_report.json` + physique)

Modèle : 7.21B params MoE, **2.38B actifs/token** → coût ≈ 6 × 2.38B ≈ **14.3 GFLOP/token**.

| Cible | Tokens | Statut |
|---|---|---|
| Chinchilla optimal (20 tok / param actif — la bonne lecture MoE) | **47.66B** | optimum théorique, pas un objectif à GPU unique (~172 j sur 3090, ~19 j sur H100) |
| EDT Scénario C (**la vraie cible**) | **1.36B** | ~35× moins, recommandée par le repo |

**Temps pour la cible 1.36B** (chemin infini, tok/s = pic_bf16 × MFU / 14.3 GFLOP) :

| GPU | tok/s (réaliste) | Cible 1.36B | **Ce soir (8h)** | `--tokens-per-phase` conseillé |
|---|---|---|---|---|
| RTX 3090 | 2.5-5.5k | ~3-6 j | **~70-160M** | 150 000 000 |
| RTX 4090 | 6-12k | ~1-2 j | **~170-350M** | 300 000 000 |
| A100 40GB | 8-11k | ~1.5-2 j | **~220-320M** | 300 000 000 |
| H100 SXM | 24-35k | ~0.5-1 j | **~0.7-1.0B** | 1 000 000 000 |
| CPU seul | ~10-30 (1B — inutile) | — | tiny uniquement (`train.py --demo`) | — |

> Fourchettes honnêtes (MFU réel inconnu avant mesure). Le target conseillé
> est le **haut** de la fourchette : si le run finit avant l'aube, tant mieux ;
> sinon le checkpoint resumable reprend demain. Mesurez ensuite :
> `python3 train.py --benchmark`.

## 2. Ce soir : commandes copier-coller (RTX 3090)

```bash
cd cognet_pack
pip install -r requirements.txt && pip install bitsandbytes   # 8-bit Adam (VRAM)
export TOKENIZERS_PARALLELISM=true

# 1. Corpus → .bin (prendre ≥ target : 150M tokens ≈ 600 Mo de texte BPE-16k)
python3 train.py --build-bin --txt /chemin/corpus.txt \
    --out data/aurore_p0 --tokenizer cognet_tokenizer.json

# 2. Phase 0, ~8h, dans tmux (batch 8 + accum 8 = recette 3090 serrée)
tmux new -s aurore
python3 run_infinite.py --bins data/aurore_p0.bin --tokens-per-phase 150000000 \
    --batch-size 8 --grad-accum 8 --optimizer adamw8bit \
    --compile reduce-overhead --ckpt-dir ./aurore_p0
# Ctrl+B puis D pour détacher. Au matin : tmux attach -t aurore
```

**Variantes** : 4090/A100 → `--batch-size 16 --grad-accum 4`, target 300M.
H100 → batch 16, target 1B (= quasi toute la cible 1.36B en une nuit).
OOM ? → `--batch-size 4`, puis `--seq-len 256`.

## 3. Au matin

```bash
cp aurore_p0/final.pt aurore-p0-final.pt   # ← le modèle Aurore P0, à garder
# tok/s réel = tokens_seen / temps → calibre les prochaines nuits.
# Continuer (nouveau corpus → phase 1, +2 experts/bloc, anciens gelés) :
python3 train.py --build-bin --txt /chemin/corpus2.txt --out data/p1 \
    --tokenizer cognet_tokenizer.json
python3 run_infinite.py --bins data/aurore_p0.bin,data/p1.bin \
    --resume aurore-p0-final.pt --start-phase 1 --tokens-per-phase 1000000000 \
    --batch-size 8 --grad-accum 8 --optimizer adamw8bit --compile reduce-overhead
```

## 4. Checklist avant de lancer

- [ ] `nvidia-smi` : GPU visible, ≥ 20 Go libres (3090 juste, 4090+ confortable)
- [ ] `pip install bitsandbytes` OK (sinon AdamW standard = +VRAM, risque OOM sur 3090)
- [ ] Corpus suffisant (≥ target, §1) + disque : `.bin` ≈ 2 o/token, ckpt ≈ 15-30 Go
- [ ] `tmux` / `nohup` (le run survit à la fermeture du terminal)
- [ ] `python3 train.py --self-test` vert (prouve que le pack marche sur TA machine)
