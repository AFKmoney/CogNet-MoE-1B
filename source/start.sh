#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# CogNet-1B — SSH Launcher
# ═══════════════════════════════════════════════════════════════
# Usage:
#   curl -sL https://huggingface.co/thefinalboss/CogNet-1B/resolve/main/start.sh | HF_TOKEN=hf_xxx bash
#
# Or:
#   git clone https://huggingface.co/thefinalboss/CogNet-1B cognet-1b
#   cd cognet-1b
#   chmod +x start.sh
#   HF_TOKEN=hf_xxx ./start.sh
# ═══════════════════════════════════════════════════════════════
set -e

TOKEN="${HF_TOKEN:-}"
MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SEQ_LEN="${SEQ_LEN:-512}"

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║       CogNet-1B — SSH Launcher                 ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# ── 1. Clone si pas déjà dans le repo ──
if [ ! -f "train_ultra.py" ]; then
    echo "[1/5] Cloning CogNet-1B..."
    git clone https://huggingface.co/thefinalboss/CogNet-1B cognet-1b 2>/dev/null || true
    cd cognet-1b
    echo "  Cloné dans $(pwd)"
else
    echo "[1/5] Déjà dans le repo: $(pwd)"
fi

# ── 2. Dépendances ──
echo ""
echo "[2/5] Installation des dépendances..."
pip install -q torch datasets huggingface_hub tokenizers 2>/dev/null || pip install torch datasets huggingface_hub tokenizers
echo "  OK"

# ── 3. GPU check ──
echo ""
echo "[3/5] Vérification GPU..."
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "  GPU: ${GPU_COUNT}x ${GPU_NAME} (${VRAM}MB)"
else
    GPU_COUNT=0
    echo "  ATTENTION: Pas de GPU détecté — training sur CPU (lent!)"
fi

# ── 4. HF Token ──
echo ""
echo "[4/5] HuggingFace token..."
if [ -z "$TOKEN" ]; then
    echo "  NON DÉFINI — export HF_TOKEN=hf_xxx avant de lancer"
    echo "  Le data download HF peut échouer sans token"
else
    echo "  OK: ${TOKEN:0:8}..."
    export HF_TOKEN="$TOKEN"
fi

# ── 5. Lancer le training ──
echo ""
echo "[5/5] Lancement du training..."
echo "  max_steps=${MAX_STEPS} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} seq_len=${SEQ_LEN}"
echo ""

export COGNET_WORKSPACE="$(pwd)"

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "  Multi-GPU détecté → torchrun avec ${GPU_COUNT} GPUs"
    python -m torch.distributed.run \
        --standalone \
        --nproc_per_node="$GPU_COUNT" \
        train_ultra.py \
        --max-steps "$MAX_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --seq-len "$SEQ_LEN" \
        --use-fsdp \
        --compile \
        --cuda-prefetch \
        --seq-warmup \
        --async-ckpt
else
    echo "  Single GPU/CPU → python direct"
    python train_ultra.py \
        --max-steps "$MAX_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --seq-len "$SEQ_LEN" \
        --compile \
        --cuda-prefetch \
        --seq-warmup \
        --async-ckpt
fi
