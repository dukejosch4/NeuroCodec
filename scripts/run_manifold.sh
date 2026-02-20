#!/bin/bash
# ================================================================
# ManifoldProjector Pipeline — run on Lambda A100
#
# Trains a tiny ConvNet (~46K params) to project off-manifold
# latent predictions back onto the CogVideoX VAE manifold.
#
# Expected total time: ~20-30 min on A100
# ================================================================
set -euo pipefail

DATA_DIR="${HOME}"
CKPT_RES="${DATA_DIR}/residual_v2_spectral_beta0.01_best.pt"
CKPT_DYN="${DATA_DIR}/dynamics_best_d2.pt"
CKPT_SLOT="${DATA_DIR}/slot_v2_64slots_2k_best.pt"
PAIRS="${DATA_DIR}/manifold_pairs.pt"
CKPT_PROJ="${DATA_DIR}/manifold_projector_best.pt"
OUTPUT_DIR="${HOME}/NeuroCodec/results/json"

cd ~/NeuroCodec

echo "============================================"
echo "  Step 0: Pre-flight checks"
echo "============================================"
python scripts/manifold_preflight.py --data-dir "${DATA_DIR}"

echo ""
echo "============================================"
echo "  Step 1: Generate (z_pred, z_gt) pairs"
echo "============================================"
python scripts/manifold_generate_pairs.py \
    --data-dir "${DATA_DIR}" \
    --checkpoint "${CKPT_RES}" \
    --dynamics-checkpoint "${CKPT_DYN}" \
    --output "${PAIRS}" \
    --batch-size 64

echo ""
echo "============================================"
echo "  Step 2: Train ManifoldProjector"
echo "============================================"
python scripts/manifold_train.py \
    --pairs "${PAIRS}" \
    --output "${CKPT_PROJ}" \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --spectral-beta 0.01

echo ""
echo "============================================"
echo "  Step 3: Evaluate"
echo "============================================"
mkdir -p "${OUTPUT_DIR}"
python scripts/manifold_eval.py \
    --data-dir "${DATA_DIR}" \
    --res-checkpoint "${CKPT_RES}" \
    --dynamics-checkpoint "${CKPT_DYN}" \
    --slot-checkpoint "${CKPT_SLOT}" \
    --projector-checkpoint "${CKPT_PROJ}" \
    --n-videos 50 \
    --skip-lpips \
    --output "${OUTPUT_DIR}/manifold_eval.json"

echo ""
echo "============================================"
echo "  DONE — results at ${OUTPUT_DIR}/manifold_eval.json"
echo "============================================"
echo ""
echo "To download results:"
echo "  scp lambda:~/NeuroCodec/results/json/manifold_eval.json results/json/"
echo "  scp lambda:~/manifold_projector_best.pt checkpoints/"
echo "  scp lambda:~/manifold_projector_best_log.json results/json/"
