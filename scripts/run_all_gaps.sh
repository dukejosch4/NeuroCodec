#!/bin/bash
# ================================================================
# NeuroCodec — Run All Gap Fixes (Paper Revision)
# ================================================================
#
# Orchestrates Gap 1-3 fixes + visual generation.
# Total GPU time: ~40-50 minutes on A100-40GB.
#
# Usage:
#   ssh -i lambda.pem ubuntu@<ip>
#   cd ~/NeuroCodec  # or wherever repo is cloned
#   bash scripts/run_all_gaps.sh
#
# Prerequisites:
#   - latents_2000videos.pt in ~/
#   - aligned_video_slots.pt in ~/
#   - All checkpoints in ~/
#   - Python deps installed (torch, diffusers, lpips, etc.)
# ================================================================

set -euo pipefail

HOME_DIR="$HOME"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$RESULTS_DIR/gap_fixes_${TIMESTAMP}.log"

mkdir -p "$RESULTS_DIR/json" "$RESULTS_DIR/figures"

echo "=================================================="
echo "  NeuroCodec Gap Fixes — $(date)"
echo "  Repo: $REPO_DIR"
echo "  Data: $HOME_DIR"
echo "  Log:  $LOG_FILE"
echo "=================================================="

# Tee output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

# ── Step 0: Install Dependencies ──
echo ""
echo "[STEP 0] Installing dependencies..."
pip install -q 'numpy<2' einops diffusers transformers accelerate \
    lpips scikit-image scipy matplotlib Pillow torchmetrics 2>/dev/null || true

# ── Step 1: Pre-flight Check ──
echo ""
echo "[STEP 1] Pre-flight checks..."
python "$REPO_DIR/scripts/preflight.py" --data-dir "$HOME_DIR"
if [ $? -ne 0 ]; then
    echo "PREFLIGHT FAILED — fix errors above before continuing"
    exit 1
fi

# ── Step 2: Gap 2 — Fair Latency Benchmark (~5 min) ──
echo ""
echo "=================================================="
echo "[STEP 2] Gap 2: Fair Latency Benchmark"
echo "=================================================="
python "$REPO_DIR/scripts/gap2_fair_latency.py" \
    --output "$RESULTS_DIR/json/gap2_latency.json" \
    --n-iters 200

# ── Step 3: Gap 3 — Slot Robustness (~15 min) ──
echo ""
echo "=================================================="
echo "[STEP 3] Gap 3: Slot Robustness Experiment"
echo "=================================================="
python "$REPO_DIR/scripts/gap3_slot_robustness.py" \
    --data-dir "$HOME_DIR" \
    --checkpoint "$HOME_DIR/residual_v2_spectral_beta0.01_best.pt" \
    --dynamics-checkpoint "$HOME_DIR/dynamics_best_d2.pt" \
    --slot-checkpoint "$HOME_DIR/slot_v2_64slots_2k_best.pt" \
    --n-videos 50 \
    --output "$RESULTS_DIR/json/gap3_slot_robustness.json"

# ── Step 4: Gap 1 — Expanded Perceptual Eval (~25 min) ──
echo ""
echo "=================================================="
echo "[STEP 4] Gap 1: Expanded Perceptual Evaluation"
echo "=================================================="
python "$REPO_DIR/scripts/gap1_expanded_perceptual.py" \
    --data-dir "$HOME_DIR" \
    --checkpoint "$HOME_DIR/residual_v2_spectral_beta0.01_best.pt" \
    --mse-checkpoint "$HOME_DIR/residual_decoder_v2_best.pt" \
    --dynamics-checkpoint "$HOME_DIR/dynamics_best_d2.pt" \
    --n-val-videos 100 \
    --n-rollout-videos 50 \
    --output "$RESULTS_DIR/json/gap1_perceptual.json"

# ── Step 5: Visual Examples (~5 min) ──
echo ""
echo "=================================================="
echo "[STEP 5] Visual Examples for Paper"
echo "=================================================="
python "$REPO_DIR/scripts/generate_visuals.py" \
    --data-dir "$HOME_DIR" \
    --checkpoint "$HOME_DIR/residual_v2_spectral_beta0.01_best.pt" \
    --dynamics-checkpoint "$HOME_DIR/dynamics_best_d2.pt" \
    --output-dir "$RESULTS_DIR/figures/" \
    --n-examples 6 \
    --n-rollout-strips 3

# ── Done ──
echo ""
echo "=================================================="
echo "  ALL GAP FIXES COMPLETE — $(date)"
echo "=================================================="
echo ""
echo "Results:"
echo "  Gap 1: $RESULTS_DIR/json/gap1_perceptual.json"
echo "  Gap 2: $RESULTS_DIR/json/gap2_latency.json"
echo "  Gap 3: $RESULTS_DIR/json/gap3_slot_robustness.json"
echo "  Visuals: $RESULTS_DIR/figures/"
echo ""
echo "Next: scp results back and update paper/main.tex"
echo ""
echo "  scp -i lambda.pem -r ubuntu@<ip>:~/NeuroCodec/results/ ."
