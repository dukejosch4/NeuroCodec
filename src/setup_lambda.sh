#!/bin/bash
# ================================================================
# Lambda Instance Setup Script
#
# Run this ONCE after SSH-ing into a fresh Lambda A100 instance.
# Prepares the environment, uploads checkpoints, verifies everything.
#
# Usage (from LOCAL machine):
#   1. Start Lambda instance (A100-40GB)
#   2. SSH: ssh -i lambda.pem ubuntu@<IP>
#   3. git clone the repo (or scp)
#   4. bash src/setup_lambda.sh
#
# Or from LOCAL machine for full automated setup:
#   LAMBDA_IP=<IP> bash src/setup_lambda_remote.sh
# ================================================================

set -euo pipefail

echo "================================================================"
echo "LAMBDA INSTANCE SETUP"
echo "================================================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

# ── 1. System info ──
echo ""
echo "[1/7] System info..."
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  RAM: $(free -h | awk '/Mem:/ {print $2}')"
echo "  Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "  Python: $(python3 --version 2>/dev/null || echo 'not found')"

# ── 2. Python environment ──
echo ""
echo "[2/7] Setting up Python environment..."
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    echo "  Created .venv"
fi
source .venv/bin/activate
echo "  Activated .venv ($(python --version))"

# ── 3. Install dependencies ──
echo ""
echo "[3/7] Installing dependencies..."
pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || \
    pip install torch torchvision -q
pip install numpy scipy scikit-image lpips torchmetrics diffusers transformers accelerate -q
pip install av -q  # for torchvision.io video reading

# Verify torch CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# ── 4. Verify PYTHONPATH and imports ──
echo ""
echo "[4/7] Verifying imports..."
export PYTHONPATH="$SCRIPT_DIR"
python -c "from models import SlotLatentAutoencoderV2, DynamicsTransformer, ResidualDecoderV2, ManifoldProjector; print('  models.py: OK')"
python -c "from losses import combined_loss, spectral_loss; print('  losses.py: OK')"
python -c "from vae_utils import load_cogvideox_vae; print('  vae_utils.py: OK')"

# ── 5. Create directory structure ──
echo ""
echo "[5/7] Creating directories..."
mkdir -p checkpoints/ssv2
mkdir -p data/ssv2_latents
mkdir -p data/ssv2_slot_pairs
mkdir -p results/json
mkdir -p logs
echo "  Done"

# ── 6. Check data ──
echo ""
echo "[6/7] Checking data..."
N_WEBM=$(find data/ssv2/20bn-something-something-v2/ -name '*.webm' 2>/dev/null | wc -l | tr -d ' ')
N_LATENTS=$(find data/ssv2_latents/ -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')
N_SHARDS=$(find data/ssv2_slot_pairs/ -name 'shard_*.npy' 2>/dev/null | wc -l | tr -d ' ')
echo "  WebM videos: $N_WEBM"
echo "  Latent files: $N_LATENTS"
echo "  Slot pair shards: $N_SHARDS"

if [[ "$N_WEBM" -eq 0 && "$N_LATENTS" -eq 0 ]]; then
    echo ""
    echo "  WARNING: No data found!"
    echo "  You need either:"
    echo "    a) SSv2 WebM videos in data/ssv2/20bn-something-something-v2/"
    echo "    b) Pre-encoded latents in data/ssv2_latents/"
    echo ""
    echo "  For SSv2 download, see: https://developer.qualcomm.com/software/ai-datasets/something-something"
    echo "  Or transfer pre-encoded latents from a previous run."
fi

# ── 7. Check checkpoints ──
echo ""
echo "[7/7] Checking checkpoints..."
for ckpt in slot_encoder_best.pt dynamics_best.pt residual_spectral_best.pt manifold_projector_best.pt simvp_best.pt; do
    if [[ -f "checkpoints/ssv2/$ckpt" ]]; then
        SIZE=$(du -h "checkpoints/ssv2/$ckpt" | cut -f1)
        echo "  $ckpt: EXISTS ($SIZE)"
    else
        echo "  $ckpt: MISSING"
    fi
done

# ── Summary ──
echo ""
echo "================================================================"
echo "SETUP COMPLETE"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Upload checkpoints (if available locally):"
echo "     scp -i lambda.pem checkpoints/ssv2/slot_encoder_best.pt ubuntu@\$(hostname -I | awk '{print \$1}'):$BASE_DIR/checkpoints/ssv2/"
echo "     scp -i lambda.pem checkpoints/ssv2/dynamics_best.pt ubuntu@\$(hostname -I | awk '{print \$1}'):$BASE_DIR/checkpoints/ssv2/"
echo ""
echo "  2. Ensure SSv2 data is available (videos or latents)"
echo ""
echo "  3. Run the benchmark:"
echo "     source .venv/bin/activate"
echo "     bash src/run_ssv2_benchmark.sh 2>&1 | tee benchmark.log"
echo ""
echo "  4. For a quick test first:"
echo "     bash src/run_ssv2_benchmark.sh --dry-run"
echo ""
echo "  5. If latents already exist (from previous run):"
echo "     bash src/run_ssv2_benchmark.sh --skip-encode"
