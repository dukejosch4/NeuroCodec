#!/bin/bash
# ================================================================
# SSv2 Benchmark Pipeline — Complete Orchestrator
#
# Runs the FULL benchmark pipeline for NeuroCodec paper revision:
#   Phase 0: Preflight validation (GPU, disk, Python, data)
#   Phase 1: VAE encoding (if latents don't exist, ~18h)
#   Phase 2: Slot encoder training (if checkpoint missing, ~2h)
#   Phase 3: Dynamics training + shard extraction (~3h)
#   Phase 4: Residual decoder training (~2h)
#   Phase 5: ManifoldProjector training (~0.5h)
#   Phase 6: SimVP baseline training (~1h)
#   Phase 7: Full pixel evaluation (~2h)
#   Phase 8: Results collection + download instructions
#
# Idempotent: safe to restart after failure. Each phase checks for
# existing outputs and skips if already complete.
#
# Usage:
#   bash src/run_ssv2_benchmark.sh [--dry-run] [--skip-encode]
#
# Expected runtime: 28-32h on A100-40GB (full pipeline from scratch)
# Expected cost:    ~$30-35 on Lambda ($1.10/h)
#
# Prerequisites:
#   - SSv2 WebM videos in data/ssv2/20bn-something-something-v2/*.webm
#   - OR: pre-encoded latents in data/ssv2_latents/*.npy (use --skip-encode)
#   - Slot/Dynamics checkpoints can be pre-uploaded or trained from scratch
# ================================================================

set -euo pipefail

# ── Configuration ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$BASE_DIR/logs"
CKPT_DIR="$BASE_DIR/checkpoints/ssv2"
RESULT_DIR="$BASE_DIR/results/json"
LATENT_DIR="$BASE_DIR/data/ssv2_latents"
SHARD_DIR="$BASE_DIR/data/ssv2_slot_pairs"
VIDEO_DIR="$BASE_DIR/data/ssv2/20bn-something-something-v2"

export PYTHONPATH="$SCRIPT_DIR"

DRY_RUN=""
SKIP_ENCODE=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run"; echo "[DRY-RUN MODE] Minimal data" ;;
        --skip-encode) SKIP_ENCODE="1"; echo "[SKIP-ENCODE] Assuming latents exist" ;;
    esac
done

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$RESULT_DIR" "$LATENT_DIR" "$SHARD_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/benchmark_${TIMESTAMP}.log"

# ── Logging helper ──
log() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"
}

fail() {
    log "FATAL: $*"
    log "Pipeline aborted."
    exit 1
}

phase_header() {
    log ""
    log "================================================================"
    log "PHASE $1: $2"
    log "================================================================"
}

# ── GPU monitoring (background) ──
start_gpu_monitor() {
    while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader >> "$LOG_DIR/gpu_monitor_${TIMESTAMP}.csv" 2>/dev/null
        sleep 60
    done &
    GPU_MONITOR_PID=$!
    log "GPU monitor started (PID: $GPU_MONITOR_PID)"
}

stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        log "GPU monitor stopped"
    fi
}
trap stop_gpu_monitor EXIT

# ================================================================
# PHASE 0: PREFLIGHT
# ================================================================
phase_header 0 "PREFLIGHT VALIDATION"

# Check GPU
if ! nvidia-smi &>/dev/null; then
    fail "No NVIDIA GPU detected"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
log "GPU: $GPU_NAME ($GPU_MEM)"

# Check disk space
DISK_FREE=$(df -BG "$BASE_DIR" | tail -1 | awk '{print $4}' | tr -d 'G')
log "Disk free: ${DISK_FREE}GB"
if [[ -z "$SKIP_ENCODE" && "$DISK_FREE" -lt 100 ]]; then
    fail "Insufficient disk: ${DISK_FREE}GB (need >=100GB for encoding)"
elif [[ -n "$SKIP_ENCODE" && "$DISK_FREE" -lt 10 ]]; then
    fail "Insufficient disk: ${DISK_FREE}GB (need >=10GB)"
fi

# Activate virtualenv
cd "$BASE_DIR"
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    log "Activated .venv"
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    log "Activated venv"
fi

# Check Python + PyTorch
python -c "import torch; assert torch.cuda.is_available(); print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" \
    || fail "PyTorch CUDA check failed"

# Check core imports
python -c "from models import SlotLatentAutoencoderV2, DynamicsTransformer, ResidualDecoderV2, ManifoldProjector; print('models.py: OK')" \
    || fail "Cannot import models. Check PYTHONPATH=$PYTHONPATH"
python -c "from losses import combined_loss, spectral_loss; print('losses.py: OK')" \
    || fail "Cannot import losses"

# Install missing dependencies
for entry in lpips:lpips scipy:scipy scikit-image:skimage torchmetrics:torchmetrics diffusers:diffusers torchvision:torchvision; do
    pkg="${entry%%:*}"
    mod="${entry##*:}"
    python -c "import $mod" 2>/dev/null || {
        log "Installing missing package: $pkg"
        pip install "$pkg" -q || log "WARNING: Failed to install $pkg"
    }
done

# Check data availability
N_LATENTS=$(find "$LATENT_DIR" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')
log "Latent files found: $N_LATENTS"

if [[ "$N_LATENTS" -lt 100 && -z "$SKIP_ENCODE" ]]; then
    # Need to encode — check for raw videos
    N_VIDEOS=$(find "$VIDEO_DIR" -name '*.webm' 2>/dev/null | wc -l | tr -d ' ')
    log "Raw WebM videos found: $N_VIDEOS"
    if [[ "$N_VIDEOS" -lt 1000 && -z "$DRY_RUN" ]]; then
        fail "Too few WebM videos: $N_VIDEOS (need SSv2 dataset in $VIDEO_DIR)"
    fi
elif [[ "$N_LATENTS" -lt 100 && -n "$SKIP_ENCODE" ]]; then
    fail "Too few latent files ($N_LATENTS) and --skip-encode set. Cannot proceed."
fi

# Check existing checkpoints
SLOT_CKPT="$CKPT_DIR/slot_encoder_best.pt"
DYN_CKPT="$CKPT_DIR/dynamics_best.pt"
RES_CKPT="$CKPT_DIR/residual_spectral_best.pt"
PROJ_CKPT="$CKPT_DIR/manifold_projector_best.pt"
SIMVP_CKPT="$CKPT_DIR/simvp_best.pt"

log "Checkpoints:"
for f in "$SLOT_CKPT" "$DYN_CKPT" "$RES_CKPT" "$PROJ_CKPT" "$SIMVP_CKPT"; do
    if [[ -f "$f" ]]; then
        log "  $(basename "$f"): EXISTS ($(du -h "$f" | cut -f1))"
    else
        log "  $(basename "$f"): MISSING"
    fi
done

log "PREFLIGHT PASSED"
start_gpu_monitor

# ================================================================
# PHASE 1: VAE ENCODING
# ================================================================
MIN_LATENTS=1000
if [[ -n "$DRY_RUN" ]]; then MIN_LATENTS=50; fi

if [[ "$N_LATENTS" -ge "$MIN_LATENTS" ]]; then
    phase_header 1 "VAE ENCODING — SKIPPED ($N_LATENTS latents exist)"
else
    if [[ -n "$SKIP_ENCODE" ]]; then
        phase_header 1 "VAE ENCODING — SKIPPED (--skip-encode flag)"
    else
        phase_header 1 "VAE ENCODING (~18h for 220K videos)"

        # Run preflight first (quick test of 100 videos)
        log "Running VAE preflight (100 videos)..."
        python src/ssv2_pipeline.py --phase preflight \
            2>&1 | tee "$LOG_DIR/encode_preflight_${TIMESTAMP}.log"

        PREFLIGHT_EXIT=${PIPESTATUS[0]}
        if [[ "$PREFLIGHT_EXIT" -ne 0 ]]; then
            fail "VAE preflight failed (exit $PREFLIGHT_EXIT)"
        fi

        # Check kill criterion
        SSIM=$(python -c "
import json
r = json.load(open('results/json/ssv2_preflight.json'))
print(f\"{r['vae_ssim_mean']:.4f}\")
passed = r['kill_criterion_passed']
print('PASSED' if passed else 'FAILED')
")
        log "Preflight SSIM: $SSIM"
        if echo "$SSIM" | grep -q "FAILED"; then
            fail "VAE quality too low (SSIM < 0.90). Check GPU/model."
        fi

        # Full encoding
        log "Starting full VAE encoding..."
        python src/ssv2_pipeline.py --phase encode \
            2>&1 | tee "$LOG_DIR/encode_full_${TIMESTAMP}.log"

        ENCODE_EXIT=${PIPESTATUS[0]}
        if [[ "$ENCODE_EXIT" -ne 0 ]]; then
            fail "VAE encoding failed (exit $ENCODE_EXIT)"
        fi

        # Verify
        N_LATENTS=$(find "$LATENT_DIR" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')
        log "Encoding complete. Latent files: $N_LATENTS"
        if [[ "$N_LATENTS" -lt "$MIN_LATENTS" ]]; then
            fail "Too few latents after encoding: $N_LATENTS"
        fi

        log "Phase 1 complete."
    fi
fi

# Refresh latent count after encoding
N_LATENTS=$(find "$LATENT_DIR" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')
log "Working with $N_LATENTS latent files"

# ================================================================
# PHASE 2: SLOT ENCODER TRAINING
# ================================================================
if [[ -f "$SLOT_CKPT" ]]; then
    phase_header 2 "SLOT ENCODER — SKIPPED (checkpoint exists)"
else
    phase_header 2 "SLOT ENCODER TRAINING (~2h)"

    SLOT_EPOCHS=50
    SLOT_ARGS=""
    if [[ -n "$DRY_RUN" ]]; then
        SLOT_EPOCHS=3
        SLOT_ARGS="--max-videos 200"
    fi

    python src/train_ssv2_slots.py \
        --epochs "$SLOT_EPOCHS" \
        $SLOT_ARGS \
        2>&1 | tee "$LOG_DIR/slots_${TIMESTAMP}.log"

    PHASE2_EXIT=${PIPESTATUS[0]}
    if [[ "$PHASE2_EXIT" -ne 0 ]]; then
        fail "Slot encoder training failed (exit $PHASE2_EXIT)"
    fi
    [[ -f "$SLOT_CKPT" ]] || fail "Slot encoder checkpoint not created"
    log "Phase 2 complete."
fi

# ================================================================
# PHASE 3: DYNAMICS TRAINING + SHARD EXTRACTION
# ================================================================
if [[ -f "$DYN_CKPT" ]]; then
    phase_header 3 "DYNAMICS TRANSFORMER — SKIPPED (checkpoint exists)"

    # But check if shards exist (needed for residual training)
    N_SHARDS=$(find "$SHARD_DIR" -name 'shard_*.npy' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$N_SHARDS" -eq 0 ]]; then
        log "WARNING: Dynamics checkpoint exists but no shards found."
        log "  Residual training will use on-the-fly extraction (slower but works)."
    else
        log "Slot pair shards: $N_SHARDS"
    fi
else
    phase_header 3 "DYNAMICS TRAINING + SHARD EXTRACTION (~3h)"

    DYN_ARGS="--slot-ckpt $SLOT_CKPT"
    if [[ -n "$DRY_RUN" ]]; then
        DYN_ARGS="$DYN_ARGS --max-videos 200"
    fi

    python src/train_ssv2_dynamics.py $DYN_ARGS \
        2>&1 | tee "$LOG_DIR/dynamics_${TIMESTAMP}.log"

    PHASE3_EXIT=${PIPESTATUS[0]}
    if [[ "$PHASE3_EXIT" -ne 0 ]]; then
        fail "Dynamics training failed (exit $PHASE3_EXIT)"
    fi
    [[ -f "$DYN_CKPT" ]] || fail "Dynamics checkpoint not created"

    N_SHARDS=$(find "$SHARD_DIR" -name 'shard_*.npy' 2>/dev/null | wc -l | tr -d ' ')
    log "Shards created: $N_SHARDS"
    log "Phase 3 complete."
fi

# ================================================================
# PHASE 4: RESIDUAL DECODER TRAINING
# ================================================================
if [[ -f "$RES_CKPT" && -z "$DRY_RUN" ]]; then
    phase_header 4 "RESIDUAL DECODER — SKIPPED (checkpoint exists)"
else
    phase_header 4 "RESIDUAL DECODER TRAINING (~2h)"

    RES_ARGS="--slot-ckpt $SLOT_CKPT --latent-dir $LATENT_DIR --shard-dir $SHARD_DIR --max-videos 20000"
    if [[ -n "$DRY_RUN" ]]; then
        RES_ARGS="$RES_ARGS $DRY_RUN"
    fi

    python src/train_ssv2_residual.py $RES_ARGS \
        2>&1 | tee "$LOG_DIR/residual_${TIMESTAMP}.log"

    PHASE4_EXIT=${PIPESTATUS[0]}
    if [[ "$PHASE4_EXIT" -ne 0 ]]; then
        fail "Residual training failed (exit $PHASE4_EXIT)"
    fi
    [[ -f "$RES_CKPT" ]] || [[ -n "$DRY_RUN" ]] || fail "Residual checkpoint not created"
    log "Phase 4 complete."
fi

# ================================================================
# PHASE 5: MANIFOLD PROJECTOR TRAINING
# ================================================================
if [[ -f "$PROJ_CKPT" && -z "$DRY_RUN" ]]; then
    phase_header 5 "MANIFOLD PROJECTOR — SKIPPED (checkpoint exists)"
else
    phase_header 5 "MANIFOLD PROJECTOR TRAINING (~30min)"

    PROJ_ARGS="--slot-ckpt $SLOT_CKPT --dynamics-ckpt $DYN_CKPT --residual-ckpt $RES_CKPT --latent-dir $LATENT_DIR --max-videos 20000"
    if [[ -n "$DRY_RUN" ]]; then
        PROJ_ARGS="$PROJ_ARGS $DRY_RUN"
    fi

    python src/train_ssv2_projector.py $PROJ_ARGS \
        2>&1 | tee "$LOG_DIR/projector_${TIMESTAMP}.log"

    PHASE5_EXIT=${PIPESTATUS[0]}
    if [[ "$PHASE5_EXIT" -ne 0 ]]; then
        log "WARNING: Projector training failed. Continuing without projector."
    else
        log "Phase 5 complete."
    fi
fi

# ================================================================
# PHASE 6: SIMVP BASELINE TRAINING
# ================================================================
if [[ -f "$SIMVP_CKPT" && -z "$DRY_RUN" ]]; then
    phase_header 6 "SIMVP BASELINE — SKIPPED (checkpoint exists)"
else
    phase_header 6 "SIMVP BASELINE TRAINING (~1h)"

    SIMVP_ARGS="--latent-dir $LATENT_DIR --max-videos 20000"
    if [[ -n "$DRY_RUN" ]]; then
        SIMVP_ARGS="$SIMVP_ARGS $DRY_RUN"
    fi

    python src/train_simvp_baseline.py $SIMVP_ARGS \
        2>&1 | tee "$LOG_DIR/simvp_${TIMESTAMP}.log"

    PHASE6_EXIT=${PIPESTATUS[0]}
    if [[ "$PHASE6_EXIT" -ne 0 ]]; then
        log "WARNING: SimVP training failed. Evaluation will run without SimVP."
    else
        log "Phase 6 complete."
    fi
fi

# ================================================================
# PHASE 7: FULL PIXEL EVALUATION
# ================================================================
phase_header 7 "FULL PIXEL EVALUATION (~2h)"

EVAL_ARGS=(
    --slot-ckpt "$SLOT_CKPT"
    --dynamics-ckpt "$DYN_CKPT"
    --residual-ckpt "$RES_CKPT"
    --latent-dir "$LATENT_DIR"
)

if [[ -f "$PROJ_CKPT" ]]; then
    EVAL_ARGS+=(--projector-ckpt "$PROJ_CKPT")
fi
if [[ -f "$SIMVP_CKPT" ]]; then
    EVAL_ARGS+=(--simvp-ckpt "$SIMVP_CKPT")
fi
if [[ -n "$DRY_RUN" ]]; then
    EVAL_ARGS+=($DRY_RUN)
fi

python src/eval_ssv2_pixels.py "${EVAL_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/eval_${TIMESTAMP}.log"

PHASE7_EXIT=${PIPESTATUS[0]}
if [[ "$PHASE7_EXIT" -ne 0 ]]; then
    log "WARNING: Evaluation failed (exit $PHASE7_EXIT)"
fi

# ================================================================
# PHASE 8: RESULTS COLLECTION
# ================================================================
phase_header 8 "RESULTS COLLECTION"

TARBALL="$BASE_DIR/ssv2_benchmark_results_${TIMESTAMP}.tar.gz"

# Collect all artifacts
tar czf "$TARBALL" \
    -C "$BASE_DIR" \
    checkpoints/ssv2/*.pt \
    results/json/ssv2_*.json \
    results/json/diffusion_gif_metrics.json \
    results/demo/diffusion_comparison_*.gif \
    checkpoints/diffusion_phase0/*.pt \
    logs/*_${TIMESTAMP}*.log \
    logs/gpu_monitor_${TIMESTAMP}.csv \
    2>/dev/null || true

TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)
log "Results tarball: $TARBALL ($TARBALL_SIZE)"
log ""
log "Download commands:"
log "  scp -i lambda.pem ubuntu@\$(hostname -I | awk '{print \$1}'):$TARBALL ."
log ""
log "Or download individual files:"
log "  # Checkpoints"
for f in "$CKPT_DIR"/*.pt; do
    [[ -f "$f" ]] && log "  scp -i lambda.pem ubuntu@\$(hostname -I | awk '{print \$1}'):$f ."
done
log "  # Results JSON"
for f in "$RESULT_DIR"/ssv2_*.json; do
    [[ -f "$f" ]] && log "  scp -i lambda.pem ubuntu@\$(hostname -I | awk '{print \$1}'):$f ."
done

# ================================================================
# PHASE 9: DIFFUSION GIF GENERATION (~2-3h)
# ================================================================
if [[ -f "$RES_CKPT" && -f "$DYN_CKPT" && -f "$SLOT_CKPT" ]]; then
    phase_header 9 "DIFFUSION GIF GENERATION (~2-3h)"

    DIFF_ARGS=(
        --slot-ckpt "$SLOT_CKPT"
        --dynamics-ckpt "$DYN_CKPT"
        --residual-ckpt "$RES_CKPT"
        --latent-dir "$LATENT_DIR"
        --n-train-videos 100
        --steps 10000
        --n-gifs 5
        --n-context 3
        --n-predict 6
    )

    if [[ -f "$PROJ_CKPT" ]]; then
        DIFF_ARGS+=(--projector-ckpt "$PROJ_CKPT")
    fi

    python src/generate_diffusion_gif.py "${DIFF_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/diffusion_gif_${TIMESTAMP}.log"

    PHASE9_EXIT=${PIPESTATUS[0]}
    if [[ "$PHASE9_EXIT" -ne 0 ]]; then
        log "WARNING: Diffusion GIF generation failed (exit $PHASE9_EXIT). Non-critical."
    else
        log "Phase 9 complete. GIFs in results/demo/"
    fi
else
    log ""
    log "SKIPPING Phase 9 (Diffusion GIF): missing required checkpoints"
fi

# ── Final summary ──
log ""
log "================================================================"
log "PIPELINE COMPLETE"
log "================================================================"
log "Checkpoints:"
for f in "$CKPT_DIR"/*.pt; do
    [[ -f "$f" ]] && log "  $(basename "$f") ($(du -h "$f" | cut -f1))"
done
log ""
log "Results:"
for f in "$RESULT_DIR"/ssv2_*.json; do
    [[ -f "$f" ]] && log "  $(basename "$f")"
done
log ""

# Cost estimate
if [[ -f "$LOG_DIR/gpu_monitor_${TIMESTAMP}.csv" ]]; then
    LINES=$(wc -l < "$LOG_DIR/gpu_monitor_${TIMESTAMP}.csv")
    HOURS=$(echo "scale=1; $LINES / 60" | bc 2>/dev/null || echo "?")
    COST=$(echo "scale=2; $HOURS * 1.10" | bc 2>/dev/null || echo "?")
    log "Estimated runtime: ~${HOURS}h"
    log "Estimated cost: ~\$${COST}"
fi

log ""
log "IMPORTANT: Download results before terminating the instance!"
log "  tar: $TARBALL"
log ""
log "Quick check — expected results for paper:"
log "  1. checkpoints/ssv2/*.pt (5 checkpoints)"
log "  2. results/json/ssv2_pixel_evaluation.json (SSIM, PSNR, LPIPS, FID)"
log "  3. results/json/ssv2_*_training.json (training histories)"
