#!/bin/bash
# ================================================================
# Upload code + checkpoints to Lambda instance (run from LOCAL Mac)
#
# Usage:
#   LAMBDA_IP=<ip> bash src/upload_to_lambda.sh
#   LAMBDA_IP=<ip> LAMBDA_KEY=~/.ssh/lambda.pem bash src/upload_to_lambda.sh
# ================================================================

set -euo pipefail

LAMBDA_IP="${LAMBDA_IP:?Set LAMBDA_IP=<your-instance-ip>}"
LAMBDA_KEY="${LAMBDA_KEY:-lambda.pem}"
LAMBDA_USER="${LAMBDA_USER:-ubuntu}"
REMOTE_DIR="/home/$LAMBDA_USER/NeuroCodec"

SSH_OPTS="-i $LAMBDA_KEY -o StrictHostKeyChecking=no"
SCP="scp $SSH_OPTS"
SSH="ssh $SSH_OPTS $LAMBDA_USER@$LAMBDA_IP"

echo "================================================================"
echo "UPLOADING TO LAMBDA: $LAMBDA_USER@$LAMBDA_IP"
echo "================================================================"

# ── 1. Create remote dirs ──
echo "[1/4] Creating remote directories..."
$SSH "mkdir -p $REMOTE_DIR/{src,checkpoints/ssv2,data,results/json,logs}"

# ── 2. Upload source code ──
echo "[2/4] Uploading source code..."
$SCP src/*.py "$LAMBDA_USER@$LAMBDA_IP:$REMOTE_DIR/src/"
$SCP src/*.sh "$LAMBDA_USER@$LAMBDA_IP:$REMOTE_DIR/src/"

# Upload requirements if exists
if [[ -f requirements.txt ]]; then
    $SCP requirements.txt "$LAMBDA_USER@$LAMBDA_IP:$REMOTE_DIR/"
fi

# ── 3. Upload existing checkpoints ──
echo "[3/4] Uploading checkpoints..."
for ckpt in checkpoints/ssv2/slot_encoder_best.pt checkpoints/ssv2/dynamics_best.pt; do
    if [[ -f "$ckpt" ]]; then
        SIZE=$(du -h "$ckpt" | cut -f1)
        echo "  $(basename "$ckpt") ($SIZE)..."
        $SCP "$ckpt" "$LAMBDA_USER@$LAMBDA_IP:$REMOTE_DIR/$ckpt"
    else
        echo "  $(basename "$ckpt"): NOT FOUND LOCALLY (will train from scratch)"
    fi
done

# ── 4. Verify remote ──
echo "[4/4] Verifying remote..."
$SSH "ls -la $REMOTE_DIR/src/*.py | wc -l | xargs echo '  Python files:'"
$SSH "ls -la $REMOTE_DIR/checkpoints/ssv2/*.pt 2>/dev/null | wc -l | xargs echo '  Checkpoints:'" || echo "  Checkpoints: 0"

echo ""
echo "================================================================"
echo "UPLOAD COMPLETE"
echo "================================================================"
echo ""
echo "Next: SSH into the instance and run setup:"
echo "  ssh $SSH_OPTS $LAMBDA_USER@$LAMBDA_IP"
echo "  cd $REMOTE_DIR"
echo "  bash src/setup_lambda.sh"
echo "  bash src/run_ssv2_benchmark.sh"
