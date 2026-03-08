"""Phase 5: Train ManifoldProjector on SSv2.

Generates (z_pred, z_gt) pairs on-the-fly using the trained pipeline:
  latent_t -> slot_encoder -> dynamics -> residual_decoder -> z_pred
  z_gt = latent_t+1

Then trains the ManifoldProjector (46K params, ~30 min on A100).

Usage:
    python src/train_ssv2_projector.py \
        --slot-ckpt checkpoints/ssv2/slot_encoder_best.pt \
        --dynamics-ckpt checkpoints/ssv2/dynamics_best.pt \
        --residual-ckpt checkpoints/ssv2/residual_spectral_best.pt

Output:
    checkpoints/ssv2/manifold_projector_best.pt
    results/json/ssv2_projector_training.json
"""

import argparse
import bisect
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

LATENT_DIR = Path("data/ssv2_latents")
CHECKPOINT_DIR = Path("checkpoints/ssv2")
RESULTS_DIR = Path("results/json")

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 100
SPECTRAL_BETA = 0.01
SEED = 42
VAL_FRACTION = 0.15
MAX_PAIRS_PER_VIDEO = 1  # subsample: 1 pair per video to keep dataset manageable


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class LatentSinglePairDataset(Dataset):
    """Yields one (latent_t, latent_t+1) pair per video (first pair)."""

    def __init__(self, latent_files):
        self.files = latent_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        latent = np.load(self.files[idx]).astype(np.float32)
        # Use first frame pair: t=0, t+1=1
        lt = latent[:, 0, :, :]   # [16, 32, 32]
        lt1 = latent[:, 1, :, :]  # [16, 32, 32]
        return torch.from_numpy(lt.copy()), torch.from_numpy(lt1.copy())


def main():
    parser = argparse.ArgumentParser(description="SSv2 ManifoldProjector Training")
    parser.add_argument("--slot-ckpt", type=str, required=True)
    parser.add_argument("--dynamics-ckpt", type=str, required=True)
    parser.add_argument("--residual-ckpt", type=str, required=True)
    parser.add_argument("--latent-dir", type=str, default=str(LATENT_DIR))
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.max_videos = 200
        args.epochs = 10

    device = "cuda"
    torch.manual_seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("PHASE 5: MANIFOLD PROJECTOR TRAINING (SSv2)")
    log("=" * 60)

    # ── Pre-flight ──
    for ckpt_name, ckpt_path in [("slot", args.slot_ckpt), ("dynamics", args.dynamics_ckpt),
                                   ("residual", args.residual_ckpt)]:
        if not os.path.exists(ckpt_path):
            log(f"  ERROR: {ckpt_name} checkpoint not found: {ckpt_path}")
            return
    if not torch.cuda.is_available():
        log("  ERROR: CUDA not available")
        return

    # ── Load frozen models ──
    log("\n[1/4] Loading frozen pipeline models...")
    from models import SlotLatentAutoencoderV2, DynamicsTransformer, ResidualDecoderV2, ManifoldProjector

    slot_encoder = SlotLatentAutoencoderV2(
        n_slots=64, slot_dim=128, input_dim=16, n_tokens=1024, n_iter=5
    ).to(device)
    slot_encoder.load_state_dict(torch.load(args.slot_ckpt, map_location=device, weights_only=False))
    slot_encoder.eval()
    log(f"  Slot encoder: loaded")

    dynamics = DynamicsTransformer(n_tokens=64, token_dim=128, d_model=128, n_heads=4, n_layers=2).to(device)
    dynamics.load_state_dict(torch.load(args.dynamics_ckpt, map_location=device, weights_only=False))
    dynamics.eval()
    log(f"  Dynamics: loaded")

    residual = ResidualDecoderV2().to(device)
    residual.load_state_dict(torch.load(args.residual_ckpt, map_location=device, weights_only=False))
    residual.eval()
    log(f"  Residual decoder: loaded")

    # ── Dataset ──
    log("\n[2/4] Building dataset...")
    latent_dir = Path(args.latent_dir)
    files = sorted(latent_dir.glob("*.npy"))
    if args.max_videos:
        files = files[:args.max_videos]
    files = [str(f) for f in files]

    # Train/val split (same as residual training)
    n = len(files)
    indices = list(range(n))
    random.seed(args.seed)
    random.shuffle(indices)
    n_val = int(n * VAL_FRACTION)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    full_ds = LatentSinglePairDataset(files)
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    log(f"  Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    # ── Model ──
    log("\n[3/4] Creating ManifoldProjector...")
    projector = ManifoldProjector().to(device)
    n_params = sum(p.numel() for p in projector.parameters())
    log(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(projector.parameters(), lr=LR, weight_decay=0.01)
    total_steps = args.epochs * len(train_dl)

    def lr_lambda(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Compute baseline ──
    log("\n  Computing baseline (no projector)...")
    from losses import spectral_loss
    baseline_mse_sum, baseline_n = 0.0, 0

    with torch.no_grad():
        for lt_batch, lt1_batch in val_dl:
            lt_batch = lt_batch.to(device)
            lt1_batch = lt1_batch.to(device)

            # Run frozen pipeline
            tok_t = lt_batch.flatten(2).permute(0, 2, 1)
            slots_t = slot_encoder.encode(tok_t)
            slots_t1 = dynamics(slots_t)
            delta = residual(tok_t, slots_t, slots_t1)
            z_pred = (tok_t + delta).permute(0, 2, 1).reshape(-1, 16, 32, 32)

            baseline_mse_sum += F.mse_loss(z_pred, lt1_batch).item()
            baseline_n += 1

    baseline_mse = baseline_mse_sum / baseline_n
    log(f"  Baseline MSE (no projector): {baseline_mse:.6f}")

    # ── Training ──
    log(f"\n[4/4] Training for {args.epochs} epochs (beta={SPECTRAL_BETA})...")
    best_val = float("inf")
    history = []
    save_path = str(CHECKPOINT_DIR / "manifold_projector_best.pt")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        projector.train()
        ep_loss = 0.0
        n_batches = 0

        for lt_batch, lt1_batch in train_dl:
            lt_batch = lt_batch.to(device)
            lt1_batch = lt1_batch.to(device)

            with torch.no_grad():
                tok_t = lt_batch.flatten(2).permute(0, 2, 1)
                slots_t = slot_encoder.encode(tok_t)
                slots_t1 = dynamics(slots_t)
                delta = residual(tok_t, slots_t, slots_t1)
                z_pred = (tok_t + delta).permute(0, 2, 1).reshape(-1, 16, 32, 32)

            z_proj = projector(z_pred)
            l_mse = F.mse_loss(z_proj, lt1_batch)
            l_spec = spectral_loss(z_proj, lt1_batch) if SPECTRAL_BETA > 0 else torch.tensor(0.0)
            loss = l_mse + SPECTRAL_BETA * l_spec

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            n_batches += 1

        # Validate
        if epoch <= 5 or epoch % 5 == 0 or epoch == args.epochs:
            projector.eval()
            val_mse_sum, val_n = 0.0, 0

            with torch.no_grad():
                for lt_batch, lt1_batch in val_dl:
                    lt_batch = lt_batch.to(device)
                    lt1_batch = lt1_batch.to(device)

                    tok_t = lt_batch.flatten(2).permute(0, 2, 1)
                    slots_t = slot_encoder.encode(tok_t)
                    slots_t1 = dynamics(slots_t)
                    delta = residual(tok_t, slots_t, slots_t1)
                    z_pred = (tok_t + delta).permute(0, 2, 1).reshape(-1, 16, 32, 32)
                    z_proj = projector(z_pred)

                    val_mse_sum += F.mse_loss(z_proj, lt1_batch).item()
                    val_n += 1

            val_mse = val_mse_sum / val_n
            improved = val_mse < best_val
            if improved:
                best_val = val_mse
                torch.save(projector.state_dict(), save_path)

            improvement = (1 - val_mse / baseline_mse) * 100
            elapsed = time.time() - t_start
            log(f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Val MSE: {val_mse:.6f} (improv: {improvement:+.1f}%) | "
                f"{elapsed / 60:.1f}min {'*' if improved else ''}")

            history.append({"epoch": epoch, "val_mse": val_mse, "improvement_pct": improvement})

    elapsed = time.time() - t_start
    final_improvement = (1 - best_val / baseline_mse) * 100

    log(f"\n{'='*60}")
    log("PROJECTOR TRAINING RESULTS")
    log(f"{'='*60}")
    log(f"  Baseline MSE: {baseline_mse:.6f}")
    log(f"  Best val MSE: {best_val:.6f}")
    log(f"  Improvement:  {final_improvement:+.1f}%")
    log(f"  Training time: {elapsed / 60:.1f}min")

    results = {
        "phase": "projector_training",
        "dataset": "ssv2",
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_params": n_params,
        "baseline_mse": baseline_mse,
        "best_val_mse": best_val,
        "improvement_pct": final_improvement,
        "training_minutes": elapsed / 60,
        "history": history,
        "gpu": torch.cuda.get_device_name(0),
    }
    out_path = RESULTS_DIR / "ssv2_projector_training.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
