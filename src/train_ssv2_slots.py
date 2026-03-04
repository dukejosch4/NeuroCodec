"""Phase 2: Train SlotLatentAutoencoderV2 on SSv2 latents.

Same architecture + hyperparameters as UCF-101. Only the data changes.
Uses lazy-loading Dataset to avoid RAM bottleneck (220K videos = ~183GB if loaded at once).

Usage:
    python src/train_ssv2_slots.py                    # Train with defaults
    python src/train_ssv2_slots.py --epochs 100       # Shorter run
    python src/train_ssv2_slots.py --resume ckpt.pt   # Resume from checkpoint

Expects: data/ssv2_latents/*.npy (from ssv2_pipeline.py --phase encode)
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Config ───────────────────────────────────────────────────────────
LATENT_DIR = Path("data/ssv2_latents")
CHECKPOINT_DIR = Path("checkpoints/ssv2")
RESULTS_DIR = Path("results/json")

# Identical to UCF-101 training
N_SLOTS = 64
SLOT_DIM = 128
INPUT_DIM = 16
N_TOKENS = 1024  # 32 × 32
N_ITER = 5

BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
VAL_FRACTION = 0.15


# ── Dataset (lazy-loading) ───────────────────────────────────────────
class SSv2FrameDataset(Dataset):
    """Lazy-loading dataset: reads .npy files on-the-fly.

    Builds a flat index of (file_idx, frame_idx) pairs at init (~3MB for 220K videos).
    Actual data is loaded per-__getitem__ call — no RAM bottleneck.
    """

    def __init__(self, files, frame_counts):
        self.files = files
        # Build flat index: [(file_idx, frame_idx), ...]
        self.index = []
        for fi, n_frames in enumerate(frame_counts):
            for t in range(n_frames):
                self.index.append((fi, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, t = self.index[idx]
        latent = np.load(self.files[fi]).astype(np.float32)  # [16, T, 32, 32]
        C, T, H, W = latent.shape
        # Extract single frame as tokens: [1024, 16]
        frame = latent[:, t, :, :]  # [16, 32, 32]
        tokens = frame.reshape(C, H * W).T  # [1024, 16]
        return torch.from_numpy(tokens)


def scan_latents(latent_dir, max_videos=None):
    """Scan latent files and count frames without loading data.

    Returns:
        files: list of Path objects
        frame_counts: list of ints (frames per video)
        n_total_frames: total frame count
    """
    files = sorted(latent_dir.glob("*.npy"))
    if max_videos:
        files = files[:max_videos]

    print(f"  Scanning {len(files)} latent files...")
    frame_counts = []
    for i, f in enumerate(files):
        # Only read header to get shape — np.load with mmap is fast
        arr = np.load(f, mmap_mode="r")
        frame_counts.append(arr.shape[1])  # [16, T, 32, 32]

        if (i + 1) % 50000 == 0:
            print(f"    Scanned {i + 1}/{len(files)}", flush=True)

    n_total = sum(frame_counts)
    print(f"  Total: {len(files)} videos, {n_total} frames")
    return files, frame_counts, n_total


# ── Training ─────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 2: SLOT ENCODER TRAINING (SSv2)")
    print("=" * 60)

    # Scan data (no loading — just count frames)
    print("\n[1/3] Scanning latent files...")
    files, frame_counts, n_total = scan_latents(LATENT_DIR, max_videos=args.max_videos)
    n_videos = len(files)

    # Train/val split at VIDEO level (not frame level) to avoid leakage
    indices = list(range(n_videos))
    random.seed(42)
    random.shuffle(indices)
    n_val_videos = int(n_videos * VAL_FRACTION)

    val_files = [files[i] for i in indices[:n_val_videos]]
    val_counts = [frame_counts[i] for i in indices[:n_val_videos]]
    train_files = [files[i] for i in indices[n_val_videos:]]
    train_counts = [frame_counts[i] for i in indices[n_val_videos:]]

    n_train_frames = sum(train_counts)
    n_val_frames = sum(val_counts)
    print(f"  Train: {len(train_files)} videos, {n_train_frames} frames")
    print(f"  Val: {len(val_files)} videos, {n_val_frames} frames")

    train_ds = SSv2FrameDataset(train_files, train_counts)
    val_ds = SSv2FrameDataset(val_files, val_counts)

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    print("\n[2/3] Creating model...")
    from models import SlotLatentAutoencoderV2
    model = SlotLatentAutoencoderV2(
        n_slots=N_SLOTS, slot_dim=SLOT_DIM,
        input_dim=INPUT_DIM, n_tokens=N_TOKENS, n_iter=N_ITER,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} parameters")

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)
        print(f"  Resumed from {args.resume}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = args.epochs * len(train_dl)

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    print(f"\n[3/3] Training for {args.epochs} epochs...")
    print(f"  Steps/epoch: {len(train_dl)}, Total steps: {total_steps}")

    best_val_mse = float("inf")
    best_var_explained = 0.0
    history = {"train_mse": [], "val_mse": [], "var_explained": [], "lr": []}
    t_start = time.time()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        for tok_batch in train_dl:
            tok_batch = tok_batch.to(device)
            recon, slots = model(tok_batch)
            loss = F.mse_loss(recon, tok_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            global_step += 1

        train_mse = ep_loss / len(train_dl)

        # Validate every 5 epochs (or every epoch for first 10)
        if epoch <= 10 or epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            resid_var_sum = 0.0
            total_var_sum = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for tok_batch in val_dl:
                    tok_batch = tok_batch.to(device)
                    recon, _ = model(tok_batch)
                    val_loss += F.mse_loss(recon, tok_batch).item()
                    resid_var_sum += (tok_batch - recon).var().item()
                    total_var_sum += tok_batch.var().item()
                    n_val_batches += 1

            val_mse = val_loss / n_val_batches
            var_explained = (1 - resid_var_sum / total_var_sum) * 100

            elapsed = time.time() - t_start
            lr_now = scheduler.get_last_lr()[0]

            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train MSE: {train_mse:.6f} | "
                  f"Val MSE: {val_mse:.6f} | "
                  f"VarExp: {var_explained:.1f}% | "
                  f"LR: {lr_now:.2e} | "
                  f"{elapsed/60:.1f}min", flush=True)

            history["train_mse"].append(train_mse)
            history["val_mse"].append(val_mse)
            history["var_explained"].append(var_explained)
            history["lr"].append(lr_now)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_var_explained = var_explained
                torch.save(model.state_dict(), CHECKPOINT_DIR / "slot_encoder_best.pt")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"slot_encoder_ep{epoch:03d}.pt")

    # Final results
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("SLOT TRAINING RESULTS")
    print("=" * 60)
    print(f"  Videos: {n_videos}")
    print(f"  Frames: {n_total}")
    print(f"  Best Val MSE: {best_val_mse:.6f}")
    print(f"  Best Var Explained: {best_var_explained:.1f}%")
    print(f"  Training time: {elapsed/3600:.1f}h")

    results = {
        "phase": "slot_training",
        "dataset": "ssv2",
        "n_videos": n_videos,
        "n_frames": n_total,
        "epochs": args.epochs,
        "best_val_mse": best_val_mse,
        "best_var_explained": best_var_explained,
        "training_hours": elapsed / 3600,
        "n_params": n_params,
        "history": history,
    }
    with open(RESULTS_DIR / "ssv2_slot_training.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {RESULTS_DIR / 'ssv2_slot_training.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSv2 Slot Encoder Training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Limit number of videos (for testing)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args)
