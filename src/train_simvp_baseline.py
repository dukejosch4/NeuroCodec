"""SimVP baseline for latent-space video prediction on SSv2.

Implements a SimVP-style ConvNet baseline that operates in the same
CogVideoX latent space as NeuroCodec for a fair comparison:
  - Same VAE backbone (CogVideoX-2B)
  - Same evaluation protocol (decode + SSIM/PSNR/LPIPS)
  - Similar parameter count (~2M vs NeuroCodec's ~1.87M)

Architecture:
  Input: T_in=4 consecutive latent frames [B, 4, 16, 32, 32]
  Encoder: Conv projection to hidden dim
  Translator: 6x residual ConvBlocks for spatial mixing
  Decoder: Conv projection back + residual connection
  Output: next latent frame [B, 16, 32, 32]

Usage:
    python src/train_simvp_baseline.py
    python src/train_simvp_baseline.py --dry-run  # test on 100 videos

Output:
    checkpoints/ssv2/simvp_best.pt
    results/json/ssv2_simvp_training.json
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

LATENT_DIR = Path("data/ssv2_latents")
CHECKPOINT_DIR = Path("checkpoints/ssv2")
RESULTS_DIR = Path("results/json")

T_IN = 4
HIDDEN = 128
N_BLOCKS = 6
BATCH_SIZE = 256
LR = 3e-4
EPOCHS = 30
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
VAL_FRACTION = 0.15
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Model ─────────────────────────────────────────────────────────────
class SimVPBlock(nn.Module):
    """Residual conv block with GroupNorm."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(self.norm(x))))


class SimVPLatent(nn.Module):
    """SimVP-style baseline for latent-space next-frame prediction.

    Predicts next_frame = last_input_frame + delta(input_window).
    Zero-initialized output for stable residual start.
    """

    def __init__(self, in_channels=16, T_in=4, hidden=128, n_blocks=6):
        super().__init__()
        self.T_in = T_in
        self.in_channels = in_channels

        # Encoder: project concatenated input frames to hidden dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * T_in, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GELU(),
        )

        # Translator: spatial mixing blocks
        self.blocks = nn.Sequential(*[SimVPBlock(hidden) for _ in range(n_blocks)])

        # Decoder: project back to latent dim
        self.decoder = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, in_channels, 1),
        )
        # Zero-init last layer for residual start
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x):
        """
        Args:
            x: [B, T_in, C, H, W] input window of latent frames

        Returns:
            pred: [B, C, H, W] predicted next latent frame
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T * C, H, W)
        h = self.encoder(x_flat)
        h = self.blocks(h)
        delta = self.decoder(h)
        return x[:, -1] + delta  # residual: last frame + predicted delta


# ── Dataset ───────────────────────────────────────────────────────────
class SimVPWindowDataset(Dataset):
    """Yields (input_window, target_frame) pairs from latent .npy files.

    For each video with T temporal frames, generates (T - T_in) windows:
      input: frames [t, t+1, ..., t+T_in-1]
      target: frame t+T_in
    """

    def __init__(self, latent_files, frame_counts, T_in=4):
        self.files = latent_files
        self.T_in = T_in

        # Build flat index
        windows_per_video = [max(0, T - T_in) for T in frame_counts]
        self.cumulative = [0]
        for w in windows_per_video:
            self.cumulative.append(self.cumulative[-1] + w)
        self.total = self.cumulative[-1]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        vid_idx = bisect.bisect_right(self.cumulative, idx) - 1
        win_idx = idx - self.cumulative[vid_idx]

        latent = np.load(self.files[vid_idx]).astype(np.float32)  # [16, T, 32, 32]

        # Input window: [T_in, 16, 32, 32]
        window = latent[:, win_idx:win_idx + self.T_in, :, :]  # [16, T_in, 32, 32]
        window = window.transpose(1, 0, 2, 3)  # [T_in, 16, 32, 32]

        # Target: [16, 32, 32]
        target = latent[:, win_idx + self.T_in, :, :]

        return torch.from_numpy(window.copy()), torch.from_numpy(target.copy())


# ── Helpers ───────────────────────────────────────────────────────────
def scan_latents(latent_dir, max_videos=None):
    files = sorted(Path(latent_dir).glob("*.npy"))
    if max_videos:
        files = files[:max_videos]

    log(f"  Scanning {len(files)} latent files...")
    frame_counts = []
    for i, f in enumerate(files):
        shape = np.load(f, mmap_mode="r").shape
        frame_counts.append(shape[1])
        if (i + 1) % 50000 == 0:
            log(f"    Scanned {i + 1}/{len(files)}")

    total_windows = sum(max(0, T - T_IN) for T in frame_counts)
    log(f"  Total: {len(files)} videos, {total_windows} windows (T_in={T_IN})")
    return [str(f) for f in files], frame_counts


def get_train_val_indices(n_videos, frame_counts, T_in):
    """Video-level split consistent with other scripts."""
    indices = list(range(n_videos))
    random.seed(SEED)
    random.shuffle(indices)
    n_val = int(n_videos * VAL_FRACTION)
    val_video_set = set(indices[:n_val])

    # Build pair_to_video mapping for windows
    pair_to_video = []
    for vid_idx, T in enumerate(frame_counts):
        for _ in range(max(0, T - T_in)):
            pair_to_video.append(vid_idx)

    train_idx = [i for i, v in enumerate(pair_to_video) if v not in val_video_set]
    val_idx = [i for i, v in enumerate(pair_to_video) if v in val_video_set]
    return train_idx, val_idx


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SimVP Latent Baseline Training")
    parser.add_argument("--latent-dir", type=str, default=str(LATENT_DIR))
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.max_videos = 200
        args.epochs = 3

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("SIMVP BASELINE TRAINING (SSv2 Latent Space)")
    log("=" * 60)
    log(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Scan data ──
    log("\n[1/4] Scanning latent files...")
    files, frame_counts = scan_latents(args.latent_dir, args.max_videos)
    n_videos = len(files)

    # ── Dataset ──
    log("\n[2/4] Building dataset...")
    full_ds = SimVPWindowDataset(files, frame_counts, T_IN)
    train_idx, val_idx = get_train_val_indices(n_videos, frame_counts, T_IN)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    log(f"  Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    # ── Model ──
    log("\n[3/4] Creating SimVP model...")
    model = SimVPLatent(
        in_channels=16, T_in=T_IN, hidden=args.hidden, n_blocks=args.n_blocks
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Parameters: {n_params:,}")
    log(f"  Architecture: T_in={T_IN}, hidden={args.hidden}, blocks={args.n_blocks}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = args.epochs * len(train_dl)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──
    log(f"\n[4/4] Training for {args.epochs} epochs...")
    log(f"  Steps/epoch: {len(train_dl)}, Total steps: {total_steps}")

    best_val = float("inf")
    best_copy_improvement = 0.0
    history = {"train_mse": [], "val_mse": [], "copy_mse": [], "improvement_pct": []}
    t_start = time.time()
    save_path = str(CHECKPOINT_DIR / "simvp_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0

        for window_batch, target_batch in train_dl:
            window_batch = window_batch.to(device)  # [B, T_in, 16, 32, 32]
            target_batch = target_batch.to(device)   # [B, 16, 32, 32]

            pred = model(window_batch)
            loss = F.mse_loss(pred, target_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            n_batches += 1

        train_mse = ep_loss / n_batches

        # Validate
        if epoch <= 5 or epoch % 3 == 0 or epoch == args.epochs:
            model.eval()
            val_loss = 0.0
            copy_loss = 0.0
            val_n = 0

            with torch.no_grad():
                for window_batch, target_batch in val_dl:
                    window_batch = window_batch.to(device)
                    target_batch = target_batch.to(device)

                    pred = model(window_batch)
                    val_loss += F.mse_loss(pred, target_batch).item()
                    # Copy baseline: last input frame
                    copy_loss += F.mse_loss(window_batch[:, -1], target_batch).item()
                    val_n += 1

            val_mse = val_loss / val_n
            copy_mse = copy_loss / val_n
            improvement = (copy_mse - val_mse) / copy_mse * 100

            improved = val_mse < best_val
            if improved:
                best_val = val_mse
                best_copy_improvement = improvement
                torch.save({
                    "model": model.state_dict(),
                    "config": {"in_channels": 16, "T_in": T_IN,
                               "hidden": args.hidden, "n_blocks": args.n_blocks},
                }, save_path)

            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            log(f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Train: {train_mse:.6f} | Val: {val_mse:.6f} | "
                f"Copy: {copy_mse:.6f} | vs Copy: {improvement:+.1f}% | "
                f"LR: {lr:.2e} | {elapsed / 60:.1f}min {'*' if improved else ''}")

            history["train_mse"].append(train_mse)
            history["val_mse"].append(val_mse)
            history["copy_mse"].append(copy_mse)
            history["improvement_pct"].append(improvement)

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                "model": model.state_dict(),
                "config": {"in_channels": 16, "T_in": T_IN,
                           "hidden": args.hidden, "n_blocks": args.n_blocks},
            }, str(CHECKPOINT_DIR / f"simvp_ep{epoch:03d}.pt"))

    elapsed = time.time() - t_start

    log(f"\n{'='*60}")
    log("SIMVP TRAINING RESULTS")
    log(f"{'='*60}")
    log(f"  Parameters: {n_params:,}")
    log(f"  Best Val MSE: {best_val:.6f}")
    log(f"  vs Copy: {best_copy_improvement:+.1f}%")
    log(f"  Training time: {elapsed / 3600:.1f}h")

    results = {
        "phase": "simvp_training",
        "dataset": "ssv2",
        "n_videos": n_videos,
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "n_params": n_params,
        "architecture": {
            "T_in": T_IN, "hidden": args.hidden,
            "n_blocks": args.n_blocks, "type": "SimVPLatent",
        },
        "best_val_mse": best_val,
        "best_copy_improvement_pct": best_copy_improvement,
        "training_hours": elapsed / 3600,
        "history": history,
        "gpu": torch.cuda.get_device_name(0),
    }
    out_path = RESULTS_DIR / "ssv2_simvp_training.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
