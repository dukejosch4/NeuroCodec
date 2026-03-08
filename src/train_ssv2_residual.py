"""Phase 4: Train ResidualDecoderV2 on SSv2 latent pairs.

Two-phase training:
  Phase A: MSE-only warm-up (5 epochs)
  Phase B: MSE + Spectral fine-tune (15 epochs)

Data modes:
  1. Shard-backed (fast): Uses pre-cached slot pair shards from dynamics training
  2. On-the-fly (slower): Extracts + Hungarian-aligns slots per batch

Usage:
    python src/train_ssv2_residual.py \
        --slot-ckpt checkpoints/ssv2/slot_encoder_best.pt

Output:
    checkpoints/ssv2/residual_mse_best.pt
    checkpoints/ssv2/residual_spectral_best.pt
    results/json/ssv2_residual_training.json
"""

import argparse
import bisect
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

# ── Config ────────────────────────────────────────────────────────────
LATENT_DIR = Path("data/ssv2_latents")
PAIRS_DIR = Path("data/ssv2_slot_pairs")
CHECKPOINT_DIR = Path("checkpoints/ssv2")
RESULTS_DIR = Path("results/json")
LOG_DIR = Path("logs")

BATCH_SIZE = 128
LR_MSE = 3e-4
LR_SPECTRAL = 1e-4
EPOCHS_MSE = 5
EPOCHS_SPECTRAL = 15
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
VAL_FRACTION = 0.15
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Pre-flight ────────────────────────────────────────────────────────
def preflight(args):
    """Validate all dependencies before training."""
    log("Pre-flight checks...")
    errors = []

    if not torch.cuda.is_available():
        errors.append("CUDA not available")
    else:
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"  GPU: {gpu} ({mem:.0f}GB)")

    latent_dir = Path(args.latent_dir)
    if not latent_dir.exists():
        errors.append(f"Latent dir not found: {latent_dir}")
    else:
        n = len(list(latent_dir.glob("*.npy")))
        log(f"  Latent files: {n}")
        if n < 100:
            errors.append(f"Too few latent files: {n}")

    if not os.path.exists(args.slot_ckpt):
        errors.append(f"Slot encoder checkpoint not found: {args.slot_ckpt}")

    _, _, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    log(f"  Disk free: {free_gb:.1f}GB")
    if free_gb < 5:
        errors.append(f"Low disk: {free_gb:.1f}GB free (need >=5GB)")

    try:
        from models import ResidualDecoderV2, SlotLatentAutoencoderV2
        from losses import combined_loss
    except ImportError as e:
        errors.append(f"Import error: {e}. Set PYTHONPATH to include src/")

    if errors:
        log("  PREFLIGHT FAILED:")
        for e in errors:
            log(f"    - {e}")
        return False

    log("  PREFLIGHT PASSED")
    return True


# ── Datasets ──────────────────────────────────────────────────────────
class ShardedPairDataset(Dataset):
    """Memory-mapped slot pair shard."""

    def __init__(self, shard_path):
        self.data = np.load(shard_path, mmap_mode="r")
        self._len = len(self.data)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        pair = self.data[idx].astype(np.float32)
        return torch.from_numpy(pair[0].copy()), torch.from_numpy(pair[1].copy())


class ResidualDatasetShardBacked(Dataset):
    """Latent .npy files + pre-cached slot pair shards."""

    def __init__(self, latent_files, frame_counts, slot_concat_ds):
        self.files = latent_files
        self.slot_data = slot_concat_ds
        self.cumulative = [0]
        for T in frame_counts:
            self.cumulative.append(self.cumulative[-1] + (T - 1))
        self.total = self.cumulative[-1]

        if self.total != len(self.slot_data):
            raise ValueError(
                f"Pair count mismatch: latent files={self.total}, shards={len(self.slot_data)}. "
                f"Use --no-shards to fall back to on-the-fly extraction."
            )

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        vid_idx = bisect.bisect_right(self.cumulative, idx) - 1
        frame_t = idx - self.cumulative[vid_idx]

        latent = np.load(self.files[vid_idx]).astype(np.float32)
        lt = latent[:, frame_t, :, :]
        lt1 = latent[:, frame_t + 1, :, :]

        C, H, W = lt.shape
        lt_tokens = torch.from_numpy(lt.reshape(C, H * W).T.copy())
        lt1_tokens = torch.from_numpy(lt1.reshape(C, H * W).T.copy())
        delta_tokens = lt1_tokens - lt_tokens

        slots_t, slots_t1 = self.slot_data[idx]
        return lt_tokens, delta_tokens, slots_t, slots_t1


class LatentPairDataset(Dataset):
    """Yields (lt_tokens, delta_tokens) for on-the-fly slot extraction."""

    def __init__(self, latent_files, frame_counts):
        self.files = latent_files
        self.cumulative = [0]
        for T in frame_counts:
            self.cumulative.append(self.cumulative[-1] + (T - 1))
        self.total = self.cumulative[-1]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        vid_idx = bisect.bisect_right(self.cumulative, idx) - 1
        frame_t = idx - self.cumulative[vid_idx]

        latent = np.load(self.files[vid_idx]).astype(np.float32)
        lt = latent[:, frame_t, :, :]
        lt1 = latent[:, frame_t + 1, :, :]

        C, H, W = lt.shape
        lt_tokens = torch.from_numpy(lt.reshape(C, H * W).T.copy())
        lt1_tokens = torch.from_numpy(lt1.reshape(C, H * W).T.copy())
        delta_tokens = lt1_tokens - lt_tokens

        return lt_tokens, delta_tokens


# ── Helpers ───────────────────────────────────────────────────────────
def scan_latents(latent_dir, max_videos=None):
    """Scan latent files and count frames. Returns (files, frame_counts, total_pairs)."""
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

    total_pairs = sum(T - 1 for T in frame_counts)
    log(f"  Total: {len(files)} videos, {total_pairs} pairs")
    return [str(f) for f in files], frame_counts, total_pairs


def get_train_val_indices(n_videos, frame_counts, val_fraction=VAL_FRACTION, seed=SEED):
    """Video-level train/val split, consistent across all scripts."""
    indices = list(range(n_videos))
    random.seed(seed)
    random.shuffle(indices)
    n_val = int(n_videos * val_fraction)
    val_video_set = set(indices[:n_val])

    pair_to_video = []
    for vid_idx, T in enumerate(frame_counts):
        for _ in range(T - 1):
            pair_to_video.append(vid_idx)

    train_pairs = [i for i, v in enumerate(pair_to_video) if v not in val_video_set]
    val_pairs = [i for i, v in enumerate(pair_to_video) if v in val_video_set]
    return train_pairs, val_pairs


def hungarian_align_batch(slots_a, slots_b):
    """Batch Hungarian alignment. [B, K, D] -> [B, K, D]."""
    B = slots_a.shape[0]
    aligned = torch.empty_like(slots_b)
    a_cpu = slots_a.cpu()
    b_cpu = slots_b.cpu()
    for b in range(B):
        a_norm = F.normalize(a_cpu[b], dim=-1)
        b_norm = F.normalize(b_cpu[b], dim=-1)
        cost = 1 - torch.mm(a_norm, b_norm.t())
        _, col_ind = linear_sum_assignment(cost.numpy())
        aligned[b] = slots_b[b][col_ind]
    return aligned


# ── Training ──────────────────────────────────────────────────────────
def train_phase(model, optimizer, scheduler, train_dl, val_dl, epochs, beta,
                device, save_path, best_val=float("inf"), slot_encoder=None):
    """Train one phase (MSE-only or MSE+Spectral).

    If slot_encoder is provided, extracts slots on-the-fly (batch items are 2-tuples).
    Otherwise uses pre-cached slots (batch items are 4-tuples).
    """
    from losses import combined_loss
    on_the_fly = slot_encoder is not None
    phase_name = "MSE+Spectral" if beta > 0 else "MSE-only"
    log(f"\n  Training {phase_name} (beta={beta}) for {epochs} epochs...")

    history = {"train_loss": [], "val_loss": [], "val_mse": []}
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0

        for batch in train_dl:
            if on_the_fly:
                lt_tokens = batch[0].to(device)
                delta_tokens = batch[1].to(device)
                lt1_tokens = lt_tokens + delta_tokens
                with torch.no_grad():
                    slots_t = slot_encoder.encode(lt_tokens)
                    slots_t1_raw = slot_encoder.encode(lt1_tokens)
                    slots_t1 = hungarian_align_batch(slots_t, slots_t1_raw)
            else:
                lt_tokens = batch[0].to(device)
                delta_tokens = batch[1].to(device)
                slots_t = batch[2].to(device)
                slots_t1 = batch[3].to(device)

            pred = model(lt_tokens, slots_t, slots_t1)
            loss, _ = combined_loss(pred, delta_tokens, lt_tokens, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            n_batches += 1

        train_loss = ep_loss / n_batches

        # Validate
        if epoch <= 3 or epoch % 2 == 0 or epoch == epochs:
            model.eval()
            val_loss_sum = 0.0
            val_mse_sum = 0.0
            val_n = 0

            with torch.no_grad():
                for batch in val_dl:
                    if on_the_fly:
                        lt_tokens = batch[0].to(device)
                        delta_tokens = batch[1].to(device)
                        lt1_tokens = lt_tokens + delta_tokens
                        slots_t = slot_encoder.encode(lt_tokens)
                        slots_t1_raw = slot_encoder.encode(lt1_tokens)
                        slots_t1 = hungarian_align_batch(slots_t, slots_t1_raw)
                    else:
                        lt_tokens = batch[0].to(device)
                        delta_tokens = batch[1].to(device)
                        slots_t = batch[2].to(device)
                        slots_t1 = batch[3].to(device)

                    pred = model(lt_tokens, slots_t, slots_t1)
                    loss, details = combined_loss(pred, delta_tokens, lt_tokens, beta=beta)
                    val_loss_sum += loss.item()
                    val_mse_sum += details["mse"]
                    val_n += 1

            val_avg = val_loss_sum / val_n
            val_mse_avg = val_mse_sum / val_n

            improved = val_avg < best_val
            if improved:
                best_val = val_avg
                torch.save(model.state_dict(), save_path)

            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            log(f"  Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_avg:.6f} (MSE: {val_mse_avg:.6f}) | "
                f"LR: {lr:.2e} | {elapsed / 60:.1f}min {'*' if improved else ''}")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_avg)
            history["val_mse"].append(val_mse_avg)

        # Periodic checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_path.replace("best", f"ep{epoch:03d}"))

    return best_val, history


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SSv2 Residual Decoder Training")
    parser.add_argument("--slot-ckpt", type=str, required=True,
                        help="Path to trained slot encoder checkpoint")
    parser.add_argument("--latent-dir", type=str, default=str(LATENT_DIR))
    parser.add_argument("--shard-dir", type=str, default=str(PAIRS_DIR))
    parser.add_argument("--no-shards", action="store_true",
                        help="Force on-the-fly slot extraction (slower)")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--epochs-mse", type=int, default=EPOCHS_MSE)
    parser.add_argument("--epochs-spectral", type=int, default=EPOCHS_SPECTRAL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run on 100 videos for validation")
    args = parser.parse_args()

    if args.dry_run:
        args.max_videos = 100
        args.epochs_mse = 2
        args.epochs_spectral = 3

    device = "cuda"
    torch.manual_seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("PHASE 4: RESIDUAL DECODER TRAINING (SSv2)")
    log("=" * 60)

    # ── Pre-flight ──
    if not preflight(args):
        return

    # ── Scan data ──
    log("\n[1/5] Scanning latent files...")
    files, frame_counts, total_pairs = scan_latents(args.latent_dir, args.max_videos)
    n_videos = len(files)

    # ── Train/val split ──
    log("\n[2/5] Building train/val split (seed={}, val={}%)...".format(args.seed, int(VAL_FRACTION * 100)))
    train_indices, val_indices = get_train_val_indices(n_videos, frame_counts, VAL_FRACTION, args.seed)
    log(f"  Train: {len(train_indices)} pairs, Val: {len(val_indices)} pairs")

    # ── Build dataset ──
    log("\n[3/5] Building dataset...")
    shard_dir = Path(args.shard_dir)
    use_shards = (not args.no_shards and shard_dir.exists()
                  and len(list(shard_dir.glob("shard_*.npy"))) > 0)

    if use_shards:
        shard_files = sorted(shard_dir.glob("shard_*.npy"))
        log(f"  Shard-backed mode: {len(shard_files)} shards found")
        shard_datasets = [ShardedPairDataset(s) for s in shard_files]
        slot_concat = ConcatDataset(shard_datasets)

        try:
            full_ds = ResidualDatasetShardBacked(files, frame_counts, slot_concat)
        except ValueError as e:
            log(f"  WARNING: {e}")
            log("  Falling back to on-the-fly mode")
            use_shards = False

    slot_encoder = None
    if use_shards:
        train_ds = Subset(full_ds, train_indices)
        val_ds = Subset(full_ds, val_indices)
        log(f"  Dataset ready (shard-backed, ~{12 * len(train_indices) / 1000:.0f}ms/epoch)")
    else:
        log("  On-the-fly mode: loading slot encoder for live extraction")
        from models import SlotLatentAutoencoderV2
        slot_encoder = SlotLatentAutoencoderV2(
            n_slots=64, slot_dim=128, input_dim=16, n_tokens=1024, n_iter=5
        ).to(device)
        state = torch.load(args.slot_ckpt, map_location=device, weights_only=False)
        slot_encoder.load_state_dict(state)
        slot_encoder.eval()
        log(f"  Slot encoder loaded from {os.path.basename(args.slot_ckpt)}")

        full_ds = LatentPairDataset(files, frame_counts)
        train_ds = Subset(full_ds, train_indices)
        val_ds = Subset(full_ds, val_indices)
        log(f"  Dataset ready (on-the-fly, ~{56 * len(train_indices) / 1000:.0f}ms/epoch est.)")

    num_workers = 4 if use_shards else 0
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # ── Model ──
    log("\n[4/5] Creating ResidualDecoderV2...")
    from models import ResidualDecoderV2
    model = ResidualDecoderV2().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Parameters: {n_params:,}")

    # ── Phase A: MSE-only ──
    log("\n[5/5] Training...")
    log(f"\n{'='*60}")
    log("PHASE A: MSE-only warm-up")
    log(f"{'='*60}")

    optimizer_a = torch.optim.AdamW(model.parameters(), lr=LR_MSE, weight_decay=0.01)
    total_steps_a = args.epochs_mse * len(train_dl)

    def lr_lambda_a(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps_a - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler_a = torch.optim.lr_scheduler.LambdaLR(optimizer_a, lr_lambda_a)
    mse_path = str(CHECKPOINT_DIR / "residual_mse_best.pt")

    best_a, history_a = train_phase(
        model, optimizer_a, scheduler_a, train_dl, val_dl,
        epochs=args.epochs_mse, beta=0.0, device=device,
        save_path=mse_path, slot_encoder=slot_encoder,
    )
    log(f"  Phase A best val: {best_a:.6f}")

    # ── Phase B: MSE + Spectral ──
    log(f"\n{'='*60}")
    log("PHASE B: MSE + Spectral fine-tune")
    log(f"{'='*60}")

    # Reload best MSE checkpoint
    state = torch.load(mse_path, map_location=device, weights_only=False)
    model.load_state_dict(state)

    optimizer_b = torch.optim.AdamW(model.parameters(), lr=LR_SPECTRAL, weight_decay=0.01)
    total_steps_b = args.epochs_spectral * len(train_dl)

    def lr_lambda_b(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps_b - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler_b = torch.optim.lr_scheduler.LambdaLR(optimizer_b, lr_lambda_b)
    spectral_path = str(CHECKPOINT_DIR / "residual_spectral_best.pt")

    best_b, history_b = train_phase(
        model, optimizer_b, scheduler_b, train_dl, val_dl,
        epochs=args.epochs_spectral, beta=0.01, device=device,
        save_path=spectral_path, slot_encoder=slot_encoder,
    )
    log(f"  Phase B best val: {best_b:.6f}")

    # ── Save results ──
    results = {
        "phase": "residual_training",
        "dataset": "ssv2",
        "n_videos": n_videos,
        "n_train_pairs": len(train_indices),
        "n_val_pairs": len(val_indices),
        "n_params": n_params,
        "mode": "shard-backed" if use_shards else "on-the-fly",
        "phase_a": {"epochs": args.epochs_mse, "best_val": best_a, "history": history_a},
        "phase_b": {"epochs": args.epochs_spectral, "best_val": best_b, "history": history_b},
        "config": {
            "lr_mse": LR_MSE, "lr_spectral": LR_SPECTRAL,
            "batch_size": args.batch_size, "seed": args.seed,
        },
        "gpu": torch.cuda.get_device_name(0),
    }
    out_path = RESULTS_DIR / "ssv2_residual_training.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")
    log("Done.")


if __name__ == "__main__":
    main()
