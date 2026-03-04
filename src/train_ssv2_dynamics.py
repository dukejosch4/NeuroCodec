"""Phase 3: Extract aligned slots + Train DynamicsTransformer on SSv2.

Step 1: Extract slots from all latents using trained encoder (streamed to shards)
Step 2: Hungarian-align temporal slot sequences
Step 3: Train DynamicsTransformer on (S_t, S_{t+1}) pairs
Step 4: Evaluate dynamics vs copy baseline

Uses sharded extraction to avoid RAM bottleneck (~169GB if all pairs in memory).

Usage:
    python src/train_ssv2_dynamics.py --slot-ckpt checkpoints/ssv2/slot_encoder_best.pt
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset, ConcatDataset

LATENT_DIR = Path("data/ssv2_latents")
CHECKPOINT_DIR = Path("checkpoints/ssv2")
RESULTS_DIR = Path("results/json")
PAIRS_DIR = Path("data/ssv2_slot_pairs")

# Architecture (identical to UCF-101)
N_SLOTS = 64
SLOT_DIM = 128
DYN_HEADS = 4
DYN_LAYERS = 2

# Training
BATCH_SIZE = 256
LR = 3e-4
EPOCHS = 200
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
VAL_FRACTION = 0.15

# Extraction
SHARD_SIZE = 50000  # pairs per shard file (~3.1GB each)


def hungarian_align(slots_a, slots_b):
    """Align slots_b to match slots_a ordering via Hungarian matching."""
    a_norm = F.normalize(slots_a, dim=-1)
    b_norm = F.normalize(slots_b, dim=-1)
    cost = 1 - torch.mm(a_norm, b_norm.t())
    _, col_ind = linear_sum_assignment(cost.cpu().numpy())
    return slots_b[col_ind]


def extract_and_save_shards(slot_encoder, latent_dir, pairs_dir, device, max_videos=None):
    """Extract slots, Hungarian-align, and save to sharded .npy files.

    Each shard: [N, 2, 64, 128] float16 — ~390MB per 50K pairs.
    """
    pairs_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(latent_dir.glob("*.npy"))
    if max_videos:
        files = files[:max_videos]

    print(f"  Extracting slots from {len(files)} videos...")
    current_shard = []
    shard_idx = 0
    total_pairs = 0

    slot_encoder.eval()
    with torch.no_grad():
        for i, f in enumerate(files):
            latent = np.load(f).astype(np.float32)  # [16, T, 32, 32]
            C, T, H, W = latent.shape

            # Get per-frame tokens
            frames = np.transpose(latent, (1, 0, 2, 3))  # [T, 16, 32, 32]
            tokens = frames.reshape(T, C, H * W).transpose(0, 2, 1)  # [T, 1024, 16]

            # Encode all frames through slot attention
            slots_list = []
            for t in range(T):
                tok = torch.from_numpy(tokens[t:t+1]).to(device)
                s = slot_encoder.encode(tok)  # [1, 64, 128]
                slots_list.append(s.cpu())

            # Hungarian-align consecutive frames
            for t in range(T - 1):
                s_t = slots_list[t][0]       # [64, 128]
                s_t1 = slots_list[t + 1][0]  # [64, 128]
                s_t1_aligned = hungarian_align(s_t, s_t1)
                pair = torch.stack([s_t, s_t1_aligned]).numpy()  # [2, 64, 128]
                current_shard.append(pair)
                total_pairs += 1

                # Flush shard when full
                if len(current_shard) >= SHARD_SIZE:
                    shard_path = pairs_dir / f"shard_{shard_idx:04d}.npy"
                    np.save(shard_path, np.stack(current_shard).astype(np.float16))
                    print(f"    Saved {shard_path.name}: {len(current_shard)} pairs", flush=True)
                    current_shard = []
                    shard_idx += 1

            if (i + 1) % 5000 == 0:
                print(f"    {i + 1}/{len(files)} videos, "
                      f"{total_pairs} pairs, {shard_idx} shards", flush=True)

    # Save remaining
    if current_shard:
        shard_path = pairs_dir / f"shard_{shard_idx:04d}.npy"
        np.save(shard_path, np.stack(current_shard).astype(np.float16))
        print(f"    Saved {shard_path.name}: {len(current_shard)} pairs")
        shard_idx += 1

    print(f"  Total: {len(files)} videos, {total_pairs} pairs, {shard_idx} shards")
    return total_pairs


class ShardedPairDataset(Dataset):
    """Memory-mapped dataset over sharded .npy pair files.

    Uses mmap to avoid loading entire shard into RAM.
    Only reads the accessed pair on __getitem__.
    """

    def __init__(self, shard_path):
        self.data = np.load(shard_path, mmap_mode="r")  # [N, 2, 64, 128] mmap'd
        self._len = len(self.data)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        pair = self.data[idx].astype(np.float32)  # [2, 64, 128], copy from mmap
        return torch.from_numpy(pair[0].copy()), torch.from_numpy(pair[1].copy())


def train_dynamics(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3: DYNAMICS TRANSFORMER TRAINING (SSv2)")
    print("=" * 60)

    # Load slot encoder
    print("\n[1/4] Loading slot encoder...")
    from models import SlotLatentAutoencoderV2
    slot_encoder = SlotLatentAutoencoderV2(
        n_slots=N_SLOTS, slot_dim=SLOT_DIM,
        input_dim=16, n_tokens=1024, n_iter=5,
    ).to(device)
    state = torch.load(args.slot_ckpt, map_location=device)
    slot_encoder.load_state_dict(state)
    slot_encoder.eval()
    print(f"  Loaded from {args.slot_ckpt}")

    # Extract and align slots (sharded)
    print("\n[2/4] Extracting and aligning slots...")
    existing_shards = sorted(PAIRS_DIR.glob("shard_*.npy")) if PAIRS_DIR.exists() else []

    if existing_shards and not args.force_extract:
        print(f"  Found {len(existing_shards)} cached shards in {PAIRS_DIR}")
        total_pairs = sum(len(np.load(s, mmap_mode="r")) for s in existing_shards)
        print(f"  Total cached pairs: {total_pairs}")
    else:
        total_pairs = extract_and_save_shards(
            slot_encoder, LATENT_DIR, PAIRS_DIR, device, max_videos=args.max_videos
        )
        existing_shards = sorted(PAIRS_DIR.glob("shard_*.npy"))

    # Split into train/val
    if len(existing_shards) >= 7:
        # Enough shards: split at shard level
        n_val_shards = max(1, int(len(existing_shards) * VAL_FRACTION))
        val_shards = existing_shards[:n_val_shards]
        train_shards = existing_shards[n_val_shards:]
        print(f"  Train shards: {len(train_shards)}, Val shards: {len(val_shards)}")
        train_ds = ConcatDataset([ShardedPairDataset(s) for s in train_shards])
        val_ds = ConcatDataset([ShardedPairDataset(s) for s in val_shards])
    else:
        # Few shards: load all and split at pair level
        all_ds = ConcatDataset([ShardedPairDataset(s) for s in existing_shards])
        n_val = max(1, int(len(all_ds) * VAL_FRACTION))
        n_train = len(all_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(all_ds, [n_train, n_val])

    print(f"  Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    # Create dynamics model
    print("\n[3/4] Creating DynamicsTransformer...")
    from models import DynamicsTransformer
    model = DynamicsTransformer(
        n_tokens=N_SLOTS, token_dim=SLOT_DIM,
        d_model=SLOT_DIM, n_heads=DYN_HEADS, n_layers=DYN_LAYERS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(train_dl)

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Train
    print(f"\n[4/4] Training for {EPOCHS} epochs...")
    best_val_mse = float("inf")
    history = {"train_mse": [], "val_mse": [], "dynamics_vs_copy": []}
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for s_t_batch, s_t1_batch in train_dl:
            s_t_batch = s_t_batch.to(device)
            s_t1_batch = s_t1_batch.to(device)

            pred = model(s_t_batch)
            loss = F.mse_loss(pred, s_t1_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()

        train_mse = ep_loss / len(train_dl)

        # Validate
        if epoch <= 10 or epoch % 10 == 0:
            model.eval()
            val_loss = 0.0
            copy_loss = 0.0
            n_batches = 0

            with torch.no_grad():
                for s_t_batch, s_t1_batch in val_dl:
                    s_t_batch = s_t_batch.to(device)
                    s_t1_batch = s_t1_batch.to(device)

                    pred = model(s_t_batch)
                    val_loss += F.mse_loss(pred, s_t1_batch).item()
                    copy_loss += F.mse_loss(s_t_batch, s_t1_batch).item()
                    n_batches += 1

            val_mse = val_loss / n_batches
            copy_mse = copy_loss / n_batches
            improvement = (copy_mse - val_mse) / copy_mse * 100

            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train: {train_mse:.6f} | Val: {val_mse:.6f} | "
                  f"Copy: {copy_mse:.6f} | Dyn vs Copy: {improvement:+.1f}% | "
                  f"{elapsed/60:.1f}min", flush=True)

            history["train_mse"].append(train_mse)
            history["val_mse"].append(val_mse)
            history["dynamics_vs_copy"].append(improvement)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                torch.save(model.state_dict(), CHECKPOINT_DIR / "dynamics_best.pt")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"dynamics_ep{epoch:03d}.pt")

    # Final results
    elapsed = time.time() - t_start
    final_improvement = history["dynamics_vs_copy"][-1] if history["dynamics_vs_copy"] else 0

    print("\n" + "=" * 60)
    print("DYNAMICS TRAINING RESULTS")
    print("=" * 60)
    print(f"  Pairs: {len(train_ds) + len(val_ds)}")
    print(f"  Best Val MSE: {best_val_mse:.6f}")
    print(f"  Dynamics vs Copy: {final_improvement:+.1f}%")
    print(f"  Training time: {elapsed/3600:.1f}h")

    if final_improvement > 25:
        print(f"  P-005 PASSED: {final_improvement:.1f}% > 25%")
    elif final_improvement > 15:
        print(f"  P-005 PARTIAL: {final_improvement:.1f}% (UCF-101 level)")
    else:
        print(f"  P-005 FAILED: {final_improvement:.1f}% < 15%")

    results = {
        "phase": "dynamics_training",
        "dataset": "ssv2",
        "n_pairs": len(train_ds) + len(val_ds),
        "epochs": EPOCHS,
        "best_val_mse": best_val_mse,
        "dynamics_vs_copy_pct": final_improvement,
        "training_hours": elapsed / 3600,
        "n_params": n_params,
        "history": history,
    }
    with open(RESULTS_DIR / "ssv2_dynamics_training.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {RESULTS_DIR / 'ssv2_dynamics_training.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot-ckpt", type=str, required=True,
                        help="Path to trained slot encoder checkpoint")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--force-extract", action="store_true",
                        help="Re-extract slots even if cached")
    args = parser.parse_args()
    train_dynamics(args)
