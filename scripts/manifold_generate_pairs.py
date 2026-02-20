#!/usr/bin/env python3
"""Generate (z_pred, z_gt) training pairs for ManifoldProjector.

Runs the trained ResidualDecoderV2 + DynamicsTransformer on all video pairs
to produce off-manifold predictions paired with ground-truth targets.

Output: manifold_pairs.pt containing {z_pred, z_gt, split_idx} in spatial format.

GPU time: ~2-3 minutes on A100.

Usage:
    python scripts/manifold_generate_pairs.py \
        --data-dir ~/ \
        --checkpoint ~/residual_v2_spectral_beta0.01_best.pt \
        --dynamics-checkpoint ~/dynamics_best_d2.pt \
        --output ~/manifold_pairs.pt
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import ResidualDecoderV2, DynamicsTransformer


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dynamics-checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="manifold_pairs.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available()
    log(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load Data ──
    log("Loading data...")
    latents_raw = torch.load(
        os.path.join(args.data_dir, "latents_2000videos.pt"),
        map_location="cpu", weights_only=False,
    )
    video_slots = torch.load(
        os.path.join(args.data_dir, "aligned_video_slots.pt"),
        map_location="cpu", weights_only=False,
    )

    if latents_raw.shape[1] == 16 and latents_raw.shape[2] == 9:
        latent_frames = latents_raw.permute(0, 2, 1, 3, 4)
    else:
        latent_frames = latents_raw
    del latents_raw

    N, T, C, H, W = latent_frames.shape
    n_pairs = N * (T - 1)
    log(f"Data: {N} videos, {T} frames -> {n_pairs} pairs")

    # Flatten to frame pairs
    latent_t = latent_frames[:, :-1].reshape(n_pairs, C, H, W)    # [n, 16, 32, 32]
    latent_t1 = latent_frames[:, 1:].reshape(n_pairs, C, H, W)    # [n, 16, 32, 32] = z_gt
    slots_t = video_slots[:, :-1].reshape(n_pairs, 64, 128)
    slots_t1 = video_slots[:, 1:].reshape(n_pairs, 64, 128)
    del latent_frames, video_slots

    # Convert to token format for ResidualDecoder
    latent_tokens_t = latent_t.flatten(2).permute(0, 2, 1)  # [n, 1024, 16]

    # ── Load Models ──
    log("Loading models...")
    res_decoder = ResidualDecoderV2().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    res_decoder.load_state_dict(state)
    res_decoder.eval()
    log("  ResidualDecoderV2 loaded")

    dynamics = DynamicsTransformer().to(device)
    dyn_state = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(dyn_state, dict) and "model" in dyn_state:
        dyn_state = dyn_state["model"]
    dynamics.load_state_dict(dyn_state)
    dynamics.eval()
    log("  DynamicsTransformer loaded")

    # ── Generate Predictions ──
    log("Generating predictions...")
    z_pred_all = torch.empty_like(latent_t1)  # [n, 16, 32, 32]
    mse_sum = 0.0
    n_batches = 0
    t_start = time.time()

    with torch.no_grad():
        for i in range(0, n_pairs, args.batch_size):
            j = min(i + args.batch_size, n_pairs)
            lt = latent_tokens_t[i:j].to(device)
            st = slots_t[i:j].to(device)

            # Use dynamics-predicted slots (realistic inference scenario)
            st1_dyn = dynamics(st)

            delta = res_decoder(lt, st, st1_dyn)
            z_pred = (lt + delta).permute(0, 2, 1).reshape(-1, C, H, W)
            z_pred_all[i:j] = z_pred.cpu()

            # Track MSE for sanity
            z_gt_batch = latent_t1[i:j].to(device)
            mse_sum += F.mse_loss(z_pred, z_gt_batch).item()
            n_batches += 1

            if (i // args.batch_size) % 50 == 0:
                elapsed = time.time() - t_start
                pct = j / n_pairs * 100
                log(f"  {j}/{n_pairs} ({pct:.0f}%) — {elapsed:.1f}s")

    elapsed = time.time() - t_start
    avg_mse = mse_sum / n_batches
    log(f"Done in {elapsed:.1f}s. Average MSE: {avg_mse:.6f}")

    # ── Compute Error Statistics ──
    residual = z_pred_all - latent_t1
    log(f"\nError statistics (z_pred - z_gt):")
    log(f"  Mean:  {residual.mean():.6f}")
    log(f"  Std:   {residual.std():.6f}")
    log(f"  |Max|: {residual.abs().max():.6f}")

    per_channel_std = residual.std(dim=(0, 2, 3))
    log(f"  Per-channel std: {per_channel_std.tolist()}")

    # ── Save ──
    # Train/val split: same as original training (1800 train / 200 val videos)
    n_val_videos = 200
    n_train_videos = N - n_val_videos
    split_idx = n_train_videos * (T - 1)

    output = {
        "z_pred": z_pred_all,           # [n_pairs, 16, 32, 32] float32
        "z_gt": latent_t1,              # [n_pairs, 16, 32, 32] float32
        "split_idx": split_idx,         # train pairs = [:split_idx], val = [split_idx:]
        "n_videos": N,
        "n_frames": T,
        "avg_mse": avg_mse,
        "error_std": residual.std().item(),
        "gpu": torch.cuda.get_device_name(0),
    }

    torch.save(output, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    log(f"\nSaved: {args.output} ({size_mb:.0f} MB)")
    log(f"  Train pairs: {split_idx}")
    log(f"  Val pairs:   {n_pairs - split_idx}")


if __name__ == "__main__":
    main()
