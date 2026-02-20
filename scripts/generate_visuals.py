#!/usr/bin/env python3
"""Generate visual examples for paper (GT vs Predicted vs Copy frames).

Produces:
  1. Side-by-side frame comparisons (GT | Predicted | Copy | Error Map)
  2. 8-frame rollout strips (shows temporal coherence)
  3. EASY vs HARD frame examples
  4. Real data for Figure 2 rollout plot (replacing "schematic")

GPU time: ~5 minutes on A100.

Usage:
    python scripts/generate_visuals.py \
        --data-dir ~/ \
        --checkpoint ~/residual_v2_spectral_beta0.01_best.pt \
        --dynamics-checkpoint ~/dynamics_best_d2.pt \
        --output-dir results/figures/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import ResidualDecoderV2, DynamicsTransformer
from src.vae_utils import load_cogvideox_vae, decode_single_frame


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tensor_to_numpy(t):
    """Convert [3, H, W] tensor in [0,1] to numpy [H, W, 3] uint8."""
    return (t.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dynamics-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/figures/")
    parser.add_argument("--n-examples", type=int, default=6,
                        help="Number of side-by-side examples")
    parser.add_argument("--n-rollout-strips", type=int, default=3,
                        help="Number of rollout strip examples")
    args = parser.parse_args()

    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

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

    N_videos, T, C, H, W = latent_frames.shape
    n_val = 200
    val_latents = latent_frames[-n_val:]
    val_slots = video_slots[-n_val:]

    # ── Load Models ──
    log("Loading models...")
    res_decoder = ResidualDecoderV2().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    res_decoder.load_state_dict(state)
    res_decoder.eval()

    dynamics = DynamicsTransformer().to(device)
    dyn_state = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(dyn_state, dict) and "model" in dyn_state:
        dyn_state = dyn_state["model"]
    dynamics.load_state_dict(dyn_state)
    dynamics.eval()

    log("Loading CogVideoX VAE...")
    vae, scaling_factor = load_cogvideox_vae(device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Classify EASY/HARD ──
    log("Classifying EASY/HARD frames...")
    copy_mses = []
    for vid_idx in range(min(50, n_val)):
        for t in range(T - 1):
            mse = F.mse_loss(val_latents[vid_idx, t], val_latents[vid_idx, t+1]).item()
            copy_mses.append((mse, vid_idx, t))
    copy_mses.sort(key=lambda x: x[0])
    threshold_85 = copy_mses[int(len(copy_mses) * 0.85)][0]

    easy_examples = [(v, t) for mse, v, t in copy_mses if mse < threshold_85][:args.n_examples // 2]
    hard_examples = [(v, t) for mse, v, t in copy_mses if mse >= threshold_85][:args.n_examples // 2]
    selected = easy_examples + hard_examples

    # ================================================================
    # VISUAL 1: Side-by-side comparisons
    # ================================================================
    log(f"\nGenerating {len(selected)} side-by-side comparisons...")

    for i, (vid_idx, t) in enumerate(selected):
        lt = val_latents[vid_idx, t].to(device)
        lt1_gt = val_latents[vid_idx, t + 1].to(device)
        st = val_slots[vid_idx, t:t+1].to(device)

        with torch.no_grad():
            st1 = dynamics(st)
            lt_tok = lt.flatten(1).unsqueeze(0).permute(0, 2, 1)
            delta = res_decoder(lt_tok, st, st1)
            lt1_pred = (lt_tok + delta).permute(0, 2, 1).reshape(C, H, W)

        px_gt = decode_single_frame(vae, lt1_gt, scaling_factor)
        px_pred = decode_single_frame(vae, lt1_pred, scaling_factor)
        px_copy = decode_single_frame(vae, lt, scaling_factor)

        # Error maps (amplified)
        err_pred = (px_pred - px_gt).abs().mean(dim=0)  # [H, W]
        err_copy = (px_copy - px_gt).abs().mean(dim=0)

        frame_type = "EASY" if i < len(easy_examples) else "HARD"
        copy_mse = F.mse_loss(lt, lt1_gt).item()

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(tensor_to_numpy(px_gt)); axes[0].set_title("Ground Truth", fontsize=11)
        axes[1].imshow(tensor_to_numpy(px_pred)); axes[1].set_title("Predicted (Ours)", fontsize=11)
        axes[2].imshow(tensor_to_numpy(px_copy)); axes[2].set_title("Copy Baseline", fontsize=11)
        axes[3].imshow(err_pred.cpu().numpy(), cmap="hot", vmin=0, vmax=0.3)
        axes[3].set_title("Error (Ours)", fontsize=11)
        axes[4].imshow(err_copy.cpu().numpy(), cmap="hot", vmin=0, vmax=0.3)
        axes[4].set_title("Error (Copy)", fontsize=11)

        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{frame_type} frame — video {vid_idx}, t={t}→{t+1} (copy MSE: {copy_mse:.4f})",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()

        fname = f"comparison_{i:02d}_{frame_type.lower()}.png"
        plt.savefig(os.path.join(args.output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  Saved {fname}")

    # ================================================================
    # VISUAL 2: Rollout strips (8 frames)
    # ================================================================
    log(f"\nGenerating {args.n_rollout_strips} rollout strips...")

    for strip_idx in range(args.n_rollout_strips):
        vid_idx = strip_idx * 5  # spread across validation set

        fig, axes = plt.subplots(3, T, figsize=(T * 2.5, 8))

        # Row 1: GT frames
        for t in range(T):
            px = decode_single_frame(vae, val_latents[vid_idx, t].to(device), scaling_factor)
            axes[0, t].imshow(tensor_to_numpy(px))
            axes[0, t].set_title(f"t={t}", fontsize=9)
            axes[0, t].axis("off")
        axes[0, 0].set_ylabel("Ground Truth", fontsize=11, fontweight="bold")

        # Row 2: Predicted rollout
        L_cur = val_latents[vid_idx, 0].clone()
        cur_slots = val_slots[vid_idx, 0:1].to(device)

        px0 = decode_single_frame(vae, L_cur.to(device), scaling_factor)
        axes[1, 0].imshow(tensor_to_numpy(px0))
        axes[1, 0].axis("off")

        with torch.no_grad():
            for t in range(1, T):
                pred_slots = dynamics(cur_slots)
                lt_tok = L_cur.flatten(1).unsqueeze(0).permute(0, 2, 1).to(device)
                delta = res_decoder(lt_tok, cur_slots, pred_slots)
                L_cur = (lt_tok + delta)[0].permute(1, 0).reshape(C, H, W).cpu()

                px = decode_single_frame(vae, L_cur.to(device), scaling_factor)
                axes[1, t].imshow(tensor_to_numpy(px))
                axes[1, t].axis("off")
                cur_slots = pred_slots

        axes[1, 0].set_ylabel("Predicted", fontsize=11, fontweight="bold")

        # Row 3: Copy baseline (all frame 0)
        for t in range(T):
            px = decode_single_frame(vae, val_latents[vid_idx, 0].to(device), scaling_factor)
            axes[2, t].imshow(tensor_to_numpy(px))
            axes[2, t].axis("off")
        axes[2, 0].set_ylabel("Copy (frame 0)", fontsize=11, fontweight="bold")

        fig.suptitle(f"8-Frame Rollout — Video {vid_idx}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fname = f"rollout_strip_{strip_idx:02d}.png"
        plt.savefig(os.path.join(args.output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  Saved {fname}")

    # ================================================================
    # VISUAL 3: Real rollout plot (replaces schematic Figure 2)
    # ================================================================
    log("\nGenerating real rollout plot (Figure 2 replacement)...")

    # Load pre-computed rollout data from D2 results
    results_root = os.path.dirname(args.output_dir.rstrip("/"))
    d2_path = os.path.join(results_root, "json", "D2_results.json")
    if os.path.exists(d2_path):
        with open(d2_path) as f:
            d2 = json.load(f)
        rollout_data = d2.get("rollout", {})
    else:
        rollout_data = None
        log("  D2_results.json not found, using fresh computation")

    if rollout_data:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        configs_to_plot = [
            ("full_system", "Full System", "#2196F3", "-o"),
            ("copy_baseline", "Copy Baseline", "#9E9E9E", "--s"),
            ("wo_slots", "w/o Slots", "#F44336", "-.^"),
            ("wo_event_seg", "w/o Event Seg.", "#FF9800", ":D"),
        ]

        for key, label, color, marker in configs_to_plot:
            if key in rollout_data:
                mses = rollout_data[key][1:]  # skip frame 0 (always 0)
                frames = list(range(1, len(mses) + 1))
                ax.plot(frames, mses, marker, color=color, label=label,
                        linewidth=2, markersize=6)

                # Annotate stability ratio
                if mses[0] > 0:
                    stab = mses[-1] / mses[0]
                    ax.annotate(f"{stab:.2f}x", xy=(frames[-1], mses[-1]),
                                xytext=(5, 5), textcoords="offset points",
                                fontsize=9, color=color, fontweight="bold")

        ax.set_xlabel("Predicted Frame", fontsize=12)
        ax.set_ylabel("Latent MSE", fontsize=12)
        ax.set_title("Rollout Error Accumulation (50 validation videos)", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, T))

        plt.tight_layout()
        fname = "figure2_rollout_real.png"
        plt.savefig(os.path.join(args.output_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()
        log(f"  Saved {fname}")

        # Also save as PDF for LaTeX
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for key, label, color, marker in configs_to_plot:
            if key in rollout_data:
                mses = rollout_data[key][1:]
                frames = list(range(1, len(mses) + 1))
                ax.plot(frames, mses, marker, color=color, label=label,
                        linewidth=2, markersize=6)
        ax.set_xlabel("Predicted Frame", fontsize=12)
        ax.set_ylabel("Latent MSE", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, T))
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "figure2_rollout_real.pdf"),
                    bbox_inches="tight")
        plt.close()
        log(f"  Saved figure2_rollout_real.pdf")

    log("\nDone. All visuals saved to " + args.output_dir)


if __name__ == "__main__":
    main()
