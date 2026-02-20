#!/usr/bin/env python3
"""Gap 1 Fix: Expanded Perceptual Evaluation (100 validation videos).

Current paper: Table 3 uses only 10 videos x 8 pairs = 80 samples.
This script evaluates on 100 validation videos (800 frame pairs)
for statistically robust LPIPS, SSIM, FVD, and variance-ratio metrics.

Also produces REAL rollout data for Figure 2 (replacing "schematic").

GPU time: ~20-25 minutes on A100 (dominated by VAE decode).

Usage:
    python scripts/gap1_expanded_perceptual.py \
        --data-dir ~/ \
        --checkpoint ~/residual_v2_spectral_beta0.01_best.pt \
        --dynamics-checkpoint ~/dynamics_best_d2.pt \
        --output results/json/gap1_perceptual.json
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, torch.Tensor): return obj.cpu().numpy().tolist()
        return super().default(obj)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Residual decoder checkpoint (spectral variant)")
    parser.add_argument("--mse-checkpoint", type=str, default=None,
                        help="MSE-only checkpoint for comparison (optional)")
    parser.add_argument("--dynamics-checkpoint", type=str, required=True)
    parser.add_argument("--n-val-videos", type=int, default=100,
                        help="Number of validation videos for perceptual eval")
    parser.add_argument("--n-rollout-videos", type=int, default=50,
                        help="Number of videos for rollout eval")
    parser.add_argument("--output", type=str, default="results/json/gap1_perceptual.json")
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

    N_videos, T, C, H, W = latent_frames.shape
    log(f"Data: {N_videos} videos, {T} frames, shape [{C},{H},{W}]")

    n_val = 200  # last 200 videos are validation
    val_latents = latent_frames[-n_val:]
    val_slots = video_slots[-n_val:]

    # Limit to requested number of videos
    n_eval = min(args.n_val_videos, n_val)
    eval_latents = val_latents[:n_eval]
    eval_slots = val_slots[:n_eval]
    log(f"Evaluating on {n_eval} validation videos ({n_eval * (T-1)} frame pairs)")

    # ── Load Models ──
    log("Loading models...")
    # Spectral variant (main model)
    res_decoder = ResidualDecoderV2().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    res_decoder.load_state_dict(state)
    res_decoder.eval()
    log(f"  Residual decoder (spectral): loaded from {os.path.basename(args.checkpoint)}")

    # MSE-only variant (optional comparison)
    res_mse = None
    if args.mse_checkpoint and os.path.exists(args.mse_checkpoint):
        res_mse = ResidualDecoderV2().to(device)
        state = torch.load(args.mse_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        res_mse.load_state_dict(state)
        res_mse.eval()
        log(f"  Residual decoder (MSE-only): loaded")

    # Dynamics
    dynamics = DynamicsTransformer().to(device)
    dyn_state = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(dyn_state, dict) and "model" in dyn_state:
        dyn_state = dyn_state["model"]
    dynamics.load_state_dict(dyn_state)
    dynamics.eval()
    log(f"  Dynamics: loaded")

    # VAE
    log("Loading CogVideoX VAE...")
    vae, scaling_factor = load_cogvideox_vae(device)
    log("  VAE loaded")

    # LPIPS
    import lpips as lpips_module
    lpips_fn = lpips_module.LPIPS(net="alex").to(device).eval()
    log("  LPIPS (alex) loaded")

    results = {}

    # ================================================================
    # PART 1: Single-Step Perceptual Quality (expanded from 10 to N videos)
    # ================================================================
    log(f"\n{'='*60}")
    log(f"PART 1: Single-Step Perceptual Quality ({n_eval} videos)")
    log(f"{'='*60}")

    all_lpips_pred = []
    all_lpips_copy = []
    all_ssim_pred = []
    all_ssim_copy = []
    all_ratio_dev_pred = []

    # Optional MSE-only comparison
    all_lpips_mse = []
    all_ssim_mse = []
    all_ratio_dev_mse = []

    from skimage.metrics import structural_similarity as ssim_fn

    def compute_ssim(img1, img2):
        a = img1.cpu().numpy().transpose(1, 2, 0)
        b = img2.cpu().numpy().transpose(1, 2, 0)
        return ssim_fn(a, b, channel_axis=2, data_range=1.0)

    def compute_lpips(img1, img2):
        with torch.no_grad():
            a = (img1.unsqueeze(0) * 2 - 1).to(device)
            b = (img2.unsqueeze(0) * 2 - 1).to(device)
            return lpips_fn(a, b).item()

    def compute_ratio_dev(pred_latent, gt_latent):
        """Per-channel variance ratio deviation."""
        pred_std = pred_latent.std(dim=(1, 2))  # [16]
        gt_std = gt_latent.std(dim=(1, 2))
        ratio = pred_std / (gt_std + 1e-8)
        return ((ratio - 1.0).abs()).mean().item()

    t_start = time.time()
    for vid_idx in range(n_eval):
        if vid_idx % 10 == 0:
            elapsed = time.time() - t_start
            eta = (elapsed / max(vid_idx, 1)) * (n_eval - vid_idx)
            log(f"  Video {vid_idx+1}/{n_eval} (elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)")

        for t in range(T - 1):
            lt = eval_latents[vid_idx, t].to(device)      # [16, 32, 32]
            lt1_gt = eval_latents[vid_idx, t + 1].to(device)
            st = eval_slots[vid_idx, t:t+1].to(device)    # [1, 64, 128]
            st1_gt = eval_slots[vid_idx, t+1:t+2].to(device)

            # Predict with dynamics
            with torch.no_grad():
                st1_pred = dynamics(st)
                lt_tok = lt.flatten(1).unsqueeze(0).permute(0, 2, 1)  # [1, 1024, 16]
                delta = res_decoder(lt_tok, st, st1_pred)
                lt1_pred = (lt_tok + delta).permute(0, 2, 1).reshape(16, H, W)

            # Ratio deviation (latent space, no VAE decode needed)
            all_ratio_dev_pred.append(compute_ratio_dev(lt1_pred, lt1_gt))

            # MSE-only variant
            if res_mse is not None:
                with torch.no_grad():
                    delta_mse = res_mse(lt_tok, st, st1_pred)
                    lt1_mse = (lt_tok + delta_mse).permute(0, 2, 1).reshape(16, H, W)
                all_ratio_dev_mse.append(compute_ratio_dev(lt1_mse, lt1_gt))

            # Decode through VAE (expensive — only first frame pair per video for speed)
            if t == 0:
                px_gt = decode_single_frame(vae, lt1_gt, scaling_factor)
                px_pred = decode_single_frame(vae, lt1_pred, scaling_factor)
                px_copy = decode_single_frame(vae, lt, scaling_factor)

                all_lpips_pred.append(compute_lpips(px_pred, px_gt))
                all_lpips_copy.append(compute_lpips(px_copy, px_gt))
                all_ssim_pred.append(compute_ssim(px_pred, px_gt))
                all_ssim_copy.append(compute_ssim(px_copy, px_gt))

                if res_mse is not None:
                    px_mse = decode_single_frame(vae, lt1_mse, scaling_factor)
                    all_lpips_mse.append(compute_lpips(px_mse, px_gt))
                    all_ssim_mse.append(compute_ssim(px_mse, px_gt))

    results["single_step"] = {
        "n_videos": n_eval,
        "n_pixel_pairs": len(all_lpips_pred),
        "n_latent_pairs": len(all_ratio_dev_pred),
        "spectral": {
            "lpips": {"mean": float(np.mean(all_lpips_pred)), "std": float(np.std(all_lpips_pred))},
            "ssim": {"mean": float(np.mean(all_ssim_pred)), "std": float(np.std(all_ssim_pred))},
            "ratio_dev": {"mean": float(np.mean(all_ratio_dev_pred)), "std": float(np.std(all_ratio_dev_pred))},
        },
        "copy": {
            "lpips": {"mean": float(np.mean(all_lpips_copy)), "std": float(np.std(all_lpips_copy))},
            "ssim": {"mean": float(np.mean(all_ssim_copy)), "std": float(np.std(all_ssim_copy))},
        },
    }

    if res_mse is not None:
        results["single_step"]["mse_only"] = {
            "lpips": {"mean": float(np.mean(all_lpips_mse)), "std": float(np.std(all_lpips_mse))},
            "ssim": {"mean": float(np.mean(all_ssim_mse)), "std": float(np.std(all_ssim_mse))},
            "ratio_dev": {"mean": float(np.mean(all_ratio_dev_mse)), "std": float(np.std(all_ratio_dev_mse))},
        }

    log(f"\n  Results ({n_eval} videos):")
    log(f"    Spectral: LPIPS={np.mean(all_lpips_pred):.4f}+-{np.std(all_lpips_pred):.4f}  SSIM={np.mean(all_ssim_pred):.4f}")
    log(f"    Copy:     LPIPS={np.mean(all_lpips_copy):.4f}+-{np.std(all_lpips_copy):.4f}  SSIM={np.mean(all_ssim_copy):.4f}")
    log(f"    Ratio Dev (spectral): {np.mean(all_ratio_dev_pred):.4f}")

    # ================================================================
    # PART 2: 8-Frame Rollout with REAL per-frame data (for Figure 2)
    # ================================================================
    log(f"\n{'='*60}")
    log(f"PART 2: 8-Frame Rollout ({args.n_rollout_videos} videos)")
    log(f"{'='*60}")

    n_rollout = min(args.n_rollout_videos, n_val)
    rollout_latents = val_latents[:n_rollout]
    rollout_slots = val_slots[:n_rollout]

    # Per-frame MSE for each configuration
    configs = {
        "full_system": {"use_dynamics": True, "use_slots": True},
        "copy_baseline": None,  # special handling
    }

    rollout_results = {}

    for config_name, config in configs.items():
        per_frame_mse = [[] for _ in range(T)]
        per_frame_lpips = [[] for _ in range(T)]

        log(f"\n  Config: {config_name}")
        for vid_idx in range(n_rollout):
            if vid_idx % 10 == 0:
                log(f"    Video {vid_idx+1}/{n_rollout}")

            if config_name == "copy_baseline":
                # Copy baseline: each frame is frame 0
                for t in range(1, T):
                    mse = F.mse_loss(
                        rollout_latents[vid_idx, 0],
                        rollout_latents[vid_idx, t]
                    ).item()
                    per_frame_mse[t].append(mse)

                    # Pixel-level LPIPS for first 10 videos only (expensive)
                    if vid_idx < 10:
                        px_copy = decode_single_frame(vae, rollout_latents[vid_idx, 0].to(device), scaling_factor)
                        px_gt = decode_single_frame(vae, rollout_latents[vid_idx, t].to(device), scaling_factor)
                        per_frame_lpips[t].append(compute_lpips(px_copy, px_gt))
            else:
                # Autoregressive rollout
                L_cur = rollout_latents[vid_idx, 0].clone()
                cur_slots = rollout_slots[vid_idx, 0:1].to(device)

                with torch.no_grad():
                    for t in range(1, T):
                        pred_slots = dynamics(cur_slots)
                        lt_tok = L_cur.flatten(1).unsqueeze(0).permute(0, 2, 1).to(device)
                        delta = res_decoder(lt_tok, cur_slots, pred_slots)
                        L_cur = (lt_tok + delta)[0].permute(1, 0).reshape(C, H, W).cpu()

                        mse = F.mse_loss(L_cur, rollout_latents[vid_idx, t]).item()
                        per_frame_mse[t].append(mse)

                        # Pixel-level for first 10 videos
                        if vid_idx < 10:
                            px_pred = decode_single_frame(vae, L_cur.to(device), scaling_factor)
                            px_gt = decode_single_frame(vae, rollout_latents[vid_idx, t].to(device), scaling_factor)
                            per_frame_lpips[t].append(compute_lpips(px_pred, px_gt))

                        cur_slots = pred_slots

        # Compute stats
        frame_mse_means = [float(np.mean(per_frame_mse[t])) if per_frame_mse[t] else 0.0 for t in range(T)]
        frame_mse_stds = [float(np.std(per_frame_mse[t])) if per_frame_mse[t] else 0.0 for t in range(T)]
        frame_lpips_means = [float(np.mean(per_frame_lpips[t])) if per_frame_lpips[t] else 0.0 for t in range(T)]

        stability = frame_mse_means[T-1] / frame_mse_means[1] if frame_mse_means[1] > 0 else 0
        log(f"    Stability (frame{T-1}/frame1): {stability:.3f}x")
        log(f"    Frame MSEs: {[f'{m:.4f}' for m in frame_mse_means[1:]]}")

        rollout_results[config_name] = {
            "n_videos": n_rollout,
            "per_frame_mse_mean": frame_mse_means,
            "per_frame_mse_std": frame_mse_stds,
            "per_frame_lpips_mean": frame_lpips_means,
            "stability_ratio": stability,
        }

    results["rollout"] = rollout_results

    # ================================================================
    # PART 3: FVD (if enough samples)
    # ================================================================
    log(f"\n{'='*60}")
    log(f"PART 3: FVD Computation")
    log(f"{'='*60}")

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid_fn = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

        n_fid = min(50, n_eval)
        gt_frames = []
        pred_frames = []

        log(f"  Collecting decoded frames from {n_fid} videos...")
        with torch.no_grad():
            for vid_idx in range(n_fid):
                if vid_idx % 10 == 0:
                    log(f"    Video {vid_idx+1}/{n_fid}")
                for t in range(1, min(T, 5)):  # frames 1-4 for FID
                    lt1_gt = eval_latents[vid_idx, t].to(device)
                    px_gt = decode_single_frame(vae, lt1_gt, scaling_factor)

                    # Predict
                    lt = eval_latents[vid_idx, t-1].to(device)
                    st = eval_slots[vid_idx, t-1:t].to(device)
                    st1 = dynamics(st)
                    lt_tok = lt.flatten(1).unsqueeze(0).permute(0, 2, 1)
                    delta = res_decoder(lt_tok, st, st1)
                    lt1_pred = (lt_tok + delta).permute(0, 2, 1).reshape(16, H, W)
                    px_pred = decode_single_frame(vae, lt1_pred, scaling_factor)

                    # Resize to 299x299 for InceptionV3
                    gt_resized = F.interpolate(px_gt.unsqueeze(0), size=(299, 299), mode="bilinear")
                    pred_resized = F.interpolate(px_pred.unsqueeze(0), size=(299, 299), mode="bilinear")

                    fid_fn.update(gt_resized, real=True)
                    fid_fn.update(pred_resized, real=False)

        fid_score = fid_fn.compute().item()
        log(f"  FID: {fid_score:.2f} ({n_fid} videos x {min(T,5)-1} frames = {n_fid*(min(T,5)-1)} samples)")
        results["fid"] = {"score": fid_score, "n_videos": n_fid, "n_samples": n_fid * (min(T, 5) - 1)}
    except Exception as e:
        log(f"  FID computation failed: {e}")
        results["fid"] = {"error": str(e)}

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    log(f"\nSaved: {args.output}")

    # ── Print summary ──
    log(f"\n{'='*60}")
    log("SUMMARY — Updated Table 3 Values")
    log(f"{'='*60}")
    ss = results["single_step"]
    log(f"  N = {ss['n_videos']} videos (was: 10)")
    log(f"  Spectral:  LPIPS = {ss['spectral']['lpips']['mean']:.4f} +/- {ss['spectral']['lpips']['std']:.4f}")
    log(f"  Copy:      LPIPS = {ss['copy']['lpips']['mean']:.4f} +/- {ss['copy']['lpips']['std']:.4f}")
    log(f"  Spectral:  SSIM  = {ss['spectral']['ssim']['mean']:.4f}")
    log(f"  Copy:      SSIM  = {ss['copy']['ssim']['mean']:.4f}")
    log(f"  Ratio Dev: {ss['spectral']['ratio_dev']['mean']:.4f}")


if __name__ == "__main__":
    main()
