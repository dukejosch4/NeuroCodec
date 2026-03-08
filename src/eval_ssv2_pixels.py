"""Phase 6: Full pixel-level evaluation on SSv2.

Evaluates all configurations and computes paper-ready metrics:
  1. NeuroCodec (GT slots) — upper bound
  2. NeuroCodec (predicted slots via dynamics) — realistic
  3. NeuroCodec (predicted + ManifoldProjector) — full system
  4. Copy baseline — lower bound
  5. SimVP baseline (if checkpoint provided)

Metrics: SSIM, PSNR, LPIPS (single-step) + rollout stability + FID

Usage:
    python src/eval_ssv2_pixels.py \
        --slot-ckpt checkpoints/ssv2/slot_encoder_best.pt \
        --dynamics-ckpt checkpoints/ssv2/dynamics_best.pt \
        --residual-ckpt checkpoints/ssv2/residual_spectral_best.pt \
        [--projector-ckpt checkpoints/ssv2/manifold_projector_best.pt] \
        [--simvp-ckpt checkpoints/ssv2/simvp_best.pt]

Output:
    results/json/ssv2_pixel_evaluation.json
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LATENT_DIR = Path("data/ssv2_latents")
CHECKPOINT_DIR = Path("checkpoints/ssv2")
RESULTS_DIR = Path("results/json")
SEED = 42
VAL_FRACTION = 0.15


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_ssim(img1, img2):
    """SSIM between two [3, H, W] tensors in [0, 1]."""
    from skimage.metrics import structural_similarity
    a = img1.cpu().numpy().transpose(1, 2, 0)
    b = img2.cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(a, b, channel_axis=2, data_range=1.0)


def compute_psnr(img1, img2):
    """PSNR between two [3, H, W] tensors in [0, 1]."""
    from skimage.metrics import peak_signal_noise_ratio
    a = img1.cpu().numpy().transpose(1, 2, 0)
    b = img2.cpu().numpy().transpose(1, 2, 0)
    return peak_signal_noise_ratio(a, b, data_range=1.0)


def compute_lpips(img1, img2, lpips_fn, device):
    """LPIPS between two [3, H, W] tensors in [0, 1]."""
    with torch.no_grad():
        a = (img1.unsqueeze(0) * 2 - 1).to(device)
        b = (img2.unsqueeze(0) * 2 - 1).to(device)
        return lpips_fn(a, b).item()


def predict_neurocodec(lt_spatial, slot_encoder, dynamics, residual, projector,
                       device, use_gt_slots=False, lt1_spatial_gt=None):
    """Single-step NeuroCodec prediction.

    Args:
        lt_spatial: [16, 32, 32] current latent frame
        slot_encoder: trained slot encoder
        dynamics: trained dynamics transformer
        residual: trained residual decoder
        projector: ManifoldProjector (or None)
        use_gt_slots: if True, extract slots from gt frame t+1 (oracle)
        lt1_spatial_gt: [16, 32, 32] gt next frame (needed if use_gt_slots)

    Returns:
        predicted latent [16, 32, 32]
    """
    C, H, W = lt_spatial.shape
    tok_t = lt_spatial.flatten(1).unsqueeze(0).permute(0, 2, 1).to(device)  # [1, 1024, 16]
    slots_t = slot_encoder.encode(tok_t)

    if use_gt_slots and lt1_spatial_gt is not None:
        tok_gt = lt1_spatial_gt.flatten(1).unsqueeze(0).permute(0, 2, 1).to(device)
        slots_t1 = slot_encoder.encode(tok_gt)
        # Hungarian-align
        from scipy.optimize import linear_sum_assignment
        a = F.normalize(slots_t[0], dim=-1)
        b = F.normalize(slots_t1[0], dim=-1)
        cost = 1 - torch.mm(a, b.t())
        _, col_ind = linear_sum_assignment(cost.cpu().numpy())
        slots_t1 = slots_t1[:, col_ind]
    else:
        slots_t1 = dynamics(slots_t)

    delta = residual(tok_t, slots_t, slots_t1)
    lt1_pred = (tok_t + delta).permute(0, 2, 1).reshape(C, H, W)

    if projector is not None:
        lt1_pred = projector(lt1_pred.unsqueeze(0))[0]

    return lt1_pred


def main():
    parser = argparse.ArgumentParser(description="SSv2 Full Pixel Evaluation")
    parser.add_argument("--slot-ckpt", type=str, required=True)
    parser.add_argument("--dynamics-ckpt", type=str, required=True)
    parser.add_argument("--residual-ckpt", type=str, required=True)
    parser.add_argument("--projector-ckpt", type=str, default=None)
    parser.add_argument("--simvp-ckpt", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default=str(LATENT_DIR))
    parser.add_argument("--n-eval-videos", type=int, default=100,
                        help="Videos for single-step eval")
    parser.add_argument("--n-rollout-videos", type=int, default=50,
                        help="Videos for rollout eval")
    parser.add_argument("--n-pixel-pairs-per-video", type=int, default=3,
                        help="Frame pairs per video for pixel-level metrics (VAE decode)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "ssv2_pixel_evaluation.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.n_eval_videos = 10
        args.n_rollout_videos = 5
        args.n_pixel_pairs_per_video = 1

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("PHASE 6: FULL PIXEL EVALUATION (SSv2)")
    log("=" * 60)
    log(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load models ──
    log("\n[1/5] Loading models...")
    from models import (SlotLatentAutoencoderV2, DynamicsTransformer,
                        ResidualDecoderV2, ManifoldProjector)

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

    projector = None
    if args.projector_ckpt and os.path.exists(args.projector_ckpt):
        projector = ManifoldProjector().to(device)
        projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device, weights_only=False))
        projector.eval()
        log(f"  ManifoldProjector: loaded")

    # SimVP (optional)
    simvp = None
    if args.simvp_ckpt and os.path.exists(args.simvp_ckpt):
        simvp_state = torch.load(args.simvp_ckpt, map_location=device, weights_only=False)
        from train_simvp_baseline import SimVPLatent
        simvp = SimVPLatent(**simvp_state.get("config", {})).to(device)
        simvp.load_state_dict(simvp_state["model"])
        simvp.eval()
        log(f"  SimVP: loaded ({sum(p.numel() for p in simvp.parameters()):,} params)")

    # VAE
    log("  Loading CogVideoX VAE...")
    from vae_utils import load_cogvideox_vae, decode_single_frame
    vae, scaling_factor = load_cogvideox_vae(device)
    log(f"  VAE: loaded")

    # LPIPS
    import lpips as lpips_module
    lpips_fn = lpips_module.LPIPS(net="alex").to(device).eval()
    log(f"  LPIPS: loaded")

    # ── Select validation videos ──
    log("\n[2/5] Selecting validation videos...")
    latent_dir = Path(args.latent_dir)
    all_files = sorted(latent_dir.glob("*.npy"))
    n_all = len(all_files)

    indices = list(range(n_all))
    random.seed(args.seed)
    random.shuffle(indices)
    n_val = int(n_all * VAL_FRACTION)
    val_indices = indices[:n_val]

    # Limit to requested number
    n_eval = min(args.n_eval_videos, len(val_indices))
    eval_indices = val_indices[:n_eval]
    eval_files = [str(all_files[i]) for i in eval_indices]

    n_rollout = min(args.n_rollout_videos, n_eval)
    log(f"  Total validation videos: {len(val_indices)}")
    log(f"  Single-step eval: {n_eval} videos")
    log(f"  Rollout eval: {n_rollout} videos")

    results = {
        "metadata": {
            "dataset": "ssv2", "n_eval_videos": n_eval, "n_rollout_videos": n_rollout,
            "n_pixel_pairs_per_video": args.n_pixel_pairs_per_video,
            "gpu": torch.cuda.get_device_name(0),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    # ================================================================
    # PART 1: SINGLE-STEP EVALUATION
    # ================================================================
    log(f"\n{'='*60}")
    log(f"PART 1: Single-Step Evaluation ({n_eval} videos)")
    log(f"{'='*60}")

    configs = {
        "neurocodec_gt_slots": {"use_gt": True, "use_proj": False},
        "neurocodec_pred_slots": {"use_gt": False, "use_proj": False},
    }
    if projector is not None:
        configs["neurocodec_pred_projected"] = {"use_gt": False, "use_proj": True}
    configs["copy_baseline"] = None  # special handling

    single_step_results = {}

    for config_name, config in configs.items():
        all_ssim, all_psnr, all_lpips = [], [], []
        all_latent_mse = []

        log(f"\n  Config: {config_name}")
        t_start = time.time()

        for vid_i, fpath in enumerate(eval_files):
            if vid_i % 20 == 0:
                elapsed = time.time() - t_start
                eta = (elapsed / max(vid_i, 1)) * (n_eval - vid_i)
                log(f"    Video {vid_i + 1}/{n_eval} (ETA: {eta:.0f}s)")

            latent = np.load(fpath).astype(np.float32)  # [16, T, 32, 32]
            C, T, H, W = latent.shape
            n_pairs = min(args.n_pixel_pairs_per_video, T - 1)

            for t in range(n_pairs):
                lt = torch.from_numpy(latent[:, t, :, :]).to(device)
                lt1_gt = torch.from_numpy(latent[:, t + 1, :, :]).to(device)

                with torch.no_grad():
                    if config is None:  # copy baseline
                        lt1_pred = lt.clone()
                    else:
                        use_proj = config["use_proj"]
                        lt1_pred = predict_neurocodec(
                            lt, slot_encoder, dynamics, residual,
                            projector if use_proj else None,
                            device, use_gt_slots=config["use_gt"],
                            lt1_spatial_gt=lt1_gt,
                        )

                    # Latent MSE
                    all_latent_mse.append(F.mse_loss(lt1_pred, lt1_gt).item())

                    # Pixel metrics (decode through VAE)
                    px_pred = decode_single_frame(vae, lt1_pred, scaling_factor)
                    px_gt = decode_single_frame(vae, lt1_gt, scaling_factor)

                    all_ssim.append(compute_ssim(px_pred, px_gt))
                    all_psnr.append(compute_psnr(px_pred, px_gt))
                    all_lpips.append(compute_lpips(px_pred, px_gt, lpips_fn, device))

        single_step_results[config_name] = {
            "ssim": {"mean": float(np.mean(all_ssim)), "std": float(np.std(all_ssim))},
            "psnr": {"mean": float(np.mean(all_psnr)), "std": float(np.std(all_psnr))},
            "lpips": {"mean": float(np.mean(all_lpips)), "std": float(np.std(all_lpips))},
            "latent_mse": {"mean": float(np.mean(all_latent_mse)), "std": float(np.std(all_latent_mse))},
            "n_pairs": len(all_ssim),
        }

        log(f"    SSIM={np.mean(all_ssim):.4f}  PSNR={np.mean(all_psnr):.2f}dB  "
            f"LPIPS={np.mean(all_lpips):.4f}  LatMSE={np.mean(all_latent_mse):.4f}")

    # SimVP single-step
    if simvp is not None:
        log(f"\n  Config: simvp_baseline")
        all_ssim, all_psnr, all_lpips, all_latent_mse = [], [], [], []
        T_in = 4

        for vid_i, fpath in enumerate(eval_files):
            if vid_i % 20 == 0:
                log(f"    Video {vid_i + 1}/{n_eval}")

            latent = np.load(fpath).astype(np.float32)
            C, T, H, W = latent.shape
            if T < T_in + 1:
                continue

            n_pairs = min(args.n_pixel_pairs_per_video, T - T_in)
            for t in range(n_pairs):
                window = torch.from_numpy(
                    latent[:, t:t + T_in, :, :].transpose(1, 0, 2, 3).copy()
                ).unsqueeze(0).to(device)  # [1, T_in, 16, 32, 32]
                lt1_gt = torch.from_numpy(latent[:, t + T_in, :, :]).to(device)

                with torch.no_grad():
                    lt1_pred = simvp(window)[0]  # [16, 32, 32]
                    all_latent_mse.append(F.mse_loss(lt1_pred, lt1_gt).item())

                    px_pred = decode_single_frame(vae, lt1_pred, scaling_factor)
                    px_gt = decode_single_frame(vae, lt1_gt, scaling_factor)

                    all_ssim.append(compute_ssim(px_pred, px_gt))
                    all_psnr.append(compute_psnr(px_pred, px_gt))
                    all_lpips.append(compute_lpips(px_pred, px_gt, lpips_fn, device))

        if all_ssim:
            single_step_results["simvp_baseline"] = {
                "ssim": {"mean": float(np.mean(all_ssim)), "std": float(np.std(all_ssim))},
                "psnr": {"mean": float(np.mean(all_psnr)), "std": float(np.std(all_psnr))},
                "lpips": {"mean": float(np.mean(all_lpips)), "std": float(np.std(all_lpips))},
                "latent_mse": {"mean": float(np.mean(all_latent_mse)), "std": float(np.std(all_latent_mse))},
                "n_pairs": len(all_ssim),
                "T_in": T_in,
            }
            log(f"    SSIM={np.mean(all_ssim):.4f}  PSNR={np.mean(all_psnr):.2f}dB  "
                f"LPIPS={np.mean(all_lpips):.4f}")

    results["single_step"] = single_step_results

    # ================================================================
    # PART 2: ROLLOUT EVALUATION
    # ================================================================
    log(f"\n{'='*60}")
    log(f"PART 2: Rollout Evaluation ({n_rollout} videos)")
    log(f"{'='*60}")

    rollout_configs = ["neurocodec_pred_slots", "copy_baseline"]
    if simvp is not None:
        rollout_configs.append("simvp_baseline")

    rollout_files = eval_files[:n_rollout]
    rollout_results = {}

    for config_name in rollout_configs:
        log(f"\n  Config: {config_name}")

        # Load videos and find max T
        all_latents = []
        for fpath in rollout_files:
            all_latents.append(np.load(fpath).astype(np.float32))

        T_max = min(lat.shape[1] for lat in all_latents)
        n_rollout_steps = T_max - 1
        per_frame_mse = [[] for _ in range(T_max)]
        per_frame_ssim = [[] for _ in range(T_max)]

        for vid_i, latent in enumerate(all_latents):
            if vid_i % 10 == 0:
                log(f"    Video {vid_i + 1}/{n_rollout}")

            C, T, H, W = latent.shape

            with torch.no_grad():
                if config_name == "copy_baseline":
                    for t in range(1, T):
                        lt0 = torch.from_numpy(latent[:, 0, :, :])
                        lt_gt = torch.from_numpy(latent[:, t, :, :])
                        per_frame_mse[t].append(F.mse_loss(lt0, lt_gt).item())

                        # Pixel SSIM for first 5 videos only (expensive)
                        if vid_i < 5:
                            px_pred = decode_single_frame(vae, lt0.to(device), scaling_factor)
                            px_gt = decode_single_frame(vae, lt_gt.to(device), scaling_factor)
                            per_frame_ssim[t].append(compute_ssim(px_pred, px_gt))

                elif config_name == "neurocodec_pred_slots":
                    L_cur = torch.from_numpy(latent[:, 0, :, :]).to(device)
                    tok_cur = L_cur.flatten(1).unsqueeze(0).permute(0, 2, 1)
                    cur_slots = slot_encoder.encode(tok_cur)

                    for t in range(1, T):
                        pred_slots = dynamics(cur_slots)
                        delta = residual(tok_cur, cur_slots, pred_slots)
                        L_next = (tok_cur + delta)[0].permute(1, 0).reshape(C, H, W)

                        if projector is not None:
                            L_next_out = projector(L_next.unsqueeze(0))[0]
                        else:
                            L_next_out = L_next

                        lt_gt = torch.from_numpy(latent[:, t, :, :]).to(device)
                        per_frame_mse[t].append(F.mse_loss(L_next_out, lt_gt).item())

                        if vid_i < 5:
                            px_pred = decode_single_frame(vae, L_next_out, scaling_factor)
                            px_gt = decode_single_frame(vae, lt_gt, scaling_factor)
                            per_frame_ssim[t].append(compute_ssim(px_pred, px_gt))

                        # Feed back RAW prediction (decoupled feedback)
                        tok_cur = L_next.flatten(1).unsqueeze(0).permute(0, 2, 1)
                        cur_slots = pred_slots

                elif config_name == "simvp_baseline" and simvp is not None:
                    T_in = 4
                    if T < T_in + 1:
                        continue

                    # Initialize with GT first T_in frames
                    buffer = [torch.from_numpy(latent[:, i, :, :]).to(device) for i in range(T_in)]

                    for t in range(T_in, T):
                        window = torch.stack(buffer[-T_in:]).unsqueeze(0)  # [1, T_in, C, H, W]
                        L_next = simvp(window)[0]  # [C, H, W]

                        lt_gt = torch.from_numpy(latent[:, t, :, :]).to(device)
                        per_frame_mse[t].append(F.mse_loss(L_next, lt_gt).item())

                        if vid_i < 5:
                            px_pred = decode_single_frame(vae, L_next, scaling_factor)
                            px_gt = decode_single_frame(vae, lt_gt, scaling_factor)
                            per_frame_ssim[t].append(compute_ssim(px_pred, px_gt))

                        buffer.append(L_next)

        # Compile stats
        frame_mse_means = [float(np.mean(per_frame_mse[t])) if per_frame_mse[t] else 0.0
                           for t in range(T_max)]
        frame_ssim_means = [float(np.mean(per_frame_ssim[t])) if per_frame_ssim[t] else 0.0
                            for t in range(T_max)]

        # Stability ratio
        first_frame = 1 if config_name != "simvp_baseline" else 4
        last_frame = T_max - 1
        if frame_mse_means[first_frame] > 0:
            stability = frame_mse_means[last_frame] / frame_mse_means[first_frame]
        else:
            stability = 0.0

        log(f"    Stability (frame{last_frame}/frame{first_frame}): {stability:.3f}x")
        log(f"    Per-frame MSE: {[f'{m:.4f}' for m in frame_mse_means[1:] if m > 0]}")

        rollout_results[config_name] = {
            "per_frame_mse": frame_mse_means,
            "per_frame_ssim": frame_ssim_means,
            "stability_ratio": stability,
            "n_videos": len(rollout_files),
        }

    results["rollout"] = rollout_results

    # ================================================================
    # PART 3: FID
    # ================================================================
    log(f"\n{'='*60}")
    log(f"PART 3: FID Computation")
    log(f"{'='*60}")

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid_fn = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

        n_fid = min(50, n_eval)
        log(f"  Collecting frames from {n_fid} videos...")

        with torch.no_grad():
            for vid_i in range(n_fid):
                if vid_i % 10 == 0:
                    log(f"    Video {vid_i + 1}/{n_fid}")

                latent = np.load(eval_files[vid_i]).astype(np.float32)
                C, T, H, W = latent.shape

                for t in range(1, min(T, 4)):
                    lt = torch.from_numpy(latent[:, t - 1, :, :]).to(device)
                    lt1_gt = torch.from_numpy(latent[:, t, :, :]).to(device)

                    lt1_pred = predict_neurocodec(
                        lt, slot_encoder, dynamics, residual, projector,
                        device, use_gt_slots=False,
                    )

                    px_gt = decode_single_frame(vae, lt1_gt, scaling_factor)
                    px_pred = decode_single_frame(vae, lt1_pred, scaling_factor)

                    gt_r = F.interpolate(px_gt.unsqueeze(0), size=(299, 299), mode="bilinear")
                    pred_r = F.interpolate(px_pred.unsqueeze(0), size=(299, 299), mode="bilinear")

                    fid_fn.update(gt_r, real=True)
                    fid_fn.update(pred_r, real=False)

        fid_score = fid_fn.compute().item()
        n_samples = n_fid * min(T - 1, 3)
        log(f"  FID: {fid_score:.2f} ({n_samples} samples)")
        results["fid"] = {"score": fid_score, "n_videos": n_fid, "n_samples": n_samples}

    except Exception as e:
        log(f"  FID failed: {e}")
        results["fid"] = {"error": str(e)}

    # ================================================================
    # PART 4: FVD (Fréchet Video Distance)
    # ================================================================
    log(f"\n{'='*60}")
    log(f"PART 4: FVD Computation")
    log(f"{'='*60}")

    try:
        from scipy.linalg import sqrtm

        # Load R3D-18 as video feature extractor
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            r3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).to(device).eval()
        except (ImportError, TypeError):
            from torchvision.models.video import r3d_18
            r3d = r3d_18(pretrained=True).to(device).eval()

        r3d.fc = torch.nn.Identity()  # remove classifier → 512-dim features

        n_fvd = min(50, n_eval)
        T_clip = 8
        log(f"  Extracting R3D-18 features from {n_fvd} videos ({T_clip}-frame clips)...")

        feats_real, feats_fake = [], []

        # Kinetics normalization
        r3d_mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1).to(device)
        r3d_std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1).to(device)

        with torch.no_grad():
            for vid_i in range(n_fvd):
                if vid_i % 10 == 0:
                    log(f"    Video {vid_i + 1}/{n_fvd}")

                latent = np.load(eval_files[vid_i]).astype(np.float32)
                C, T, H, W = latent.shape
                T_use = min(T, T_clip + 1)
                if T_use < 4:
                    continue

                # Decode GT frames
                gt_frames = []
                for t in range(T_use):
                    lt_gt = torch.from_numpy(latent[:, t, :, :]).to(device)
                    px = decode_single_frame(vae, lt_gt, scaling_factor)
                    gt_frames.append(px)

                # Predict frames autoregressively
                pred_frames = []
                L_cur = torch.from_numpy(latent[:, 0, :, :]).to(device)
                px_first = decode_single_frame(vae, L_cur, scaling_factor)
                pred_frames.append(px_first)

                tok_cur = L_cur.flatten(1).unsqueeze(0).permute(0, 2, 1)
                cur_slots = slot_encoder.encode(tok_cur)

                for t in range(1, T_use):
                    pred_slots = dynamics(cur_slots)
                    delta = residual(tok_cur, cur_slots, pred_slots)
                    L_next = (tok_cur + delta)[0].permute(1, 0).reshape(C, H, W)
                    if projector is not None:
                        L_next_out = projector(L_next.unsqueeze(0))[0]
                    else:
                        L_next_out = L_next

                    px_pred = decode_single_frame(vae, L_next_out, scaling_factor)
                    pred_frames.append(px_pred)

                    # Feed back raw prediction (decoupled feedback)
                    tok_cur = L_next.flatten(1).unsqueeze(0).permute(0, 2, 1)
                    cur_slots = pred_slots

                # Form clips: [T, 3, H, W] → resize to 112x112 → [1, 3, T, 112, 112]
                n_cf = min(len(gt_frames), len(pred_frames), T_clip)
                gt_clip = F.interpolate(torch.stack(gt_frames[:n_cf]), size=(112, 112), mode="bilinear")
                pred_clip = F.interpolate(torch.stack(pred_frames[:n_cf]), size=(112, 112), mode="bilinear")

                gt_in = gt_clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)    # [1, 3, T, 112, 112]
                pred_in = pred_clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)

                gt_in = (gt_in - r3d_mean) / r3d_std
                pred_in = (pred_in - r3d_mean) / r3d_std

                feats_real.append(r3d(gt_in).cpu())
                feats_fake.append(r3d(pred_in).cpu())

        if len(feats_real) >= 10:
            feats_r = torch.cat(feats_real).numpy()
            feats_f = torch.cat(feats_fake).numpy()

            mu_r, sigma_r = feats_r.mean(0), np.cov(feats_r, rowvar=False)
            mu_f, sigma_f = feats_f.mean(0), np.cov(feats_f, rowvar=False)

            # Regularize for numerical stability (N may be < 512)
            eps = 1e-6
            sigma_r += eps * np.eye(sigma_r.shape[0])
            sigma_f += eps * np.eye(sigma_f.shape[0])

            diff = mu_r - mu_f
            covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fvd_score = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))
            log(f"  FVD: {fvd_score:.2f} ({len(feats_real)} clips x {T_clip} frames)")
            results["fvd"] = {"score": fvd_score, "n_clips": len(feats_real), "clip_length": T_clip}
        else:
            log(f"  FVD: too few clips ({len(feats_real)}), need >= 10")
            results["fvd"] = {"error": f"too few clips: {len(feats_real)}"}

    except Exception as e:
        log(f"  FVD failed: {e}")
        import traceback
        traceback.print_exc()
        results["fvd"] = {"error": str(e)}

    # ================================================================
    # SUMMARY
    # ================================================================
    log(f"\n{'='*60}")
    log("SUMMARY — Paper-Ready Results")
    log(f"{'='*60}")

    log(f"\n  Single-Step ({n_eval} videos):")
    log(f"  {'Config':<35s} {'SSIM':>8s} {'PSNR':>8s} {'LPIPS':>8s} {'LatMSE':>8s}")
    log(f"  {'-'*67}")
    for name, data in single_step_results.items():
        log(f"  {name:<35s} {data['ssim']['mean']:>8.4f} {data['psnr']['mean']:>7.2f} "
            f"{data['lpips']['mean']:>8.4f} {data['latent_mse']['mean']:>8.4f}")

    log(f"\n  Rollout ({n_rollout} videos):")
    for name, data in rollout_results.items():
        log(f"  {name}: stability={data['stability_ratio']:.3f}x")

    if "fid" in results and "score" in results["fid"]:
        log(f"\n  FID: {results['fid']['score']:.2f}")
    if "fvd" in results and "score" in results["fvd"]:
        log(f"  FVD: {results['fvd']['score']:.2f}")

    # ── Model info ──
    results["model_info"] = {
        "neurocodec_params": {
            "slot_encoder": sum(p.numel() for p in slot_encoder.parameters()),
            "dynamics": sum(p.numel() for p in dynamics.parameters()),
            "residual": sum(p.numel() for p in residual.parameters()),
            "projector": sum(p.numel() for p in projector.parameters()) if projector else 0,
            "total_trainable": (sum(p.numel() for p in dynamics.parameters()) +
                                sum(p.numel() for p in residual.parameters()) +
                                (sum(p.numel() for p in projector.parameters()) if projector else 0)),
        },
    }
    if simvp is not None:
        results["model_info"]["simvp_params"] = sum(p.numel() for p in simvp.parameters())

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    log(f"\nResults saved to {args.output}")
    log("Done.")


if __name__ == "__main__":
    main()
