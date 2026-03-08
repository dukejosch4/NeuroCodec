"""Phase 10: Extended Evaluation — reviewer-ready benchmarks.

Fixes gaps from Phase 7:
  1. Rollout SSIM+LPIPS+PSNR on ALL 50 videos (was 5)
  2. FID with 2000+ samples (was 150)
  3. FID/FVD for ALL methods (was NeuroCodec only)
  4. Inference speed measurement
  5. neurocodec_pred_projected added to rollout

Output: results/json/ssv2_extended_evaluation.json
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def compute_ssim(img1, img2):
    from skimage.metrics import structural_similarity
    a = img1.cpu().numpy().transpose(1, 2, 0)
    b = img2.cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(a, b, channel_axis=2, data_range=1.0)


def compute_psnr(img1, img2):
    from skimage.metrics import peak_signal_noise_ratio
    a = img1.cpu().numpy().transpose(1, 2, 0)
    b = img2.cpu().numpy().transpose(1, 2, 0)
    return peak_signal_noise_ratio(a, b, data_range=1.0)


def main():
    parser = argparse.ArgumentParser(description="Extended SSv2 evaluation")
    parser.add_argument("--slot-ckpt", required=True)
    parser.add_argument("--dynamics-ckpt", required=True)
    parser.add_argument("--residual-ckpt", required=True)
    parser.add_argument("--projector-ckpt", default=None)
    parser.add_argument("--simvp-ckpt", default=None)
    parser.add_argument("--latent-dir", default="data/ssv2_latents")
    parser.add_argument("--n-fid-videos", type=int, default=500,
                        help="Videos for FID (x4 frames = samples)")
    parser.add_argument("--n-rollout-videos", type=int, default=50)
    parser.add_argument("--n-fvd-videos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/json/ssv2_extended_evaluation.json")
    parser.add_argument("--skip-rollout", action="store_true",
                        help="Skip Part 1 if rollout results exist in output JSON")
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    log("=" * 60)
    log("PHASE 10: EXTENDED EVALUATION")
    log("=" * 60)

    # ── Load models ──
    log("[1/6] Loading models...")
    from models import (SlotLatentAutoencoderV2, DynamicsTransformer,
                        ResidualDecoderV2, ManifoldProjector)

    slot_encoder = SlotLatentAutoencoderV2(
        n_slots=64, slot_dim=128, input_dim=16, n_tokens=1024, n_iter=5
    ).to(device).eval()
    slot_encoder.load_state_dict(torch.load(args.slot_ckpt, map_location=device, weights_only=False))

    dynamics = DynamicsTransformer(
        n_tokens=64, token_dim=128, d_model=128, n_heads=4, n_layers=2
    ).to(device).eval()
    dynamics.load_state_dict(torch.load(args.dynamics_ckpt, map_location=device, weights_only=False))

    residual = ResidualDecoderV2().to(device).eval()
    residual.load_state_dict(torch.load(args.residual_ckpt, map_location=device, weights_only=False))

    projector = None
    if args.projector_ckpt and os.path.exists(args.projector_ckpt):
        projector = ManifoldProjector().to(device).eval()
        projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device, weights_only=False))

    simvp = None
    if args.simvp_ckpt and os.path.exists(args.simvp_ckpt):
        simvp_state = torch.load(args.simvp_ckpt, map_location=device, weights_only=False)
        from train_simvp_baseline import SimVPLatent
        simvp = SimVPLatent(**simvp_state.get("config", {})).to(device).eval()
        simvp.load_state_dict(simvp_state["model"])

    from vae_utils import load_cogvideox_vae, decode_single_frame
    vae, scaling_factor = load_cogvideox_vae(device)

    import lpips as lpips_module
    lpips_fn = lpips_module.LPIPS(net="alex").to(device).eval()
    log("  All models loaded.")

    # ── Validation files ──
    log("[2/6] Selecting validation videos...")
    latent_dir = Path(args.latent_dir)
    all_files = sorted(latent_dir.glob("*.npy"))
    indices = list(range(len(all_files)))
    random.shuffle(indices)
    n_val = int(len(all_files) * 0.15)
    val_files = [str(all_files[i]) for i in indices[:n_val]]
    log(f"  {len(val_files)} validation videos")

    results = {"metadata": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
    }}

    # ── Helpers ──
    @torch.no_grad()
    def predict_nc(lt_spatial, use_proj=True):
        C, H, W = lt_spatial.shape
        tok = lt_spatial.flatten(1).unsqueeze(0).permute(0, 2, 1).to(device)
        slots = slot_encoder.encode(tok)
        pred_s = dynamics(slots)
        delta = residual(tok, slots, pred_s)
        out = (tok + delta).permute(0, 2, 1).reshape(C, H, W)
        if use_proj and projector is not None:
            out = projector(out.unsqueeze(0))[0]
        return out

    def compute_lpips_val(img1, img2):
        with torch.no_grad():
            a = (img1.unsqueeze(0) * 2 - 1).to(device)
            b = (img2.unsqueeze(0) * 2 - 1).to(device)
            return lpips_fn(a, b).item()

    # Method list
    all_methods = ["neurocodec_pred_slots"]
    if projector is not None:
        all_methods.append("neurocodec_pred_projected")
    all_methods.append("copy_baseline")
    if simvp is not None:
        all_methods.append("simvp_baseline")

    # ================================================================
    # PART 1: EXTENDED ROLLOUT (SSIM + LPIPS + PSNR on ALL videos)
    # ================================================================
    # Check if we can skip Part 1
    existing_results = {}
    if args.skip_rollout and os.path.exists(args.output):
        with open(args.output) as f:
            existing_results = json.load(f)
        if "rollout_extended" in existing_results:
            log("\n[3/6] PART 1: SKIPPED (--skip-rollout, results exist)")
            results["rollout_extended"] = existing_results["rollout_extended"]
            rollout_results = existing_results["rollout_extended"]
        else:
            args.skip_rollout = False  # no results found, run anyway

    if not args.skip_rollout:
        log("\n" + "=" * 60)
        log("[3/6] PART 1: Extended Rollout — full pixel metrics on all videos")
        log("=" * 60)

    n_rollout = min(args.n_rollout_videos, len(val_files))
    rollout_files = val_files[:n_rollout]
    if not args.skip_rollout:
        rollout_results = {}

    for config_name in (all_methods if not args.skip_rollout else []):
        log(f"\n  Config: {config_name}")
        t_start = time.time()

        all_latents = [np.load(f).astype(np.float32) for f in rollout_files]
        T_max = min(lat.shape[1] for lat in all_latents)

        per_frame = {m: [[] for _ in range(T_max)] for m in ["mse", "ssim", "psnr", "lpips"]}

        for vid_i, latent in enumerate(all_latents):
            if vid_i % 5 == 0:
                elapsed = time.time() - t_start
                log(f"    Video {vid_i + 1}/{n_rollout} ({elapsed:.0f}s)")

            C, T, H, W = latent.shape

            with torch.no_grad():
                if config_name == "copy_baseline":
                    lt0 = torch.from_numpy(latent[:, 0, :, :]).to(device)
                    px0 = decode_single_frame(vae, lt0, scaling_factor)
                    for t in range(1, T):
                        lt_gt = torch.from_numpy(latent[:, t, :, :]).to(device)
                        per_frame["mse"][t].append(F.mse_loss(lt0, lt_gt).item())
                        px_gt = decode_single_frame(vae, lt_gt, scaling_factor)
                        per_frame["ssim"][t].append(compute_ssim(px0, px_gt))
                        per_frame["psnr"][t].append(compute_psnr(px0, px_gt))
                        per_frame["lpips"][t].append(compute_lpips_val(px0, px_gt))

                elif config_name in ("neurocodec_pred_slots", "neurocodec_pred_projected"):
                    use_proj = "projected" in config_name
                    L_cur = torch.from_numpy(latent[:, 0, :, :]).to(device)
                    tok_cur = L_cur.flatten(1).unsqueeze(0).permute(0, 2, 1)
                    cur_slots = slot_encoder.encode(tok_cur)

                    for t in range(1, T):
                        pred_slots = dynamics(cur_slots)
                        delta = residual(tok_cur, cur_slots, pred_slots)
                        L_next = (tok_cur + delta)[0].permute(1, 0).reshape(C, H, W)
                        L_out = projector(L_next.unsqueeze(0))[0] if (use_proj and projector) else L_next

                        lt_gt = torch.from_numpy(latent[:, t, :, :]).to(device)
                        per_frame["mse"][t].append(F.mse_loss(L_out, lt_gt).item())

                        px_pred = decode_single_frame(vae, L_out, scaling_factor)
                        px_gt = decode_single_frame(vae, lt_gt, scaling_factor)
                        per_frame["ssim"][t].append(compute_ssim(px_pred, px_gt))
                        per_frame["psnr"][t].append(compute_psnr(px_pred, px_gt))
                        per_frame["lpips"][t].append(compute_lpips_val(px_pred, px_gt))

                        # Decoupled feedback: raw prediction (before projector)
                        tok_cur = L_next.flatten(1).unsqueeze(0).permute(0, 2, 1)
                        cur_slots = pred_slots

                elif config_name == "simvp_baseline" and simvp is not None:
                    T_in = 4
                    if T < T_in + 1:
                        continue
                    buffer = [torch.from_numpy(latent[:, i, :, :]).to(device) for i in range(T_in)]
                    for t in range(T_in, T):
                        window = torch.stack(buffer[-T_in:]).unsqueeze(0)
                        L_next = simvp(window)[0]
                        lt_gt = torch.from_numpy(latent[:, t, :, :]).to(device)
                        per_frame["mse"][t].append(F.mse_loss(L_next, lt_gt).item())
                        px_pred = decode_single_frame(vae, L_next, scaling_factor)
                        px_gt = decode_single_frame(vae, lt_gt, scaling_factor)
                        per_frame["ssim"][t].append(compute_ssim(px_pred, px_gt))
                        per_frame["psnr"][t].append(compute_psnr(px_pred, px_gt))
                        per_frame["lpips"][t].append(compute_lpips_val(px_pred, px_gt))
                        buffer.append(L_next)

        # Compile statistics
        compiled = {}
        for metric in ["mse", "ssim", "psnr", "lpips"]:
            compiled[f"per_frame_{metric}"] = [float(np.mean(v)) if v else 0.0 for v in per_frame[metric]]
            compiled[f"per_frame_{metric}_std"] = [float(np.std(v)) if v else 0.0 for v in per_frame[metric]]

        first_frame = 1 if config_name != "simvp_baseline" else 4
        last_frame = T_max - 1
        mf = compiled["per_frame_mse"][first_frame]
        compiled["stability_mse"] = compiled["per_frame_mse"][last_frame] / mf if mf > 0 else 0
        compiled["n_videos"] = n_rollout
        compiled["T_max"] = T_max

        active_ssim = [v for v in compiled["per_frame_ssim"] if v > 0]
        active_lpips = [v for v in compiled["per_frame_lpips"] if v > 0]
        log(f"    Mean SSIM: {np.mean(active_ssim):.4f}, Mean LPIPS: {np.mean(active_lpips):.4f}")
        log(f"    Stability MSE (f{last_frame}/f{first_frame}): {compiled['stability_mse']:.3f}x")
        log(f"    Time: {time.time() - t_start:.0f}s")

        rollout_results[config_name] = compiled

    results["rollout_extended"] = rollout_results
    with open(args.output, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    log(f"  Saved: {args.output}")

    # ================================================================
    # PART 2: EXTENDED FID (all methods, 2000+ samples)
    # ================================================================
    log("\n" + "=" * 60)
    log("[4/6] PART 2: Extended FID — all methods, more samples")
    log("=" * 60)

    from torchmetrics.image.fid import FrechetInceptionDistance
    n_fid = min(args.n_fid_videos, len(val_files))
    log(f"  {n_fid} videos x 4 frames = ~{n_fid * 4} samples per method")

    # Pre-decode GT frames for FID (cached, reused across methods)
    log("  Pre-decoding GT frames...")
    gt_fid_images = []  # [1, 3, 299, 299] CPU tensors
    t_start = time.time()

    with torch.no_grad():
        for vid_i in range(n_fid):
            if vid_i % 100 == 0:
                log(f"    GT decode: {vid_i + 1}/{n_fid}")
            latent = np.load(val_files[vid_i]).astype(np.float32)
            C, T, H, W = latent.shape
            for t in range(1, min(T, 5)):
                lt_gt = torch.from_numpy(latent[:, t, :, :]).to(device)
                px_gt = decode_single_frame(vae, lt_gt, scaling_factor)
                gt_r = F.interpolate(px_gt.unsqueeze(0), size=(299, 299), mode="bilinear")
                gt_fid_images.append(gt_r.cpu())

    log(f"  GT cache: {len(gt_fid_images)} images ({time.time() - t_start:.0f}s)")

    fid_methods = []
    if projector is not None:
        fid_methods.append("neurocodec_pred_projected")
    else:
        fid_methods.append("neurocodec_pred_slots")
    if simvp is not None:
        fid_methods.append("simvp_baseline")
    fid_methods.append("copy_baseline")

    fid_results = {}
    for method in fid_methods:
        log(f"\n  FID: {method}")
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        t_start = time.time()

        # Feed cached GT in batches
        batch_sz = 8
        for i in range(0, len(gt_fid_images), batch_sz):
            batch = torch.cat(gt_fid_images[i:i + batch_sz]).to(device)
            fid_metric.update(batch, real=True)

        # Generate predictions
        n_fake = 0
        with torch.no_grad():
            for vid_i in range(n_fid):
                if vid_i % 100 == 0:
                    log(f"    Pred: {vid_i + 1}/{n_fid}")
                latent = np.load(val_files[vid_i]).astype(np.float32)
                C, T, H, W = latent.shape

                if method == "copy_baseline":
                    for t in range(1, min(T, 5)):
                        lt = torch.from_numpy(latent[:, t - 1, :, :]).to(device)
                        px = decode_single_frame(vae, lt, scaling_factor)
                        r = F.interpolate(px.unsqueeze(0), size=(299, 299), mode="bilinear")
                        fid_metric.update(r, real=False)
                        n_fake += 1

                elif method == "simvp_baseline" and simvp is not None:
                    T_in = 4
                    for p in range(min(4, T - T_in)):
                        window = torch.from_numpy(
                            latent[:, p:p + T_in, :, :].transpose(1, 0, 2, 3).copy()
                        ).unsqueeze(0).to(device)
                        lt1_pred = simvp(window)[0]
                        px = decode_single_frame(vae, lt1_pred, scaling_factor)
                        r = F.interpolate(px.unsqueeze(0), size=(299, 299), mode="bilinear")
                        fid_metric.update(r, real=False)
                        n_fake += 1

                else:  # neurocodec
                    use_proj = "projected" in method
                    for t in range(min(4, T - 1)):
                        lt = torch.from_numpy(latent[:, t, :, :]).to(device)
                        lt1_pred = predict_nc(lt, use_proj=use_proj)
                        px = decode_single_frame(vae, lt1_pred, scaling_factor)
                        r = F.interpolate(px.unsqueeze(0), size=(299, 299), mode="bilinear")
                        fid_metric.update(r, real=False)
                        n_fake += 1

        score = fid_metric.compute().item()
        log(f"    FID: {score:.2f} ({n_fake} samples, {time.time() - t_start:.0f}s)")
        fid_results[method] = {"score": score, "n_samples": n_fake}
        del fid_metric
        torch.cuda.empty_cache()

    results["fid_extended"] = fid_results
    with open(args.output, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)

    # ================================================================
    # PART 3: FVD per method (shared GT features)
    # ================================================================
    log("\n" + "=" * 60)
    log("[5/6] PART 3: FVD — all methods")
    log("=" * 60)

    from scipy.linalg import sqrtm
    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        r3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).to(device).eval()
    except (ImportError, TypeError):
        from torchvision.models.video import r3d_18
        r3d = r3d_18(pretrained=True).to(device).eval()
    r3d.fc = torch.nn.Identity()

    r3d_mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1).to(device)
    r3d_std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1).to(device)

    n_fvd = min(args.n_fvd_videos, len(val_files))
    T_clip = 8

    # Pre-compute GT R3D features (shared across methods)
    log(f"  Extracting GT R3D features ({n_fvd} clips)...")
    feats_real = []
    fvd_vid_indices = []

    with torch.no_grad():
        for vid_i in range(n_fvd):
            if vid_i % 20 == 0:
                log(f"    GT clip {vid_i + 1}/{n_fvd}")
            latent = np.load(val_files[vid_i]).astype(np.float32)
            C, T, H, W = latent.shape
            T_use = min(T, T_clip + 1)
            if T_use < 4:
                continue

            gt_frames = []
            for t in range(T_use):
                px = decode_single_frame(
                    vae, torch.from_numpy(latent[:, t, :, :]).to(device), scaling_factor
                )
                gt_frames.append(px)

            n_cf = min(len(gt_frames), T_clip)
            gt_clip = F.interpolate(torch.stack(gt_frames[:n_cf]), size=(112, 112), mode="bilinear")
            gt_in = gt_clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            gt_in = (gt_in - r3d_mean) / r3d_std
            feats_real.append(r3d(gt_in).cpu())
            fvd_vid_indices.append(vid_i)

    feats_r = torch.cat(feats_real).numpy()
    mu_r = feats_r.mean(0)
    sigma_r = np.cov(feats_r, rowvar=False) + 1e-6 * np.eye(feats_r.shape[1])
    log(f"  GT features: {len(feats_real)} clips")

    fvd_results = {}
    for method in fid_methods:
        log(f"\n  FVD: {method}")
        feats_fake = []
        t_start = time.time()

        with torch.no_grad():
            for idx, vid_i in enumerate(fvd_vid_indices):
                if idx % 20 == 0:
                    log(f"    Clip {idx + 1}/{len(fvd_vid_indices)}")
                latent = np.load(val_files[vid_i]).astype(np.float32)
                C, T, H, W = latent.shape
                T_use = min(T, T_clip + 1)
                pred_frames = []

                if method == "copy_baseline":
                    px0 = decode_single_frame(
                        vae, torch.from_numpy(latent[:, 0, :, :]).to(device), scaling_factor
                    )
                    pred_frames = [px0] * T_use

                elif method == "simvp_baseline" and simvp is not None:
                    T_in = 4
                    for tt in range(T_in):
                        px = decode_single_frame(
                            vae, torch.from_numpy(latent[:, tt, :, :]).to(device), scaling_factor
                        )
                        pred_frames.append(px)
                    buffer = [torch.from_numpy(latent[:, i, :, :]).to(device) for i in range(T_in)]
                    for tt in range(T_in, T_use):
                        window = torch.stack(buffer[-T_in:]).unsqueeze(0)
                        L_next = simvp(window)[0]
                        px = decode_single_frame(vae, L_next, scaling_factor)
                        pred_frames.append(px)
                        buffer.append(L_next)

                else:  # neurocodec
                    use_proj = "projected" in method
                    L_cur = torch.from_numpy(latent[:, 0, :, :]).to(device)
                    px0 = decode_single_frame(vae, L_cur, scaling_factor)
                    pred_frames.append(px0)
                    tok_cur = L_cur.flatten(1).unsqueeze(0).permute(0, 2, 1)
                    cur_slots = slot_encoder.encode(tok_cur)

                    for tt in range(1, T_use):
                        ps = dynamics(cur_slots)
                        d = residual(tok_cur, cur_slots, ps)
                        L_next = (tok_cur + d)[0].permute(1, 0).reshape(C, H, W)
                        L_out = projector(L_next.unsqueeze(0))[0] if (use_proj and projector) else L_next
                        px = decode_single_frame(vae, L_out, scaling_factor)
                        pred_frames.append(px)
                        tok_cur = L_next.flatten(1).unsqueeze(0).permute(0, 2, 1)
                        cur_slots = ps

                n_cf = min(len(pred_frames), T_clip)
                if n_cf < 4:
                    continue
                pred_clip = F.interpolate(torch.stack(pred_frames[:n_cf]), size=(112, 112), mode="bilinear")
                pred_in = pred_clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)
                pred_in = (pred_in - r3d_mean) / r3d_std
                feats_fake.append(r3d(pred_in).cpu())

        if len(feats_fake) >= 10:
            feats_f = torch.cat(feats_fake).numpy()
            mu_f = feats_f.mean(0)
            sigma_f = np.cov(feats_f, rowvar=False) + 1e-6 * np.eye(feats_f.shape[1])
            diff = mu_r - mu_f
            covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fvd_score = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))
            log(f"    FVD: {fvd_score:.2f} ({len(feats_fake)} clips, {time.time() - t_start:.0f}s)")
            fvd_results[method] = {"score": fvd_score, "n_clips": len(feats_fake)}
        else:
            log(f"    FVD: too few clips ({len(feats_fake)})")
            fvd_results[method] = {"error": f"too few clips: {len(feats_fake)}"}

    results["fvd_extended"] = fvd_results
    with open(args.output, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)

    # ================================================================
    # PART 4: INFERENCE SPEED
    # ================================================================
    log("\n" + "=" * 60)
    log("[6/6] PART 4: Inference Speed")
    log("=" * 60)

    speed_results = {}
    n_warmup, n_measure, n_vae_measure = 10, 100, 20

    latent = np.load(val_files[0]).astype(np.float32)
    C, T, H, W = latent.shape
    lt = torch.from_numpy(latent[:, 0, :, :]).to(device)

    # NeuroCodec speed
    log("  NeuroCodec...")
    for _ in range(n_warmup):
        _ = predict_nc(lt)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_measure):
        _ = predict_nc(lt)
    torch.cuda.synchronize()
    nc_lat = (time.time() - t0) / n_measure

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_vae_measure):
        out = predict_nc(lt)
        _ = decode_single_frame(vae, out, scaling_factor)
    torch.cuda.synchronize()
    nc_full = (time.time() - t0) / n_vae_measure

    speed_results["neurocodec"] = {
        "latent_ms": round(nc_lat * 1000, 2),
        "full_ms": round(nc_full * 1000, 2),
        "latent_fps": round(1 / nc_lat, 1),
        "full_fps": round(1 / nc_full, 2),
    }
    log(f"    Latent: {nc_lat * 1000:.1f}ms ({1 / nc_lat:.0f} FPS)")
    log(f"    + VAE decode: {nc_full * 1000:.1f}ms ({1 / nc_full:.1f} FPS)")

    # SimVP speed
    if simvp is not None:
        log("  SimVP...")
        window = torch.from_numpy(
            latent[:, :4, :, :].transpose(1, 0, 2, 3).copy()
        ).unsqueeze(0).to(device)
        for _ in range(n_warmup):
            _ = simvp(window)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_measure):
            _ = simvp(window)
        torch.cuda.synchronize()
        sv_lat = (time.time() - t0) / n_measure

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_vae_measure):
            out = simvp(window)[0]
            _ = decode_single_frame(vae, out, scaling_factor)
        torch.cuda.synchronize()
        sv_full = (time.time() - t0) / n_vae_measure

        speed_results["simvp"] = {
            "latent_ms": round(sv_lat * 1000, 2),
            "full_ms": round(sv_full * 1000, 2),
            "latent_fps": round(1 / sv_lat, 1),
            "full_fps": round(1 / sv_full, 2),
        }
        log(f"    Latent: {sv_lat * 1000:.1f}ms ({1 / sv_lat:.0f} FPS)")
        log(f"    + VAE decode: {sv_full * 1000:.1f}ms ({1 / sv_full:.1f} FPS)")

    # Model parameters
    speed_results["params"] = {
        "neurocodec_all": (sum(p.numel() for p in slot_encoder.parameters()) +
                           sum(p.numel() for p in dynamics.parameters()) +
                           sum(p.numel() for p in residual.parameters()) +
                           (sum(p.numel() for p in projector.parameters()) if projector else 0)),
        "neurocodec_trainable": (sum(p.numel() for p in dynamics.parameters()) +
                                 sum(p.numel() for p in residual.parameters()) +
                                 (sum(p.numel() for p in projector.parameters()) if projector else 0)),
    }
    if simvp is not None:
        speed_results["params"]["simvp"] = sum(p.numel() for p in simvp.parameters())

    results["inference_speed"] = speed_results

    # ── Final save ──
    with open(args.output, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)

    # ── Summary ──
    log("\n" + "=" * 60)
    log("EXTENDED EVALUATION COMPLETE")
    log("=" * 60)

    log("\nRollout (mean across frames):")
    for m, d in rollout_results.items():
        active_ssim = [v for v in d["per_frame_ssim"] if v > 0]
        active_lpips = [v for v in d["per_frame_lpips"] if v > 0]
        log(f"  {m:35s} SSIM={np.mean(active_ssim):.4f}  LPIPS={np.mean(active_lpips):.4f}  "
            f"Stability={d['stability_mse']:.2f}x")

    log("\nFID:")
    for m, d in fid_results.items():
        log(f"  {m:35s} {d['score']:.2f} ({d['n_samples']} samples)")

    log("\nFVD:")
    for m, d in fvd_results.items():
        if "score" in d:
            log(f"  {m:35s} {d['score']:.2f} ({d['n_clips']} clips)")

    log("\nSpeed:")
    for m, d in speed_results.items():
        if m == "params":
            continue
        log(f"  {m:35s} {d['latent_ms']:.1f}ms latent, {d['full_ms']:.1f}ms full")

    log(f"\nResults: {args.output}")


if __name__ == "__main__":
    main()
