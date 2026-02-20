#!/usr/bin/env python3
"""Evaluate ManifoldProjector: single-step metrics + rollout stability.

Compares three configurations:
1. Baseline (no projector)
2. With ManifoldProjector
3. Copy baseline (L_{t+1} = L_t)

Metrics: MSE, Spectral Loss, Variance Ratio, LPIPS (optional), Rollout Stability.

GPU time: ~15-20 min on A100 (with LPIPS), ~5 min without.

Usage:
    python scripts/manifold_eval.py \
        --data-dir ~/ \
        --res-checkpoint ~/residual_v2_spectral_beta0.01_best.pt \
        --dynamics-checkpoint ~/dynamics_best_d2.pt \
        --slot-checkpoint ~/slot_v2_64slots_2k_best.pt \
        --projector-checkpoint ~/manifold_projector_best.pt \
        --output results/json/manifold_eval.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import ResidualDecoderV2, DynamicsTransformer, ManifoldProjector
from src.losses import spectral_loss, variance_loss


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, torch.Tensor): return obj.cpu().numpy().tolist()
        return super().default(obj)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Slot Attention (must match gap3 checkpoint) ──
class SlotAttentionV2(nn.Module):
    def __init__(self, n_slots, slot_dim, input_dim, n_iter=5, hidden_dim=128):
        super().__init__()
        self.n_slots = n_slots; self.n_iter = n_iter; self.slot_dim = slot_dim
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(input_dim, slot_dim)
        self.to_v = nn.Linear(input_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_out = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, slot_dim)
        )
        self.scale = slot_dim ** -0.5

    def forward(self, x):
        B, N_tok, _ = x.shape
        slots = (
            self.slot_mu.expand(B, self.n_slots, -1) +
            self.slot_log_sigma.exp().expand(B, self.n_slots, -1) *
            torch.randn(B, self.n_slots, self.slot_dim, device=x.device)
        )
        x = self.norm_input(x)
        k = self.to_k(x); v = self.to_v(x)
        for _ in range(self.n_iter):
            sp = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            attn = (torch.einsum('bsd,bnd->bsn', q, k) * self.scale).softmax(dim=1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)
            slots = self.gru(
                torch.einsum('bsn,bnd->bsd', attn, v).reshape(-1, self.slot_dim),
                sp.reshape(-1, self.slot_dim)
            ).reshape(B, self.n_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_out(slots))
        return slots


class SlotLatentAutoencoderV2(nn.Module):
    def __init__(self, n_slots=64, slot_dim=128, input_dim=16, n_tokens=1024, n_iter=5):
        super().__init__()
        self.n_tokens = n_tokens; self.slot_dim = slot_dim
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, input_dim) * 0.02)
        self.encoder_norm = nn.LayerNorm(input_dim)
        self.encoder_proj = nn.Sequential(
            nn.Linear(input_dim, slot_dim), nn.GELU(), nn.Linear(slot_dim, slot_dim)
        )
        self.slot_attention = SlotAttentionV2(
            n_slots=n_slots, slot_dim=slot_dim, input_dim=slot_dim, n_iter=n_iter
        )
        self.decoder_pos = nn.Parameter(torch.randn(1, n_tokens, slot_dim) * 0.02)
        self.cross_norm_q1 = nn.LayerNorm(slot_dim)
        self.cross_norm_kv1 = nn.LayerNorm(slot_dim)
        self.cross_q1 = nn.Linear(slot_dim, slot_dim)
        self.cross_k1 = nn.Linear(slot_dim, slot_dim)
        self.cross_v1 = nn.Linear(slot_dim, slot_dim)
        self.decoder_mlp = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim*2),
            nn.GELU(), nn.Linear(slot_dim*2, slot_dim)
        )
        self.cross_norm_q2 = nn.LayerNorm(slot_dim)
        self.cross_norm_kv2 = nn.LayerNorm(slot_dim)
        self.cross_q2 = nn.Linear(slot_dim, slot_dim)
        self.cross_k2 = nn.Linear(slot_dim, slot_dim)
        self.cross_v2 = nn.Linear(slot_dim, slot_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim),
            nn.GELU(), nn.Linear(slot_dim, input_dim)
        )
        self.scale = slot_dim ** -0.5

    def encode(self, x):
        return self.slot_attention(self.encoder_proj(self.encoder_norm(x + self.pos_embed)))


def hungarian_align(slots_a, slots_b):
    from scipy.optimize import linear_sum_assignment
    a_norm = F.normalize(slots_a, dim=-1)
    b_norm = F.normalize(slots_b, dim=-1)
    cost = 1 - torch.mm(a_norm, b_norm.t())
    _, col_ind = linear_sum_assignment(cost.cpu().numpy())
    return slots_b[col_ind]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--res-checkpoint", type=str, required=True)
    parser.add_argument("--dynamics-checkpoint", type=str, required=True)
    parser.add_argument("--slot-checkpoint", type=str, required=True)
    parser.add_argument("--projector-checkpoint", type=str, required=True)
    parser.add_argument("--n-videos", type=int, default=50)
    parser.add_argument("--skip-lpips", action="store_true", help="Skip LPIPS (saves ~10 min)")
    parser.add_argument("--output", type=str, default="results/json/manifold_eval.json")
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
    video_slots_gt = torch.load(
        os.path.join(args.data_dir, "aligned_video_slots.pt"),
        map_location="cpu", weights_only=False,
    )

    if latents_raw.shape[1] == 16 and latents_raw.shape[2] == 9:
        latent_frames = latents_raw.permute(0, 2, 1, 3, 4)
    else:
        latent_frames = latents_raw
    del latents_raw

    N, T, C, H, W = latent_frames.shape
    n_val = 200
    val_latents = latent_frames[-n_val:]
    val_slots_gt = video_slots_gt[-n_val:]
    n_test = min(args.n_videos, n_val)
    log(f"Testing on {n_test} validation videos")

    # ── Load Models ──
    log("Loading models...")
    res_decoder = ResidualDecoderV2().to(device)
    state = torch.load(args.res_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state: state = state["model"]
    res_decoder.load_state_dict(state)
    res_decoder.eval()

    dynamics = DynamicsTransformer().to(device)
    dyn_state = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(dyn_state, dict) and "model" in dyn_state: dyn_state = dyn_state["model"]
    dynamics.load_state_dict(dyn_state)
    dynamics.eval()

    projector = ManifoldProjector().to(device)
    proj_state = torch.load(args.projector_checkpoint, map_location="cpu", weights_only=False)
    projector.load_state_dict(proj_state)
    projector.eval()
    log(f"  ManifoldProjector: {sum(p.numel() for p in projector.parameters()):,} params")

    slot_encoder = SlotLatentAutoencoderV2(
        n_slots=64, slot_dim=128, input_dim=16, n_tokens=1024
    ).to(device)
    slot_state = torch.load(args.slot_checkpoint, map_location="cpu", weights_only=False)
    slot_encoder.load_state_dict(slot_state)
    slot_encoder.eval()

    # Optional LPIPS
    lpips_fn = None
    if not args.skip_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex').to(device)
            log("  LPIPS loaded")
        except ImportError:
            log("  LPIPS not available, skipping")

    # Optional VAE for pixel-space evaluation
    vae = None
    if lpips_fn is not None:
        try:
            from diffusers import AutoencoderKLCogVideoX
            vae = AutoencoderKLCogVideoX.from_pretrained(
                "THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.float16
            ).to(device)
            vae.eval()
            log("  CogVideoX VAE loaded")
        except Exception as e:
            log(f"  VAE loading failed: {e}, skipping LPIPS")
            lpips_fn = None

    # ================================================================
    # EXPERIMENT 1: Single-step quality (with dynamics-predicted slots)
    # ================================================================
    log(f"\n{'='*60}")
    log("EXPERIMENT 1: Single-step quality comparison")
    log(f"{'='*60}")

    metrics = {
        "baseline": {"mse": [], "spectral": [], "var_ratio": []},
        "projected": {"mse": [], "spectral": [], "var_ratio": []},
        "copy": {"mse": []},
    }
    if lpips_fn:
        metrics["baseline"]["lpips"] = []
        metrics["projected"]["lpips"] = []
        metrics["copy"]["lpips"] = []

    torch.manual_seed(42)

    for vid_idx in range(n_test):
        if vid_idx % 10 == 0:
            log(f"  Video {vid_idx+1}/{n_test}")

        for t in range(T - 1):
            lt = val_latents[vid_idx, t].to(device)            # [C, H, W]
            lt1_gt = val_latents[vid_idx, t + 1].to(device)    # [C, H, W]
            lt_tok = lt.flatten(1).unsqueeze(0).permute(0, 2, 1)   # [1, 1024, 16]
            gt_spatial = lt1_gt.unsqueeze(0)                       # [1, C, H, W]

            st = val_slots_gt[vid_idx, t:t+1].to(device)
            st1_dyn = dynamics(st)

            with torch.no_grad():
                # Baseline: ResidualDecoder only
                delta = res_decoder(lt_tok, st, st1_dyn)
                z_pred = (lt_tok + delta).permute(0, 2, 1).reshape(1, C, H, W)

                mse_base = F.mse_loss(z_pred, gt_spatial).item()
                spec_base = spectral_loss(z_pred, gt_spatial).item()
                metrics["baseline"]["mse"].append(mse_base)
                metrics["baseline"]["spectral"].append(spec_base)

                # Projected: ResidualDecoder + ManifoldProjector
                z_proj = projector(z_pred)
                mse_proj = F.mse_loss(z_proj, gt_spatial).item()
                spec_proj = spectral_loss(z_proj, gt_spatial).item()
                metrics["projected"]["mse"].append(mse_proj)
                metrics["projected"]["spectral"].append(spec_proj)

                # Copy baseline
                copy_spatial = lt.unsqueeze(0)
                mse_copy = F.mse_loss(copy_spatial, gt_spatial).item()
                metrics["copy"]["mse"].append(mse_copy)

    # Compute variance ratios on batched data
    log("  Computing variance ratios...")
    with torch.no_grad():
        n_var_test = min(500, n_test * (T - 1))
        var_preds = []
        var_projs = []
        var_gts = []
        count = 0
        for vid_idx in range(n_test):
            for t in range(T - 1):
                if count >= n_var_test:
                    break
                lt = val_latents[vid_idx, t].to(device)
                lt_tok = lt.flatten(1).unsqueeze(0).permute(0, 2, 1)
                st = val_slots_gt[vid_idx, t:t+1].to(device)
                st1_dyn = dynamics(st)
                with torch.no_grad():
                    delta = res_decoder(lt_tok, st, st1_dyn)
                    z_pred = (lt_tok + delta).permute(0, 2, 1).reshape(1, C, H, W)
                    z_proj = projector(z_pred)
                var_preds.append(z_pred.cpu())
                var_projs.append(z_proj.cpu())
                var_gts.append(val_latents[vid_idx, t + 1].unsqueeze(0))
                count += 1
            if count >= n_var_test:
                break

        var_preds = torch.cat(var_preds)
        var_projs = torch.cat(var_projs)
        var_gts = torch.cat(var_gts)

        pred_std = var_preds.std(dim=(0, 2, 3))
        proj_std = var_projs.std(dim=(0, 2, 3))
        gt_std = var_gts.std(dim=(0, 2, 3))

        ratio_dev_base = ((pred_std / (gt_std + 1e-8)) - 1).abs().mean().item()
        ratio_dev_proj = ((proj_std / (gt_std + 1e-8)) - 1).abs().mean().item()

    log(f"\n  Single-step results:")
    log(f"  {'Config':<15s} {'MSE':>10s} {'Spectral':>10s} {'RatioDev':>10s}")
    log(f"  {'-'*50}")
    log(f"  {'Baseline':<15s} {np.mean(metrics['baseline']['mse']):>10.6f} {np.mean(metrics['baseline']['spectral']):>10.4f} {ratio_dev_base:>10.4f}")
    log(f"  {'Projected':<15s} {np.mean(metrics['projected']['mse']):>10.6f} {np.mean(metrics['projected']['spectral']):>10.4f} {ratio_dev_proj:>10.4f}")
    log(f"  {'Copy':<15s} {np.mean(metrics['copy']['mse']):>10.6f} {'N/A':>10s} {'N/A':>10s}")

    mse_improv = (1 - np.mean(metrics['projected']['mse']) / np.mean(metrics['baseline']['mse'])) * 100
    log(f"\n  MSE improvement: {mse_improv:+.1f}%")

    # ================================================================
    # EXPERIMENT 2: 8-frame rollout with projector in the loop
    # ================================================================
    log(f"\n{'='*60}")
    log("EXPERIMENT 2: 8-frame autoregressive rollout")
    log(f"{'='*60}")

    n_rollout = min(30, n_test)
    configs = {
        "baseline_gt_slots": "GT slots, no projector",
        "baseline_online": "Online slots, no projector",
        "projected_online": "Online slots + ManifoldProjector (coupled)",
        "decoupled_online": "Online slots from raw + projected output (decoupled)",
        "copy": "Copy baseline (L_{t+1} = L_t)",
    }

    rollout_results = {}
    for config_name, desc in configs.items():
        log(f"  Running: {config_name} ({desc})")
        per_frame_mse = [[] for _ in range(T)]

        for vid_idx in range(n_rollout):
            L_cur = val_latents[vid_idx, 0].clone()
            cur_slots = val_slots_gt[vid_idx, 0:1].to(device)

            with torch.no_grad():
                for t in range(1, T):
                    if config_name == "copy":
                        L_pred_spatial = L_cur.clone()
                    else:
                        lt_tok = L_cur.flatten(1).unsqueeze(0).permute(0, 2, 1).to(device)

                        if config_name == "baseline_gt_slots":
                            next_slots = val_slots_gt[vid_idx, t:t+1].to(device)
                        else:
                            # Online: use dynamics-predicted slots
                            next_slots = dynamics(cur_slots)

                        delta = res_decoder(lt_tok, cur_slots, next_slots)
                        L_raw_spatial = (lt_tok + delta).permute(0, 2, 1).reshape(1, C, H, W)

                        # Apply projector for output if needed
                        if config_name in ("projected_online", "decoupled_online"):
                            L_proj_spatial = projector(L_raw_spatial)
                        else:
                            L_proj_spatial = L_raw_spatial

                        L_pred_spatial = L_proj_spatial[0]  # [C, H, W]

                    mse = F.mse_loss(L_pred_spatial.cpu(), val_latents[vid_idx, t]).item()
                    per_frame_mse[t].append(mse)

                    # Update L_cur for next step
                    if config_name == "decoupled_online":
                        # DECOUPLED: feed raw (unprojected) latent back
                        L_cur = L_raw_spatial[0].cpu()
                    else:
                        L_cur = L_pred_spatial.cpu()

                    # Update slots for next step
                    if config_name == "baseline_gt_slots":
                        cur_slots = val_slots_gt[vid_idx, t:t+1].to(device)
                    elif config_name != "copy":
                        # Re-extract slots from feedback latent (raw for decoupled)
                        L_tok = L_cur.flatten(1).unsqueeze(0).permute(0, 2, 1).to(device)
                        new_slots = slot_encoder.encode(L_tok)
                        new_slots = hungarian_align(cur_slots[0], new_slots[0]).unsqueeze(0)
                        cur_slots = new_slots

        means = [float(np.mean(per_frame_mse[t])) if per_frame_mse[t] else 0.0 for t in range(T)]
        stability = means[T-1] / means[1] if means[1] > 0 else 0

        rollout_results[config_name] = {
            "per_frame_mse": means,
            "stability": stability,
            "n_videos": n_rollout,
        }
        log(f"    stability={stability:.3f}x, final_mse={means[T-1]:.4f}")

    # ================================================================
    # EXPERIMENT 3: Latency measurement
    # ================================================================
    log(f"\n{'='*60}")
    log("EXPERIMENT 3: Latency overhead")
    log(f"{'='*60}")

    dummy = torch.randn(1, C, H, W, device=device)
    # Warmup
    for _ in range(50):
        _ = projector(dummy)
    torch.cuda.synchronize()

    n_iters = 200
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]

    for i in range(n_iters):
        start_events[i].record()
        _ = projector(dummy)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    latency = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "n_iters": n_iters,
    }
    log(f"  Projector latency: {latency['mean_ms']:.3f} ± {latency['std_ms']:.3f} ms")
    log(f"  (Overhead on 2.51ms pipeline: {latency['mean_ms']/2.51*100:.1f}%)")

    # ── Compile & Save ──
    results = {
        "single_step": {
            config: {
                "mse_mean": float(np.mean(metrics[config]["mse"])),
                "mse_std": float(np.std(metrics[config]["mse"])),
                **({
                    "spectral_mean": float(np.mean(metrics[config]["spectral"])),
                } if "spectral" in metrics[config] else {}),
                "n_pairs": len(metrics[config]["mse"]),
            }
            for config in metrics
        },
        "variance_ratio": {
            "baseline": ratio_dev_base,
            "projected": ratio_dev_proj,
        },
        "rollout": rollout_results,
        "latency": latency,
        "metadata": {
            "n_test_videos": n_test,
            "n_rollout_videos": n_rollout,
            "projector_params": sum(p.numel() for p in projector.parameters()),
            "gpu": torch.cuda.get_device_name(0),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    log(f"\nSaved: {args.output}")

    # ── Summary ──
    log(f"\n{'='*60}")
    log("FINAL SUMMARY")
    log(f"{'='*60}")
    log(f"\nSingle-step MSE improvement: {mse_improv:+.1f}%")
    log(f"Variance ratio dev: {ratio_dev_base:.4f} -> {ratio_dev_proj:.4f}")
    log(f"\nRollout stability (MSE frame8 / frame1):")
    for cfg, res in rollout_results.items():
        log(f"  {cfg:<25s}: {res['stability']:.3f}x")
    log(f"\nProjector latency: {latency['mean_ms']:.3f} ms ({latency['mean_ms']/2.51*100:.1f}% overhead)")

    projected_stab = rollout_results.get("projected_online", {}).get("stability", 0)
    baseline_stab = rollout_results.get("baseline_online", {}).get("stability", 0)
    if projected_stab and baseline_stab:
        stab_improv = (1 - projected_stab / baseline_stab) * 100
        log(f"\nRollout stability improvement (projected vs baseline online): {stab_improv:+.1f}%")


if __name__ == "__main__":
    main()
