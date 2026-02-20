#!/usr/bin/env python3
"""Gap 3 Fix: Slot Robustness — slots from predicted vs GT latents.

The paper's circularity concern: slots are pre-computed from GT frames,
but at inference GT frames aren't available. This script tests whether
slot extraction from PREDICTED latents degrades quality significantly.

Experiment design:
  1. Extract slots from GT latents (baseline — current paper setup)
  2. Extract slots from predicted latents (realistic inference scenario)
  3. Add calibrated noise to GT latents, extract slots (sensitivity curve)
  4. Compare residual decoder quality under each slot source

This directly addresses the reviewer concern by showing the pipeline
is robust to slot extraction from imperfect inputs.

GPU time: ~15 minutes on A100.

Usage:
    python scripts/gap3_slot_robustness.py \
        --data-dir ~/ \
        --checkpoint ~/residual_v2_spectral_beta0.01_best.pt \
        --dynamics-checkpoint ~/dynamics_best_d2.pt \
        --slot-checkpoint ~/slot_v2_64slots_2k_best.pt \
        --output results/json/gap3_slot_robustness.json
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
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import ResidualDecoderV2, DynamicsTransformer


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, torch.Tensor): return obj.cpu().numpy().tolist()
        return super().default(obj)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Slot Attention (must match checkpoint architecture) ──
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
        # Decoder parts (needed for checkpoint loading, not used here)
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
    """Align slots_b to match slots_a ordering via Hungarian matching."""
    a_norm = F.normalize(slots_a, dim=-1)
    b_norm = F.normalize(slots_b, dim=-1)
    cost = 1 - torch.mm(a_norm, b_norm.t())
    _, col_ind = linear_sum_assignment(cost.cpu().numpy())
    return slots_b[col_ind]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dynamics-checkpoint", type=str, required=True)
    parser.add_argument("--slot-checkpoint", type=str, required=True)
    parser.add_argument("--n-videos", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/json/gap3_slot_robustness.json")
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

    N_videos, T, C, H, W = latent_frames.shape
    n_val = 200
    val_latents = latent_frames[-n_val:]
    val_slots_gt = video_slots_gt[-n_val:]

    n_test = min(args.n_videos, n_val)
    log(f"Testing on {n_test} validation videos")

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

    # Slot encoder
    log("Loading slot encoder...")
    slot_encoder = SlotLatentAutoencoderV2(
        n_slots=64, slot_dim=128, input_dim=16, n_tokens=1024
    ).to(device)
    slot_state = torch.load(args.slot_checkpoint, map_location="cpu", weights_only=False)
    slot_encoder.load_state_dict(slot_state)
    slot_encoder.eval()
    log("  Slot encoder loaded")

    # ================================================================
    # EXPERIMENT 1: GT slots vs Re-extracted GT slots vs Predicted-latent slots
    # ================================================================
    log(f"\n{'='*60}")
    log("EXPERIMENT 1: Slot source comparison")
    log(f"{'='*60}")

    results_exp1 = {
        "gt_slots": [],           # Pre-computed GT slots (current paper)
        "reextracted_gt": [],     # Fresh extraction from GT latents
        "predicted_slots": [],    # Extraction from predicted latents
    }

    # Slot cosine similarities
    slot_cos_gt_vs_reextract = []
    slot_cos_gt_vs_predicted = []

    torch.manual_seed(42)

    for vid_idx in range(n_test):
        if vid_idx % 10 == 0:
            log(f"  Video {vid_idx+1}/{n_test}")

        for t in range(T - 1):
            lt = val_latents[vid_idx, t].to(device)        # [C, H, W]
            lt1_gt = val_latents[vid_idx, t + 1].to(device)
            lt_tok = lt.flatten(1).unsqueeze(0).permute(0, 2, 1)  # [1, 1024, 16]
            target_tok = lt1_gt.flatten(1).unsqueeze(0).permute(0, 2, 1)

            # -- A: Pre-computed GT slots (paper baseline) --
            st_gt = val_slots_gt[vid_idx, t:t+1].to(device)
            st1_gt = val_slots_gt[vid_idx, t+1:t+2].to(device)

            with torch.no_grad():
                delta_gt = res_decoder(lt_tok, st_gt, st1_gt)
                pred_gt = lt_tok + delta_gt
                mse_gt = F.mse_loss(pred_gt, target_tok).item()
            results_exp1["gt_slots"].append(mse_gt)

            # -- B: Re-extracted from GT latents (fresh slot attention run) --
            with torch.no_grad():
                lt_tok_for_slot = lt.flatten(1).unsqueeze(0).permute(0, 2, 1)
                lt1_tok_for_slot = lt1_gt.flatten(1).unsqueeze(0).permute(0, 2, 1)
                st_re = slot_encoder.encode(lt_tok_for_slot)    # [1, 64, 128]
                st1_re = slot_encoder.encode(lt1_tok_for_slot)
                # Align to GT ordering
                st1_re_aligned = hungarian_align(st_re[0], st1_re[0]).unsqueeze(0)

                delta_re = res_decoder(lt_tok, st_re, st1_re_aligned)
                pred_re = lt_tok + delta_re
                mse_re = F.mse_loss(pred_re, target_tok).item()
            results_exp1["reextracted_gt"].append(mse_re)

            # Slot similarity: GT vs re-extracted
            cos = F.cosine_similarity(st_gt.mean(dim=1), st_re.mean(dim=1)).item()
            slot_cos_gt_vs_reextract.append(cos)

            # -- C: Slots from PREDICTED latents --
            with torch.no_grad():
                # First predict the latent
                st_dyn = dynamics(st_gt)  # Use GT slots for dynamics (fair: this is what paper does)
                delta_first = res_decoder(lt_tok, st_gt, st_dyn)
                lt1_pred = (lt_tok + delta_first).permute(0, 2, 1).reshape(1, C, H, W)

                # Now extract slots from the PREDICTED latent
                lt1_pred_tok = lt1_pred.flatten(2).permute(0, 2, 1)  # [1, 1024, 16]
                st1_from_pred = slot_encoder.encode(lt1_pred_tok)
                st1_from_pred_aligned = hungarian_align(st_gt[0], st1_from_pred[0]).unsqueeze(0)

                # Use these predicted-source slots for residual decoding
                delta_pred = res_decoder(lt_tok, st_gt, st1_from_pred_aligned)
                pred_pred = lt_tok + delta_pred
                mse_pred = F.mse_loss(pred_pred, target_tok).item()
            results_exp1["predicted_slots"].append(mse_pred)

            cos_pred = F.cosine_similarity(st1_gt.mean(dim=1), st1_from_pred.mean(dim=1)).item()
            slot_cos_gt_vs_predicted.append(cos_pred)

    for key in results_exp1:
        vals = results_exp1[key]
        log(f"  {key:20s}: MSE = {np.mean(vals):.6f} +/- {np.std(vals):.6f}")

    log(f"\n  Slot cosine similarity:")
    log(f"    GT vs re-extracted:  {np.mean(slot_cos_gt_vs_reextract):.4f}")
    log(f"    GT vs from-predicted: {np.mean(slot_cos_gt_vs_predicted):.4f}")

    # ================================================================
    # EXPERIMENT 2: Sensitivity — noisy latents → slot quality
    # ================================================================
    log(f"\n{'='*60}")
    log("EXPERIMENT 2: Slot robustness to latent noise")
    log(f"{'='*60}")

    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    n_noise_test = min(20, n_test)
    sensitivity = {}

    latent_std = val_latents[:n_noise_test].float().std().item()
    log(f"  Latent std: {latent_std:.4f}")

    for sigma_rel in noise_levels:
        sigma_abs = sigma_rel * latent_std
        mse_list = []
        cos_list = []

        torch.manual_seed(42)

        with torch.no_grad():
            for vid_idx in range(n_noise_test):
                for t in range(T - 1):
                    lt = val_latents[vid_idx, t].to(device)
                    lt1_gt = val_latents[vid_idx, t + 1].to(device)
                    lt_tok = lt.flatten(1).unsqueeze(0).permute(0, 2, 1)
                    target_tok = lt1_gt.flatten(1).unsqueeze(0).permute(0, 2, 1)

                    st_gt = val_slots_gt[vid_idx, t:t+1].to(device)
                    st1_gt = val_slots_gt[vid_idx, t+1:t+2].to(device)

                    # Add noise to lt1_gt, then extract slots
                    lt1_noisy = lt1_gt + torch.randn_like(lt1_gt) * sigma_abs
                    lt1_noisy_tok = lt1_noisy.flatten(1).unsqueeze(0).permute(0, 2, 1)
                    st1_noisy = slot_encoder.encode(lt1_noisy_tok)
                    st1_noisy_aligned = hungarian_align(st_gt[0], st1_noisy[0]).unsqueeze(0)

                    delta = res_decoder(lt_tok, st_gt, st1_noisy_aligned)
                    pred = lt_tok + delta
                    mse_list.append(F.mse_loss(pred, target_tok).item())

                    cos = F.cosine_similarity(st1_gt.mean(dim=1), st1_noisy.mean(dim=1)).item()
                    cos_list.append(cos)

        sensitivity[f"sigma_{sigma_rel}"] = {
            "sigma_relative": sigma_rel,
            "sigma_absolute": float(sigma_abs),
            "mse_mean": float(np.mean(mse_list)),
            "mse_std": float(np.std(mse_list)),
            "slot_cosine_mean": float(np.mean(cos_list)),
            "n_pairs": len(mse_list),
        }

        log(f"  sigma={sigma_rel:.2f}: MSE={np.mean(mse_list):.6f}  slot_cos={np.mean(cos_list):.4f}")

    # ================================================================
    # EXPERIMENT 3: Full rollout with predicted-source slots
    # ================================================================
    log(f"\n{'='*60}")
    log("EXPERIMENT 3: 8-frame rollout with predicted-source slots")
    log(f"{'='*60}")

    n_rollout = min(30, n_test)
    rollout_configs = {
        "gt_slots": "Use pre-computed GT slots at each step",
        "online_slots": "Re-extract slots from predicted latents at each step",
    }

    rollout_results = {}
    for config_name in rollout_configs:
        per_frame_mse = [[] for _ in range(T)]

        for vid_idx in range(n_rollout):
            L_cur = val_latents[vid_idx, 0].clone()
            cur_slots = val_slots_gt[vid_idx, 0:1].to(device)

            with torch.no_grad():
                for t in range(1, T):
                    lt_tok = L_cur.flatten(1).unsqueeze(0).permute(0, 2, 1).to(device)

                    if config_name == "gt_slots":
                        # Use pre-computed GT slots
                        next_slots = val_slots_gt[vid_idx, t:t+1].to(device)
                    else:
                        # Online: predict, then re-extract slots
                        pred_slots_dyn = dynamics(cur_slots)
                        delta = res_decoder(lt_tok, cur_slots, pred_slots_dyn)
                        L_pred = (lt_tok + delta).permute(0, 2, 1).reshape(1, C, H, W)
                        # Extract slots from predicted latent
                        L_pred_tok = L_pred.flatten(2).permute(0, 2, 1)
                        next_slots = slot_encoder.encode(L_pred_tok)
                        next_slots = hungarian_align(cur_slots[0], next_slots[0]).unsqueeze(0)

                    # Residual decode using chosen slots
                    delta = res_decoder(lt_tok, cur_slots, next_slots)
                    L_cur = (lt_tok + delta)[0].permute(1, 0).reshape(C, H, W).cpu()

                    mse = F.mse_loss(L_cur, val_latents[vid_idx, t]).item()
                    per_frame_mse[t].append(mse)

                    cur_slots = next_slots

        means = [float(np.mean(per_frame_mse[t])) if per_frame_mse[t] else 0.0 for t in range(T)]
        stability = means[T-1] / means[1] if means[1] > 0 else 0

        rollout_results[config_name] = {
            "per_frame_mse": means,
            "stability": stability,
            "n_videos": n_rollout,
        }

        log(f"  {config_name:15s}: stability={stability:.3f}x, final_mse={means[T-1]:.4f}")

    # ── Compile & Save ──
    results = {
        "experiment_1_slot_source": {
            key: {
                "mse_mean": float(np.mean(results_exp1[key])),
                "mse_std": float(np.std(results_exp1[key])),
                "n_pairs": len(results_exp1[key]),
            }
            for key in results_exp1
        },
        "slot_cosine_similarity": {
            "gt_vs_reextracted": {
                "mean": float(np.mean(slot_cos_gt_vs_reextract)),
                "std": float(np.std(slot_cos_gt_vs_reextract)),
            },
            "gt_vs_from_predicted": {
                "mean": float(np.mean(slot_cos_gt_vs_predicted)),
                "std": float(np.std(slot_cos_gt_vs_predicted)),
            },
        },
        "experiment_2_sensitivity": sensitivity,
        "experiment_3_rollout": rollout_results,
        "metadata": {
            "n_test_videos": n_test,
            "latent_std": latent_std,
            "gpu": torch.cuda.get_device_name(0),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    log(f"\nSaved: {args.output}")

    # ── Summary ──
    log(f"\n{'='*60}")
    log("SUMMARY — Key findings for paper")
    log(f"{'='*60}")

    gt_mse = np.mean(results_exp1["gt_slots"])
    pred_mse = np.mean(results_exp1["predicted_slots"])
    degradation = (pred_mse - gt_mse) / gt_mse * 100
    log(f"  Single-step MSE degradation with predicted-source slots: {degradation:+.2f}%")
    log(f"    GT slots:        {gt_mse:.6f}")
    log(f"    Predicted slots: {pred_mse:.6f}")
    log(f"  Slot cosine sim (GT vs predicted-source): {np.mean(slot_cos_gt_vs_predicted):.4f}")

    gt_stab = rollout_results["gt_slots"]["stability"]
    online_stab = rollout_results["online_slots"]["stability"]
    log(f"  Rollout stability:")
    log(f"    GT slots:     {gt_stab:.3f}x")
    log(f"    Online slots: {online_stab:.3f}x")

    if abs(degradation) < 5:
        log(f"\n  CONCLUSION: Slot extraction is robust to prediction errors.")
        log(f"  The circularity concern does NOT materially affect results.")
    else:
        log(f"\n  CONCLUSION: Slot quality degrades by {degradation:.1f}% — discuss in limitations.")


if __name__ == "__main__":
    main()
