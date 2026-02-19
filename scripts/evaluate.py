"""NeuroCodec evaluation pipeline.

Evaluates a trained ResidualDecoderV2 on latent MSE, rollout stability,
and pixel-level metrics (SSIM, LPIPS) via CogVideoX VAE decoding.

Usage:
    python scripts/evaluate.py --data-dir /path/to/data --checkpoint best.pt
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.models import ResidualDecoderV2, DynamicsTransformer


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuroCodec")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dynamics-checkpoint", type=str, default=None)
    parser.add_argument("--n-rollout-videos", type=int, default=50)
    parser.add_argument("--decode-pixels", action="store_true", help="Decode through VAE (slow)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print(f"Loading data from {args.data_dir}...")
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

    N_videos, T, C, H, W = latent_frames.shape
    n_pairs = N_videos * (T - 1)

    latent_t = latent_frames[:, :-1].reshape(n_pairs, C, H, W)
    latent_t1 = latent_frames[:, 1:].reshape(n_pairs, C, H, W)
    slots_t = video_slots[:, :-1].reshape(n_pairs, 64, 128)
    slots_t1 = video_slots[:, 1:].reshape(n_pairs, 64, 128)

    latent_tokens_t = latent_t.flatten(2).permute(0, 2, 1)
    latent_tokens_t1 = latent_t1.flatten(2).permute(0, 2, 1)

    n_val = 200
    n_train = N_videos - n_val
    n_train_pairs = n_train * (T - 1)
    val_idx = torch.arange(n_train_pairs, n_pairs)

    # Boundary labels
    copy_mse = ((latent_t1 - latent_t) ** 2).mean(dim=(1, 2, 3))
    threshold = torch.quantile(copy_mse, 0.85).item()
    boundary = (copy_mse > threshold).float()

    val_easy = val_idx[(boundary[val_idx] == 0).nonzero(as_tuple=True)[0]]
    val_hard = val_idx[(boundary[val_idx] == 1).nonzero(as_tuple=True)[0]]

    # Load model
    model = ResidualDecoderV2().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    # Single-step MSE
    print("\n--- Single-Step Latent MSE ---")
    for name, indices in [("EASY", val_easy), ("HARD", val_hard), ("ALL", val_idx)]:
        copy_m, res_m = [], []
        with torch.no_grad():
            for i in range(0, len(indices), 48):
                idx = indices[i : i + 48]
                lt = latent_tokens_t[idx].to(device)
                target = latent_tokens_t1[idx].to(device)
                st = slots_t[idx].to(device)
                st1 = slots_t1[idx].to(device)
                copy_m.append(F.mse_loss(lt, target).item())
                pred = lt + model(lt, st, st1)
                res_m.append(F.mse_loss(pred, target).item())
        print(f"  {name:5s}: Copy={np.mean(copy_m):.4f}, Residual={np.mean(res_m):.4f}")

    # Rollout (if dynamics checkpoint provided)
    if args.dynamics_checkpoint:
        print("\n--- 8-Frame Rollout ---")
        dynamics = DynamicsTransformer().to(device)
        dyn_state = torch.load(args.dynamics_checkpoint, map_location=device, weights_only=False)
        if isinstance(dyn_state, dict) and "model" in dyn_state:
            dyn_state = dyn_state["model"]
        dynamics.load_state_dict(dyn_state)
        dynamics.eval()

        val_latents = latent_frames[n_train:]
        val_slots = video_slots[n_train:]
        torch.manual_seed(42)
        vid_indices = torch.randperm(n_val)[: args.n_rollout_videos]

        rollout_mse = [[] for _ in range(T)]
        with torch.no_grad():
            for vi in vid_indices:
                vi = vi.item()
                L_cur = val_latents[vi, 0].clone()
                cur_slots = val_slots[vi, 0:1].to(device)
                for t in range(1, T):
                    pred_slots = dynamics(cur_slots)
                    lt_tok = L_cur.flatten(1).unsqueeze(0).permute(0, 2, 1).to(device)
                    delta = model(lt_tok, cur_slots, pred_slots)
                    L_cur = (lt_tok + delta)[0].permute(1, 0).reshape(C, H, W).cpu()
                    mse = F.mse_loss(L_cur, val_latents[vi, t]).item()
                    rollout_mse[t].append(mse)
                    cur_slots = pred_slots

        for t in range(1, T):
            print(f"  Frame {t}: MSE={np.mean(rollout_mse[t]):.4f}")

        stability = np.mean(rollout_mse[T - 1]) / np.mean(rollout_mse[1])
        print(f"  Stability (frame{T-1}/frame1): {stability:.2f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
