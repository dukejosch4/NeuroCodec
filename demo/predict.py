"""NeuroCodec demo: single-step video latent prediction.

Self-contained demo that loads a trained model and predicts the next
frame's latent from the current frame's latent and slots.

Usage:
    python demo/predict.py --checkpoint residual_v2_best.pt --frame-idx 0

Requires pre-computed data (latents + slots).
"""

import argparse
import os

import torch

from src.models import ResidualDecoderV2


def main():
    parser = argparse.ArgumentParser(description="NeuroCodec single-step prediction demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="ResidualDecoderV2 checkpoint")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory with latent/slot data")
    parser.add_argument("--video-idx", type=int, default=0, help="Video index")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index (predicts frame_idx+1)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = ResidualDecoderV2().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded ResidualDecoderV2 ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params)")

    # Load data
    latents = torch.load(
        os.path.join(args.data_dir, "latents_2000videos.pt"),
        map_location="cpu", weights_only=False,
    )
    slots = torch.load(
        os.path.join(args.data_dir, "aligned_video_slots.pt"),
        map_location="cpu", weights_only=False,
    )

    if latents.shape[1] == 16 and latents.shape[2] == 9:
        latents = latents.permute(0, 2, 1, 3, 4)

    v, t = args.video_idx, args.frame_idx
    L_t = latents[v, t]  # [16, 32, 32]
    L_gt = latents[v, t + 1]
    S_t = slots[v, t]  # [64, 128]
    S_t1 = slots[v, t + 1]

    # Predict
    lt_tok = L_t.flatten(1).unsqueeze(0).permute(0, 2, 1).to(device)
    st = S_t.unsqueeze(0).to(device)
    st1 = S_t1.unsqueeze(0).to(device)

    with torch.no_grad():
        delta = model(lt_tok, st, st1)
        L_pred = (lt_tok + delta)[0].permute(1, 0).reshape(16, 32, 32).cpu()

    # Metrics
    copy_mse = ((L_t - L_gt) ** 2).mean().item()
    pred_mse = ((L_pred - L_gt) ** 2).mean().item()
    improvement = (1 - pred_mse / copy_mse) * 100

    print(f"\nVideo {v}, Frame {t} -> {t + 1}:")
    print(f"  Copy MSE:     {copy_mse:.4f}")
    print(f"  Predicted MSE: {pred_mse:.4f}")
    print(f"  Improvement:  {improvement:+.1f}%")
    print(f"  Delta norm:   {delta.norm().item():.4f}")


if __name__ == "__main__":
    main()
