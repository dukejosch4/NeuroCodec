"""NeuroCodec training script.

Trains the ResidualDecoderV2 with MSE + spectral loss (beta=0.01).

Usage:
    python scripts/train.py --data-dir /path/to/latents --epochs 150

Requires:
    - latents_2000videos.pt: CogVideoX latents [N, 16, 9, 32, 32]
    - aligned_video_slots.pt: Slot attention outputs [N, 9, 64, 128]
    - residual_decoder_v2_best.pt: MSE-pretrained checkpoint (for warm start)
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from src.models import ResidualDecoderV2
from src.losses import combined_loss


def main():
    parser = argparse.ArgumentParser(description="Train NeuroCodec ResidualDecoderV2")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with latent/slot data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Warm-start checkpoint path")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--spectral-beta", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

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

    # Normalize shape to [N, T, C, H, W]
    if latents_raw.shape[1] == 16 and latents_raw.shape[2] == 9:
        latent_frames = latents_raw.permute(0, 2, 1, 3, 4)
    else:
        latent_frames = latents_raw

    N_videos, T = latent_frames.shape[0], latent_frames.shape[1]
    n_pairs = N_videos * (T - 1)

    # Flatten to frame pairs
    latent_t = latent_frames[:, :-1].reshape(n_pairs, 16, 32, 32)
    latent_t1 = latent_frames[:, 1:].reshape(n_pairs, 16, 32, 32)
    slots_t = video_slots[:, :-1].reshape(n_pairs, 64, 128)
    slots_t1 = video_slots[:, 1:].reshape(n_pairs, 64, 128)

    latent_tokens_t = latent_t.flatten(2).permute(0, 2, 1)
    delta_tokens = latent_t1.flatten(2).permute(0, 2, 1) - latent_tokens_t

    # Train/val split
    n_val = 200
    n_train = N_videos - n_val
    n_train_pairs = n_train * (T - 1)
    train_idx = torch.arange(n_train_pairs)
    val_idx = torch.arange(n_train_pairs, n_pairs)
    print(f"Train: {len(train_idx)} pairs, Val: {len(val_idx)} pairs")

    # Model
    model = ResidualDecoderV2().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"Warm-started from {args.checkpoint}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * (len(train_idx) // args.batch_size + 1)

    def lr_lambda(step):
        warmup = 200
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    step = 0

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_idx))

        for i in range(0, len(train_idx), args.batch_size):
            idx = train_idx[perm[i : i + args.batch_size]]
            lt = latent_tokens_t[idx].to(device)
            dt = delta_tokens[idx].to(device)
            st = slots_t[idx].to(device)
            st1 = slots_t1[idx].to(device)

            pred = model(lt, st, st1)
            loss, _ = combined_loss(pred, dt, lt, beta=args.spectral_beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

        # Validation
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            val_loss = 0
            n_vb = 0
            with torch.no_grad():
                for i in range(0, len(val_idx), args.batch_size):
                    idx = val_idx[i : i + args.batch_size]
                    lt = latent_tokens_t[idx].to(device)
                    dt = delta_tokens[idx].to(device)
                    st = slots_t[idx].to(device)
                    st1 = slots_t1[idx].to(device)
                    pred = model(lt, st, st1)
                    loss, _ = combined_loss(pred, dt, lt, beta=args.spectral_beta)
                    val_loss += loss.item()
                    n_vb += 1

            val_avg = val_loss / n_vb
            improved = val_avg < best_val
            if improved:
                best_val = val_avg
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "residual_v2_best.pt"),
                )
            print(
                f"Epoch {epoch:3d}: val_loss={val_avg:.6f} "
                f"{'*' if improved else ''}"
            )

    print(f"Done. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
