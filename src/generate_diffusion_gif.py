"""Train slot-conditioned diffusion decoder and generate comparison GIFs.

Trains a small UNet diffusion model on SSv2 latent data, then generates
side-by-side animated comparison GIFs showing:
    GT | Diffusion (Ours) | Residual+Projector | Copy

This demonstrates that diffusion decoding produces sharper results than
the residual decoder by generating on-manifold latents.

Usage:
    python src/generate_diffusion_gif.py \
        --slot-ckpt checkpoints/ssv2/slot_encoder_best.pt \
        --dynamics-ckpt checkpoints/ssv2/dynamics_best.pt \
        --residual-ckpt checkpoints/ssv2/residual_spectral_best.pt \
        [--projector-ckpt checkpoints/ssv2/manifold_projector_best.pt] \
        [--latent-dir data/ssv2_latents] \
        [--n-train-videos 100] [--steps 10000] [--n-gifs 5]

Designed to run in ~2-3h on A100-40GB after the SSv2 benchmark pipeline.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from models import (
    SlotLatentAutoencoderV2, DynamicsTransformer,
    ResidualDecoderV2, ManifoldProjector,
)
from vae_utils import load_cogvideox_vae, decode_single_frame
from diffusion_phase0 import (
    SlotConditionedUNet, linear_beta_schedule, get_diffusion_params,
    q_sample, ddim_sample,
    DIFFUSION_STEPS, LATENT_C, LATENT_H, LATENT_W,
    N_SLOTS, SLOT_DIM,
)

RESULTS_DIR = Path("results/demo")
DIFF_CKPT_DIR = Path("checkpoints/diffusion_phase0")
RESULTS_JSON_DIR = Path("results/json")

BATCH_SIZE = 32
LR = 1e-4
GRAD_CLIP = 1.0
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================
# DATASET
# ================================================================

class LatentSlotDataset(Dataset):
    """Paired (latent_frame, slots) for diffusion training."""

    def __init__(self, latent_files, slot_encoder, device):
        self.frames = []
        self.slots = []

        slot_encoder.eval()
        with torch.no_grad():
            for fi, f in enumerate(latent_files):
                latent = np.load(f).astype(np.float32)  # [16, T, 32, 32]
                C, T, H, W = latent.shape
                for t in range(T):
                    frame = latent[:, t, :, :]
                    tokens = frame.reshape(C, H * W).T  # [1024, 16]
                    tok = torch.from_numpy(tokens).unsqueeze(0).to(device)
                    s = slot_encoder.encode(tok)
                    self.frames.append(torch.from_numpy(frame.copy()))
                    self.slots.append(s.cpu().squeeze(0))

                if (fi + 1) % 20 == 0:
                    log(f"    Loaded {fi+1}/{len(latent_files)} videos "
                        f"({len(self.frames)} frames)")

        log(f"    Dataset: {len(self.frames)} frames from "
            f"{len(latent_files)} videos")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.slots[idx]


# ================================================================
# DIFFUSION TRAINING
# ================================================================

def train_diffusion(args, slot_encoder, device):
    """Train SlotConditionedUNet on SSv2 latent data."""
    DIFF_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = DIFF_CKPT_DIR / "diffusion_ssv2_best.pt"

    if ckpt_path.exists() and not args.retrain:
        log(f"  Checkpoint exists: {ckpt_path}")
        model = SlotConditionedUNet().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=False))
        model.eval()
        return model

    log("=" * 60)
    log("TRAINING SLOT-CONDITIONED DIFFUSION UNET")
    log("=" * 60)

    # Select training files (same val split as eval_ssv2_pixels.py)
    latent_dir = Path(args.latent_dir)
    all_files = sorted(latent_dir.glob("*.npy"))
    n_all = len(all_files)

    indices = list(range(n_all))
    random.seed(SEED)
    random.shuffle(indices)
    n_val = int(n_all * 0.15)
    train_indices = indices[n_val:]

    n_train = min(args.n_train_videos, len(train_indices))
    train_files = [all_files[i] for i in train_indices[:n_train]]
    log(f"  Training videos: {n_train} (from {n_all} total, {n_val} held out)")

    log("  Extracting slots for training data...")
    dataset = LatentSlotDataset(train_files, slot_encoder, device)

    train_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)

    model = SlotConditionedUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  UNet: {n_params:,} params ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    betas = linear_beta_schedule(DIFFUSION_STEPS)
    dp = get_diffusion_params(betas)
    for k, v in dp.items():
        dp[k] = v.to(device)

    total_steps = args.steps
    warmup = min(500, total_steps // 10)

    def lr_fn(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    best_loss = float("inf")
    step = 0
    t_start = time.time()

    while step < total_steps:
        model.train()
        for frames, slots in train_dl:
            if step >= total_steps:
                break

            frames = frames.to(device)
            slots = slots.to(device)

            t = torch.randint(0, DIFFUSION_STEPS, (frames.shape[0],),
                              device=device)
            noise = torch.randn_like(frames)
            x_noisy = q_sample(frames, t, noise,
                               dp["sqrt_alphas_cumprod"],
                               dp["sqrt_one_minus_alphas_cumprod"])

            pred_noise = model(x_noisy, t, slots)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 500 == 0:
                elapsed = time.time() - t_start
                log(f"  Step {step:5d}/{total_steps} | "
                    f"Loss: {loss.item():.6f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed/60:.1f}min")

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), ckpt_path)

    # Final save if best
    if loss.item() < best_loss:
        torch.save(model.state_dict(), ckpt_path)

    elapsed = time.time() - t_start
    log(f"  Training complete: {elapsed/60:.1f}min, best loss: {best_loss:.6f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=False))
    model.eval()
    return model


# ================================================================
# GIF GENERATION
# ================================================================

def generate_gifs(args, diffusion_model, slot_encoder, dynamics, residual,
                  projector, vae, scaling_factor, device):
    """Generate side-by-side comparison GIFs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("GENERATING COMPARISON GIFS")
    log("=" * 60)

    betas = linear_beta_schedule(DIFFUSION_STEPS)
    dp = get_diffusion_params(betas)

    # Select test videos from validation set
    latent_dir = Path(args.latent_dir)
    all_files = sorted(latent_dir.glob("*.npy"))
    n_all = len(all_files)

    indices = list(range(n_all))
    random.seed(SEED)
    random.shuffle(indices)
    n_val = int(n_all * 0.15)
    val_indices = indices[:n_val]

    n_context = args.n_context
    n_predict = args.n_predict
    min_frames = n_context + n_predict

    # Find videos with enough temporal frames
    test_files = []
    for idx in val_indices:
        lat = np.load(all_files[idx]).astype(np.float32)
        if lat.shape[1] >= min_frames:
            test_files.append(all_files[idx])
        if len(test_files) >= args.n_gifs:
            break

    log(f"  Selected {len(test_files)} videos "
        f"({n_context} context + {n_predict} predicted frames)")

    gif_metrics = []

    for gif_i, fpath in enumerate(test_files):
        log(f"\n  --- GIF {gif_i+1}/{len(test_files)}: {fpath.stem} ---")

        latent = np.load(fpath).astype(np.float32)  # [16, T, 32, 32]
        C, T, H, W = latent.shape
        T_use = min(T, n_context + n_predict)

        gt_pixels = []
        diff_pixels = []
        res_pixels = []
        copy_pixels = []

        with torch.no_grad():
            # ── Decode GT frames ──
            for t in range(T_use):
                lt = torch.from_numpy(latent[:, t, :, :]).to(device)
                px = decode_single_frame(vae, lt, scaling_factor)
                gt_pixels.append(px.cpu())
            log(f"    Decoded {T_use} GT frames")

            # ── Context frames (identical for all columns) ──
            for t in range(n_context):
                diff_pixels.append(gt_pixels[t])
                res_pixels.append(gt_pixels[t])
                copy_pixels.append(gt_pixels[t])

            # ── Shared: dynamics slot chain ──
            # Both paths use the same predicted slots for fair comparison
            L_ctx = torch.from_numpy(
                latent[:, n_context - 1, :, :]).to(device)
            tok_ctx = L_ctx.flatten(1).unsqueeze(0).permute(0, 2, 1)
            S_ctx = slot_encoder.encode(tok_ctx)

            pred_slot_chain = [S_ctx]
            cur_s = S_ctx
            for t in range(n_predict):
                cur_s = dynamics(cur_s)
                pred_slot_chain.append(cur_s)

            # ── Residual path (autoregressive, error compounds) ──
            L_cur = L_ctx.clone()
            tok_cur = tok_ctx.clone()
            cur_slots_r = pred_slot_chain[0]

            for step in range(n_predict):
                next_slots = pred_slot_chain[step + 1]
                delta = residual(tok_cur, cur_slots_r, next_slots)
                L_next_raw = (tok_cur + delta)[0].permute(1, 0).reshape(
                    C, H, W)

                L_next_out = L_next_raw
                if projector is not None:
                    L_next_out = projector(L_next_raw.unsqueeze(0))[0]

                px = decode_single_frame(vae, L_next_out, scaling_factor)
                res_pixels.append(px.cpu())

                # Decoupled feedback: feed back raw prediction
                tok_cur = L_next_raw.flatten(1).unsqueeze(0).permute(
                    0, 2, 1)
                cur_slots_r = next_slots

            log(f"    Generated {n_predict} residual frames")

            # ── Diffusion path (independent per frame, no error compounding) ──
            for step in range(n_predict):
                next_slots = pred_slot_chain[step + 1]
                L_diff = ddim_sample(
                    diffusion_model, next_slots, dp,
                    n_steps=50, device=device,
                )[0]  # [16, 32, 32]

                px = decode_single_frame(vae, L_diff, scaling_factor)
                diff_pixels.append(px.cpu())

            log(f"    Generated {n_predict} diffusion frames")

            # ── Copy baseline ──
            copy_frame = gt_pixels[n_context - 1]
            for _ in range(n_predict):
                copy_pixels.append(copy_frame)

            # ── Compute per-frame SSIM for metrics ──
            frame_ssims = {"diffusion": [], "residual": [], "copy": []}
            for step in range(n_predict):
                gt_f = gt_pixels[n_context + step]
                for name, pred_list in [("diffusion", diff_pixels),
                                        ("residual", res_pixels),
                                        ("copy", copy_pixels)]:
                    pred_f = pred_list[n_context + step]
                    a = gt_f.numpy().transpose(1, 2, 0)
                    b = pred_f.numpy().transpose(1, 2, 0)
                    from skimage.metrics import structural_similarity
                    ssim = structural_similarity(
                        a, b, channel_axis=2, data_range=1.0)
                    frame_ssims[name].append(ssim)

            log(f"    SSIM (mean over {n_predict} frames):")
            for name, vals in frame_ssims.items():
                log(f"      {name}: {np.mean(vals):.4f}")

            gif_metrics.append({
                "video_id": fpath.stem,
                "frame_ssims": {k: [float(v) for v in vs]
                                for k, vs in frame_ssims.items()},
            })

        # ── Build GIF ──
        build_comparison_gif(
            gt_pixels, diff_pixels, res_pixels, copy_pixels,
            n_context, gif_i, fpath.stem,
        )

    # Save metrics
    RESULTS_JSON_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_JSON_DIR / "diffusion_gif_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(gif_metrics, f, indent=2)
    log(f"\nMetrics saved to {metrics_path}")
    log(f"GIFs saved to {RESULTS_DIR}/")


def build_comparison_gif(gt_pixels, diff_pixels, res_pixels, copy_pixels,
                         n_context, gif_index, video_id):
    """Build animated side-by-side comparison GIF."""
    n_frames = len(gt_pixels)

    def to_pil(tensor):
        arr = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255
               ).astype(np.uint8)
        return Image.fromarray(arr)

    sample = to_pil(gt_pixels[0])
    fw, fh = sample.size

    pad = 4
    label_h = 30
    indicator_h = 22
    total_w = fw * 4 + pad * 5
    total_h = fh + label_h + indicator_h + pad * 3

    labels = ["Ground Truth", "Diffusion (Ours)", "Residual", "Copy"]
    label_colors = [
        (255, 255, 255),
        (0, 220, 100),
        (220, 170, 30),
        (160, 160, 160),
    ]

    # Try to load a nicer font
    font = None
    font_small = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 13)
            font_small = ImageFont.truetype(
                font_path.replace("-Bold", ""), 11)
            break
    if font is None:
        font = ImageFont.load_default()
        font_small = font

    pil_frames = []
    for t in range(n_frames):
        canvas = Image.new("RGB", (total_w, total_h), (25, 25, 30))
        draw = ImageDraw.Draw(canvas)

        # Column labels
        for col, (label, color) in enumerate(zip(labels, label_colors)):
            x = pad + col * (fw + pad)
            # Center text
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
            except AttributeError:
                tw = len(label) * 7
            draw.text((x + (fw - tw) // 2, pad // 2 + 2),
                      label, fill=color, font=font)

        # Phase indicator
        if t < n_context:
            phase_text = f"CONTEXT (t={t})"
            phase_color = (100, 180, 255)
        else:
            phase_text = f"PREDICTED (t+{t - n_context + 1})"
            phase_color = (255, 110, 110)

        try:
            bbox = draw.textbbox((0, 0), phase_text, font=font_small)
            ptw = bbox[2] - bbox[0]
        except AttributeError:
            ptw = len(phase_text) * 6
        draw.text(((total_w - ptw) // 2, label_h + pad),
                  phase_text, fill=phase_color, font=font_small)

        # Frame images
        y_off = label_h + indicator_h + pad * 2
        for col, frames in enumerate(
            [gt_pixels, diff_pixels, res_pixels, copy_pixels]
        ):
            if t < len(frames):
                img = to_pil(frames[t])
            else:
                img = Image.new("RGB", (fw, fh), (0, 0, 0))
            x = pad + col * (fw + pad)
            canvas.paste(img, (x, y_off))

        pil_frames.append(canvas)

    # Save animated GIF
    out_path = RESULTS_DIR / f"diffusion_comparison_{gif_index:02d}_{video_id}.gif"

    # Timing: show context frames faster, prediction frames slower
    durations = []
    for t in range(n_frames):
        if t < n_context:
            durations.append(400)   # 400ms for context
        else:
            durations.append(600)   # 600ms for predicted (more time to compare)
    # Pause on last frame
    durations[-1] = 1500

    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=durations, loop=0,
    )
    log(f"    Saved: {out_path} ({total_w}x{total_h}, {n_frames} frames)")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train diffusion decoder + generate comparison GIFs")
    parser.add_argument("--slot-ckpt", type=str, required=True)
    parser.add_argument("--dynamics-ckpt", type=str, required=True)
    parser.add_argument("--residual-ckpt", type=str, required=True)
    parser.add_argument("--projector-ckpt", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default="data/ssv2_latents")
    parser.add_argument("--n-train-videos", type=int, default=100,
                        help="Videos for diffusion training")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Diffusion training steps")
    parser.add_argument("--n-gifs", type=int, default=5,
                        help="Number of comparison GIFs to generate")
    parser.add_argument("--n-context", type=int, default=3,
                        help="Context frames shown before prediction")
    parser.add_argument("--n-predict", type=int, default=6,
                        help="Predicted frames shown after context")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if checkpoint exists")
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(SEED)

    log("=" * 60)
    log("DIFFUSION GIF GENERATOR")
    log("=" * 60)

    # ── 1. Load models ──
    log("\n[1/4] Loading models...")

    slot_encoder = SlotLatentAutoencoderV2(
        n_slots=64, slot_dim=128, input_dim=16, n_tokens=1024, n_iter=5,
    ).to(device)
    slot_encoder.load_state_dict(torch.load(
        args.slot_ckpt, map_location=device, weights_only=False))
    slot_encoder.eval()
    log(f"  Slot encoder: loaded")

    dynamics = DynamicsTransformer(
        n_tokens=64, token_dim=128, d_model=128, n_heads=4, n_layers=2,
    ).to(device)
    dynamics.load_state_dict(torch.load(
        args.dynamics_ckpt, map_location=device, weights_only=False))
    dynamics.eval()
    log(f"  Dynamics: loaded")

    residual = ResidualDecoderV2().to(device)
    residual.load_state_dict(torch.load(
        args.residual_ckpt, map_location=device, weights_only=False))
    residual.eval()
    log(f"  Residual decoder: loaded")

    projector = None
    if args.projector_ckpt and os.path.exists(args.projector_ckpt):
        projector = ManifoldProjector().to(device)
        projector.load_state_dict(torch.load(
            args.projector_ckpt, map_location=device, weights_only=False))
        projector.eval()
        log(f"  ManifoldProjector: loaded")

    # ── 2. Train diffusion ──
    log("\n[2/4] Diffusion model...")
    diffusion_model = train_diffusion(args, slot_encoder, device)

    # Free slot encoder from GPU during VAE loading
    slot_encoder_cpu = slot_encoder.cpu()
    torch.cuda.empty_cache()

    # ── 3. Load VAE ──
    log("\n[3/4] Loading CogVideoX VAE...")
    vae, scaling_factor = load_cogvideox_vae(device)
    log(f"  VAE: loaded")

    # Move slot encoder back
    slot_encoder = slot_encoder_cpu.to(device)

    # ── 4. Generate GIFs ──
    log("\n[4/4] Generating comparison GIFs...")
    generate_gifs(
        args, diffusion_model, slot_encoder, dynamics, residual,
        projector, vae, scaling_factor, device,
    )

    log("\n" + "=" * 60)
    log("DONE")
    log("=" * 60)
    log(f"GIFs: {RESULTS_DIR}/diffusion_comparison_*.gif")
    log(f"Metrics: {RESULTS_JSON_DIR}/diffusion_gif_metrics.json")


if __name__ == "__main__":
    main()
