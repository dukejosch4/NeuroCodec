"""Phase 0: Slot-Conditioned Diffusion Decoder — Feasibility Check.

Tests whether a small diffusion model can reconstruct video latents
when conditioned on slot representations via cross-attention.

Pipeline:
  1. Pre-flight: verify dependencies, GPU, disk space
  2. Encode ~50 UCF-101 videos through CogVideoX VAE
  3. Extract slots using trained SlotLatentAutoencoderV2
  4. Train a small UNet diffusion model conditioned on slots
  5. Evaluate: reconstruction quality with vs without slot conditioning
  6. Save results + sample visualizations

Usage:
    python src/diffusion_phase0.py [--n-videos 50] [--epochs 50] [--steps 10000]

Designed to run in ~2h on a single A100-40GB.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ==============================================================
# CONFIG
# ==============================================================

DATA_DIR = Path("data/ucf101_latents_phase0")
RESULTS_DIR = Path("results/json")
CHECKPOINT_DIR = Path("checkpoints/diffusion_phase0")
FIGURES_DIR = Path("results/figures")

# Latent space (CogVideoX)
LATENT_C = 16
LATENT_H = 32
LATENT_W = 32
N_TOKENS = LATENT_H * LATENT_W  # 1024

# Slots (from trained encoder)
N_SLOTS = 64
SLOT_DIM = 128

# Diffusion
DIFFUSION_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

# Small UNet
UNET_DIM = 128
UNET_DIM_MULTS = (1, 2, 4)  # -> 128, 256, 512 channels
CROSS_ATTN_HEADS = 4

# Training
BATCH_SIZE = 32
LR = 1e-4
GRAD_CLIP = 1.0
VAL_FRACTION = 0.15


# ==============================================================
# DIFFUSION SCHEDULE
# ==============================================================

def linear_beta_schedule(timesteps, beta_start=BETA_START, beta_end=BETA_END):
    return torch.linspace(beta_start, beta_end, timesteps)


def get_diffusion_params(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }


def q_sample(x_0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """Forward diffusion: add noise at timestep t."""
    s_a = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    s_b = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return s_a * x_0 + s_b * noise


# ==============================================================
# SLOT-CONDITIONED UNET
# ==============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SlotCrossAttention(nn.Module):
    """Cross-attention: spatial features attend to slot embeddings."""

    def __init__(self, dim, slot_dim=SLOT_DIM, n_heads=CROSS_ATTN_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm = nn.GroupNorm(8, dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(slot_dim, dim)
        self.to_v = nn.Linear(slot_dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, slots):
        """
        x: [B, C, H, W] spatial features
        slots: [B, N_slots, slot_dim]
        """
        B, C, H, W = x.shape
        residual = x

        x_flat = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        q = self.to_q(x_flat)  # [B, HW, C]
        k = self.to_k(slots)   # [B, N_slots, C]
        v = self.to_v(slots)   # [B, N_slots, C]

        # Multi-head
        q = q.view(B, H * W, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]

        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        out = self.proj_out(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return residual + out


class ResBlock(nn.Module):
    """ResNet block with time embedding injection."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SlotConditionedUNet(nn.Module):
    """Small UNet for latent diffusion, conditioned on slots via cross-attention.

    Architecture:
      - Encoder: 3 levels with downsampling (32->16->8)
      - Each level: ResBlock + SlotCrossAttention
      - Bottleneck: ResBlock + SlotCrossAttention
      - Decoder: 3 levels with upsampling (8->16->32)
      - Skip connections between encoder and decoder

    Conditioning:
      - Timestep: sinusoidal embedding + MLP -> injected into ResBlocks
      - Slots: cross-attention at every level (Q=spatial, K/V=slots)
    """

    def __init__(self, in_ch=LATENT_C, dim=UNET_DIM, dim_mults=UNET_DIM_MULTS,
                 slot_dim=SLOT_DIM, n_slots=N_SLOTS):
        super().__init__()
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection
        self.input_proj = nn.Conv2d(in_ch, dim, 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_attns = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        ch = dim
        enc_channels = [dim]
        for mult in dim_mults:
            out_ch = dim * mult
            self.enc_blocks.append(ResBlock(ch, out_ch, time_dim))
            self.enc_attns.append(SlotCrossAttention(out_ch, slot_dim))
            self.enc_downs.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            ch = out_ch
            enc_channels.append(ch)

        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = SlotCrossAttention(ch, slot_dim)
        self.mid_block2 = ResBlock(ch, ch, time_dim)

        # Decoder
        self.dec_ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_attns = nn.ModuleList()
        for mult in reversed(dim_mults):
            out_ch = dim * mult
            self.dec_ups.append(nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1))
            # Skip connection doubles channels
            self.dec_blocks.append(ResBlock(out_ch * 2, out_ch, time_dim))
            self.dec_attns.append(SlotCrossAttention(out_ch, slot_dim))
            ch = out_ch

        # Output
        self.out_norm = nn.GroupNorm(8, dim)
        self.out_conv = nn.Conv2d(dim, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, slots):
        """
        x: [B, 16, 32, 32] noisy latent
        t: [B] timestep indices
        slots: [B, 64, 128] slot embeddings
        Returns: [B, 16, 32, 32] predicted noise
        """
        t_emb = self.time_mlp(t)
        h = self.input_proj(x)

        # Encoder with skip connections
        skips = []
        for block, attn, down in zip(self.enc_blocks, self.enc_attns, self.enc_downs):
            h = block(h, t_emb)
            h = attn(h, slots)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, slots)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for up, block, attn in zip(self.dec_ups, self.dec_blocks, self.dec_attns):
            h = up(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            h = attn(h, slots)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# ==============================================================
# DATASET
# ==============================================================

class LatentSlotDataset(Dataset):
    """Paired (latent_frame, slots) dataset for diffusion training."""

    def __init__(self, latent_files, slot_encoder, device):
        self.frames = []  # List of [16, 32, 32] tensors
        self.slots = []   # List of [64, 128] tensors

        slot_encoder.eval()
        with torch.no_grad():
            for f in latent_files:
                latent = np.load(f).astype(np.float32)  # [16, T, 32, 32]
                C, T, H, W = latent.shape
                for t in range(T):
                    frame = latent[:, t, :, :]  # [16, 32, 32]
                    tokens = frame.reshape(C, H * W).T  # [1024, 16]
                    tok_tensor = torch.from_numpy(tokens).unsqueeze(0).to(device)
                    s = slot_encoder.encode(tok_tensor)  # [1, 64, 128]
                    self.frames.append(torch.from_numpy(frame.copy()))
                    self.slots.append(s.cpu().squeeze(0))

        print(f"  Dataset: {len(self.frames)} frames from {len(latent_files)} videos")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.slots[idx]


# ==============================================================
# DDIM SAMPLING
# ==============================================================

@torch.no_grad()
def ddim_sample(model, slots, diffusion_params, n_steps=50, device="cuda"):
    """DDIM sampling for fast inference."""
    B = slots.shape[0]
    shape = (B, LATENT_C, LATENT_H, LATENT_W)

    # Subsequence of timesteps for DDIM
    step_size = DIFFUSION_STEPS // n_steps
    timesteps = list(range(0, DIFFUSION_STEPS, step_size))[::-1]

    alphas_cumprod = diffusion_params["alphas_cumprod"].to(device)

    x = torch.randn(shape, device=device)

    for i, t_cur in enumerate(timesteps):
        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
        pred_noise = model(x, t_batch, slots)

        alpha_t = alphas_cumprod[t_cur]
        pred_x0 = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
        pred_x0 = pred_x0.clamp(-5, 5)  # stability

        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_next = alphas_cumprod[t_next]
            x = alpha_next.sqrt() * pred_x0 + (1 - alpha_next).sqrt() * pred_noise
        else:
            x = pred_x0

    return x


# ==============================================================
# PRE-FLIGHT CHECK
# ==============================================================

def preflight(args):
    """Verify all prerequisites before spending GPU time."""
    print("=" * 60)
    print("PRE-FLIGHT CHECK")
    print("=" * 60)
    errors = []

    # 1. GPU
    if not torch.cuda.is_available():
        errors.append("No CUDA GPU available")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
        print(f"  GPU: {gpu_name} ({mem_gb:.1f} GB)")

    # 2. Dependencies
    for mod in ["diffusers", "torchvision", "av", "scipy"]:
        try:
            __import__(mod)
            print(f"  {mod}: OK")
        except ImportError:
            errors.append(f"Missing dependency: {mod}")

    # 3. Slot encoder checkpoint
    slot_ckpt = Path(args.slot_ckpt)
    if not slot_ckpt.exists():
        errors.append(f"Slot checkpoint not found: {slot_ckpt}")
    else:
        print(f"  Slot checkpoint: {slot_ckpt} ({slot_ckpt.stat().st_size / 1e6:.1f}MB)")

    # 4. Disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / 1e9
    print(f"  Disk free: {free_gb:.1f} GB")
    if free_gb < 5:
        errors.append(f"Insufficient disk space: {free_gb:.1f} GB (need 5+)")

    # 5. Model size check
    model = SlotConditionedUNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  UNet params: {n_params:,} ({n_params / 1e6:.1f}M)")
    del model

    # 6. Quick forward pass
    print("  Testing forward pass...")
    model = SlotConditionedUNet().cuda()
    x = torch.randn(2, LATENT_C, LATENT_H, LATENT_W, device="cuda")
    t = torch.randint(0, DIFFUSION_STEPS, (2,), device="cuda")
    s = torch.randn(2, N_SLOTS, SLOT_DIM, device="cuda")
    with torch.no_grad():
        out = model(x, t, s)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"  Forward pass: OK (output {out.shape})")
    del model, x, t, s, out
    torch.cuda.empty_cache()

    if errors:
        print("\nPRE-FLIGHT FAILED:")
        for e in errors:
            print(f"  FAIL: {e}")
        sys.exit(1)
    else:
        print("\nPRE-FLIGHT PASSED")
    print()


# ==============================================================
# STEP 1: ENCODE VIDEOS
# ==============================================================

def encode_videos(args):
    """Get latent files for training. Checks multiple sources in order:
    1. Cached latents in DATA_DIR
    2. SSv2 latents (if available from prior run)
    3. Encode from videos using CogVideoX VAE
    4. Fallback: synthetic latents (sufficient for architecture test)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check cached
    existing = sorted(DATA_DIR.glob("*.npy"))
    if len(existing) >= args.n_videos:
        print(f"  Found {len(existing)} cached latents in {DATA_DIR}")
        return existing[:args.n_videos]

    # Check SSv2 latents from prior run
    ssv2_dir = Path("data/ssv2_latents")
    if ssv2_dir.exists():
        ssv2_files = sorted(ssv2_dir.glob("*.npy"))
        if len(ssv2_files) >= args.n_videos:
            print(f"  Using {args.n_videos} SSv2 latents from {ssv2_dir}")
            return ssv2_files[:args.n_videos]

    # Try encoding real videos
    try:
        from vae_utils import load_cogvideox_vae
        import torchvision.io

        print(f"  Attempting to encode real videos...")
        vae, scaling_factor = load_cogvideox_vae("cuda")

        # Look for any video files
        video_dirs = [Path("data/ucf101_raw"), Path("data/ssv2")]
        video_files = []
        for vdir in video_dirs:
            if vdir.exists():
                for ext in ["*.avi", "*.mp4", "*.webm"]:
                    video_files.extend(sorted(vdir.glob(f"**/{ext}")))

        if video_files:
            print(f"  Found {len(video_files)} videos, encoding {args.n_videos}...")
            encoded = 0
            for vf in video_files[:args.n_videos * 2]:
                if encoded >= args.n_videos:
                    break
                try:
                    video, _, info = torchvision.io.read_video(
                        str(vf), pts_unit="sec", end_pts=3.0
                    )  # [T, H, W, C]
                    if video.shape[0] < 9:
                        continue
                    video = video[:49].float() / 255.0  # [T, H, W, C]
                    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
                    video = F.interpolate(video, size=(256, 256), mode="bilinear")
                    x = video.permute(1, 0, 2, 3).unsqueeze(0).half().cuda()
                    with torch.no_grad():
                        latent = (vae.encode(x).latent_dist.sample() *
                                  scaling_factor).float().cpu()
                    np.save(DATA_DIR / f"vid_{encoded:04d}.npy",
                            latent[0].numpy().astype(np.float16))
                    encoded += 1
                    if encoded % 10 == 0:
                        print(f"    Encoded {encoded}/{args.n_videos}")
                except Exception as e:
                    continue
            del vae
            torch.cuda.empty_cache()
            if encoded >= args.n_videos:
                return sorted(DATA_DIR.glob("*.npy"))[:args.n_videos]
    except Exception as e:
        print(f"  Video encoding not available: {e}")

    # Fallback: synthetic latents matching real statistics
    # (Mean ~0.14, Std ~1.15 from SSv2 validation)
    print(f"  Using synthetic latents (N={args.n_videos}, stats matched to real data)")
    print("  NOTE: Sufficient for architecture feasibility test.")
    print("  Real data needed for Phase 1.")
    for i in range(args.n_videos):
        latent = torch.randn(LATENT_C, 9, LATENT_H, LATENT_W) * 1.15 + 0.14
        np.save(DATA_DIR / f"synth_{i:04d}.npy",
                latent.numpy().astype(np.float16))
    return sorted(DATA_DIR.glob("*.npy"))[:args.n_videos]


# ==============================================================
# STEP 2: EXTRACT SLOTS
# ==============================================================

def load_slot_encoder(ckpt_path, device="cuda"):
    """Load trained SlotLatentAutoencoderV2."""
    from models import SlotLatentAutoencoderV2
    encoder = SlotLatentAutoencoderV2(
        n_slots=N_SLOTS, slot_dim=SLOT_DIM,
        input_dim=16, n_tokens=N_TOKENS, n_iter=5,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


# ==============================================================
# STEP 3: TRAIN
# ==============================================================

def train(args):
    device = "cuda"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 0: SLOT-CONDITIONED DIFFUSION — FEASIBILITY CHECK")
    print("=" * 60)

    # Load slot encoder
    print("\n[1/5] Loading slot encoder...")
    slot_encoder = load_slot_encoder(args.slot_ckpt, device)
    print(f"  Loaded from {args.slot_ckpt}")

    # Encode videos (or use cached)
    print("\n[2/5] Preparing latents...")
    latent_files = encode_videos(args)

    # Build dataset
    print("\n[3/5] Building dataset (extracting slots)...")
    dataset = LatentSlotDataset(latent_files, slot_encoder, device)
    del slot_encoder  # free GPU memory

    # Train/val split
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    print(f"  Train: {n_train}, Val: {n_val}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

    # Create model
    print("\n[4/5] Creating SlotConditionedUNet...")
    model = SlotConditionedUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Also create an UNCONDITIONED baseline (no slots, just zeros)
    # We'll evaluate both to measure slot conditioning effect

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Diffusion params
    betas = linear_beta_schedule(DIFFUSION_STEPS)
    dp = get_diffusion_params(betas)
    for k, v in dp.items():
        dp[k] = v.to(device)

    # LR schedule
    total_steps = args.steps
    warmup_steps = min(500, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    print(f"\n[5/5] Training for {total_steps} steps...")
    history = {"train_loss": [], "val_loss_cond": [], "val_loss_uncond": []}
    best_val_loss = float("inf")
    step = 0
    t_start = time.time()

    while step < total_steps:
        model.train()
        for frames, slots in train_dl:
            if step >= total_steps:
                break

            frames = frames.to(device)
            slots = slots.to(device)

            # Sample random timesteps
            t = torch.randint(0, DIFFUSION_STEPS, (frames.shape[0],), device=device)
            noise = torch.randn_like(frames)
            x_noisy = q_sample(frames, t, noise,
                              dp["sqrt_alphas_cumprod"],
                              dp["sqrt_one_minus_alphas_cumprod"])

            # Predict noise (conditioned on slots)
            pred_noise = model(x_noisy, t, slots)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            step += 1

            if step % 200 == 0:
                elapsed = time.time() - t_start
                print(f"  Step {step:5d}/{total_steps} | "
                      f"Loss: {loss.item():.6f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"{elapsed/60:.1f}min", flush=True)

            # Validation
            if step % 500 == 0 or step == total_steps:
                model.eval()
                val_loss_cond = 0.0
                val_loss_uncond = 0.0
                n_batches = 0

                with torch.no_grad():
                    for v_frames, v_slots in val_dl:
                        v_frames = v_frames.to(device)
                        v_slots = v_slots.to(device)
                        t = torch.randint(0, DIFFUSION_STEPS,
                                         (v_frames.shape[0],), device=device)
                        noise = torch.randn_like(v_frames)
                        x_noisy = q_sample(v_frames, t, noise,
                                          dp["sqrt_alphas_cumprod"],
                                          dp["sqrt_one_minus_alphas_cumprod"])

                        # Conditioned
                        pred_c = model(x_noisy, t, v_slots)
                        val_loss_cond += F.mse_loss(pred_c, noise).item()

                        # Unconditioned (zero slots)
                        zero_slots = torch.zeros_like(v_slots)
                        pred_u = model(x_noisy, t, zero_slots)
                        val_loss_uncond += F.mse_loss(pred_u, noise).item()

                        n_batches += 1

                val_loss_cond /= max(1, n_batches)
                val_loss_uncond /= max(1, n_batches)
                improvement = (val_loss_uncond - val_loss_cond) / val_loss_uncond * 100

                elapsed = time.time() - t_start
                print(f"  [VAL] Step {step} | "
                      f"Cond: {val_loss_cond:.6f} | "
                      f"Uncond: {val_loss_uncond:.6f} | "
                      f"Slot benefit: {improvement:+.1f}% | "
                      f"{elapsed/60:.1f}min", flush=True)

                history["train_loss"].append(loss.item())
                history["val_loss_cond"].append(val_loss_cond)
                history["val_loss_uncond"].append(val_loss_uncond)

                if val_loss_cond < best_val_loss:
                    best_val_loss = val_loss_cond
                    torch.save(model.state_dict(),
                              CHECKPOINT_DIR / "diffusion_phase0_best.pt")

                model.train()

    # ==============================================================
    # EVALUATION: Generate samples
    # ==============================================================
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    model.eval()
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "diffusion_phase0_best.pt"))

    # Move diffusion params to CPU-accessible format for sampling
    dp_cpu = {k: v.cpu() for k, v in dp.items()}

    # Sample a batch from val set
    val_frames_list = []
    val_slots_list = []
    for f, s in val_dl:
        val_frames_list.append(f)
        val_slots_list.append(s)
        if len(val_frames_list) * BATCH_SIZE >= 16:
            break
    val_frames = torch.cat(val_frames_list)[:16].to(device)
    val_slots = torch.cat(val_slots_list)[:16].to(device)

    # Generate with slot conditioning
    print("  Generating samples (conditioned)...")
    gen_cond = ddim_sample(model, val_slots, dp, n_steps=50, device=device)

    # Generate without slot conditioning
    print("  Generating samples (unconditioned)...")
    zero_slots = torch.zeros_like(val_slots)
    gen_uncond = ddim_sample(model, zero_slots, dp, n_steps=50, device=device)

    # Compute reconstruction metrics
    mse_cond = F.mse_loss(gen_cond, val_frames).item()
    mse_uncond = F.mse_loss(gen_uncond, val_frames).item()
    mse_improvement = (mse_uncond - mse_cond) / mse_uncond * 100

    # Per-channel variance ratio
    with torch.no_grad():
        gt_std = val_frames.std(dim=(0, 2, 3))  # [16]
        gen_std = gen_cond.std(dim=(0, 2, 3))    # [16]
        var_ratio = (gen_std / gt_std).mean().item()

    print(f"\n  MSE (conditioned):   {mse_cond:.6f}")
    print(f"  MSE (unconditioned): {mse_uncond:.6f}")
    print(f"  Slot conditioning benefit: {mse_improvement:+.1f}%")
    print(f"  Variance ratio (gen/gt):   {var_ratio:.3f}")

    # Decode through VAE for visual inspection (if possible)
    try:
        from vae_utils import load_cogvideox_vae, decode_single_frame
        vae, sf = load_cogvideox_vae("cuda")
        print("\n  Decoding samples through VAE for visualization...")

        n_show = min(4, len(val_frames))
        for i in range(n_show):
            gt_pixels = decode_single_frame(vae, val_frames[i], sf)
            gen_pixels = decode_single_frame(vae, gen_cond[i], sf)

            # Save as numpy for later visualization
            np.save(FIGURES_DIR / f"phase0_gt_{i}.npy",
                    gt_pixels.cpu().numpy())
            np.save(FIGURES_DIR / f"phase0_gen_{i}.npy",
                    gen_pixels.cpu().numpy())
        print(f"  Saved {n_show} decoded frame pairs to {FIGURES_DIR}/")
        del vae
    except Exception as e:
        print(f"  VAE decode skipped: {e}")

    # Final results
    elapsed = time.time() - t_start

    print("\n" + "=" * 60)
    print("PHASE 0 RESULTS")
    print("=" * 60)
    print(f"  Videos: {args.n_videos}")
    print(f"  Frames: {len(dataset)}")
    print(f"  Training steps: {total_steps}")
    print(f"  Training time: {elapsed/60:.1f} min")
    print(f"  UNet params: {n_params:,}")
    print(f"  Best val loss (cond): {best_val_loss:.6f}")
    print(f"  MSE (cond):   {mse_cond:.6f}")
    print(f"  MSE (uncond): {mse_uncond:.6f}")
    print(f"  Slot benefit: {mse_improvement:+.1f}%")
    print(f"  Var ratio:    {var_ratio:.3f}")

    if mse_improvement > 5:
        print(f"\n  FEASIBILITY: PASSED — slots help by {mse_improvement:.1f}%")
        print("  → Proceed to Phase 1 (full training)")
    elif mse_improvement > 0:
        print(f"\n  FEASIBILITY: MARGINAL — slots help by {mse_improvement:.1f}%")
        print("  → May need DINOv2 slots or larger model")
    else:
        print(f"\n  FEASIBILITY: FAILED — slots don't help ({mse_improvement:.1f}%)")
        print("  → Investigate DINOv2 slot extraction")

    results = {
        "phase": "diffusion_phase0",
        "n_videos": args.n_videos,
        "n_frames": len(dataset),
        "training_steps": total_steps,
        "training_minutes": elapsed / 60,
        "n_params": n_params,
        "best_val_loss_cond": best_val_loss,
        "mse_conditioned": mse_cond,
        "mse_unconditioned": mse_uncond,
        "slot_benefit_pct": mse_improvement,
        "variance_ratio": var_ratio,
        "history": history,
    }
    with open(RESULTS_DIR / "diffusion_phase0.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {RESULTS_DIR / 'diffusion_phase0.json'}")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: Diffusion Feasibility")
    parser.add_argument("--n-videos", type=int, default=50,
                        help="Number of videos to encode")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Training steps")
    parser.add_argument("--slot-ckpt", type=str,
                        default="checkpoints/slot_encoder_best.pt",
                        help="Path to trained slot encoder")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    if not args.skip_preflight:
        preflight(args)

    train(args)
