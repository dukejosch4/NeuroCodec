"""SSv2 Dataset Pipeline: Preprocess and VAE Encode local WebM files.

Usage:
    python src/ssv2_pipeline.py --phase preflight   # Test 100 videos
    python src/ssv2_pipeline.py --phase encode       # Full 220K encoding

Expects:
    - SSv2 WebM files in data/ssv2/20bn-something-something-v2/*.webm
    - CogVideoX-2b VAE (auto-downloaded from HuggingFace)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Config ───────────────────────────────────────────────────────────
TARGET_FRAMES = 49       # CogVideoX expects 49 frames
TARGET_SIZE = (256, 256) # Spatial resolution
BATCH_SIZE_ENCODE = 8    # Videos per VAE encode batch (A100-40GB, 13GB peak)
NUM_WORKERS = 8          # Parallel video loading workers

DATA_DIR = Path("data/ssv2")
VIDEO_DIR = DATA_DIR / "20bn-something-something-v2"
LATENT_DIR = Path("data/ssv2_latents")
RESULTS_DIR = Path("results/json")


# ── Dataset ──────────────────────────────────────────────────────────
class SSv2FileDataset(Dataset):
    """Dataset that loads WebM files from a list of paths."""

    def __init__(self, file_paths):
        self.paths = file_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        video_id = path.stem
        try:
            frames = load_video(path)
            return frames, video_id, True  # success flag
        except Exception:
            # Return dummy on failure — filter in collate
            return torch.zeros(TARGET_FRAMES, 3, *TARGET_SIZE), video_id, False


def collate_filter(batch):
    """Collate that filters out failed loads."""
    frames_list, ids_list, ok_list = zip(*batch)
    valid = [(f, i) for f, i, ok in zip(frames_list, ids_list, ok_list) if ok]
    if not valid:
        return None, None
    frames, ids = zip(*valid)
    return torch.stack(frames), list(ids)


# ── Video Loading ────────────────────────────────────────────────────
def load_video(path):
    """Load a single WebM video using PyAV/torchvision, return [T, C, H, W] in [0, 1]."""
    import torchvision.io
    video, _, _ = torchvision.io.read_video(str(path), pts_unit="sec")
    # video: [T, H, W, C] uint8
    frames = video.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
    frames = F.interpolate(frames, size=TARGET_SIZE, mode="bilinear", align_corners=False)
    return pad_to_target(frames)


def pad_to_target(frames):
    """Pad or subsample to exactly TARGET_FRAMES frames."""
    T = frames.shape[0]
    if T == TARGET_FRAMES:
        return frames
    elif T > TARGET_FRAMES:
        indices = torch.linspace(0, T - 1, TARGET_FRAMES).long()
        return frames[indices]
    else:
        pad_count = TARGET_FRAMES - T
        last_frame = frames[-1:].expand(pad_count, -1, -1, -1)
        return torch.cat([frames, last_frame], dim=0)


# ── VAE Encoding ─────────────────────────────────────────────────────
def encode_videos(vae, videos, scaling_factor):
    """Encode a batch of videos through CogVideoX VAE.

    Args:
        vae: CogVideoX VAE
        videos: [B, T, C, H, W] tensor in [0, 1]
        scaling_factor: VAE scaling factor

    Returns:
        latents: [B, 16, T_latent, 32, 32]
    """
    device = next(vae.parameters()).device
    with torch.no_grad():
        # CogVideoX VAE expects [B, C, T, H, W]
        x = videos.permute(0, 2, 1, 3, 4).half().to(device)
        latent_dist = vae.encode(x).latent_dist
        latents = latent_dist.sample() * scaling_factor
    return latents.float().cpu()


def decode_latents(vae, latents, scaling_factor):
    """Decode latents back to pixels for quality check."""
    device = next(vae.parameters()).device
    with torch.no_grad():
        decoded = vae.decode(latents.half().to(device) / scaling_factor).sample
    return decoded.float().cpu().clamp(0, 1)


# ── Metrics ──────────────────────────────────────────────────────────
def compute_vae_quality(original, reconstructed):
    """Compute SSIM and PSNR between original and reconstructed videos."""
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio

    T_orig = original.shape[1]
    T_recon = reconstructed.shape[2]
    mid_orig = T_orig // 2
    mid_recon = T_recon // 2

    ssims, psnrs = [], []
    B = original.shape[0]

    for b in range(B):
        orig_frame = original[b, mid_orig].permute(1, 2, 0).numpy()
        recon_frame = reconstructed[b, :, mid_recon].permute(1, 2, 0).numpy()

        if orig_frame.shape != recon_frame.shape:
            from skimage.transform import resize
            recon_frame = resize(recon_frame, orig_frame.shape[:2], anti_aliasing=True)

        ssim = structural_similarity(orig_frame, recon_frame, channel_axis=2, data_range=1.0)
        psnr = peak_signal_noise_ratio(orig_frame, recon_frame, data_range=1.0)
        ssims.append(ssim)
        psnrs.append(psnr)

    return np.mean(ssims), np.mean(psnrs), ssims, psnrs


# ── Phases ───────────────────────────────────────────────────────────
def get_video_files():
    """Get sorted list of all WebM video files."""
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Video directory not found: {VIDEO_DIR}\n"
            f"Run: python src/download_ssv2.py"
        )
    files = sorted(VIDEO_DIR.glob("*.webm"))
    if not files:
        raise FileNotFoundError(f"No .webm files in {VIDEO_DIR}")
    return files


def run_preflight(args):
    """Phase 0+1.1: Test 100 videos through full VAE pipeline."""
    print("=" * 60)
    print("PHASE 0+1.1: PREFLIGHT (100 videos)")
    print("=" * 60)

    print("\n[1/4] Finding SSv2 video files...")
    all_files = get_video_files()
    print(f"  Found {len(all_files)} total videos")
    test_files = all_files[:100]

    print("[2/4] Loading 100 videos...")
    videos = []
    video_ids = []
    for i, path in enumerate(test_files):
        try:
            frames = load_video(path)
            videos.append(frames)
            video_ids.append(path.stem)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
            continue
        if (i + 1) % 20 == 0:
            print(f"  Loaded {i + 1}/100 videos")

    print(f"  Successfully loaded {len(videos)} videos")
    print(f"  Shape: {videos[0].shape}")

    print("\n[3/4] Loading CogVideoX VAE...")
    from vae_utils import load_cogvideox_vae
    vae, scaling_factor = load_cogvideox_vae("cuda")

    print("\n[4/4] Encoding + decoding + measuring quality...")
    all_ssims, all_psnrs = [], []
    latent_shapes = []
    encode_times = []

    for i in range(0, len(videos), BATCH_SIZE_ENCODE):
        batch = torch.stack(videos[i:i + BATCH_SIZE_ENCODE])
        t0 = time.time()
        latents = encode_videos(vae, batch, scaling_factor)
        encode_time = time.time() - t0
        encode_times.append(encode_time)

        latent_shapes.append(latents.shape)
        reconstructed = decode_latents(vae, latents, scaling_factor)

        ssim, psnr, batch_ssims, batch_psnrs = compute_vae_quality(batch, reconstructed)
        all_ssims.extend(batch_ssims)
        all_psnrs.extend(batch_psnrs)

        print(f"  Batch {i // BATCH_SIZE_ENCODE + 1}: SSIM={ssim:.4f}, PSNR={psnr:.2f}dB, "
              f"latent={latents.shape}, time={encode_time:.1f}s")

    mean_ssim = np.mean(all_ssims)
    mean_psnr = np.mean(all_psnrs)
    std_ssim = np.std(all_ssims)

    print("\n" + "=" * 60)
    print("PREFLIGHT RESULTS")
    print("=" * 60)
    print(f"  Videos tested: {len(videos)}")
    print(f"  VAE SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}")
    print(f"  VAE PSNR: {mean_psnr:.2f} dB")
    print(f"  Latent shape: {latent_shapes[0]}")
    print(f"  Encode time: {np.mean(encode_times):.1f}s per batch of {BATCH_SIZE_ENCODE}")
    print(f"  Estimated full encoding: {np.mean(encode_times) / BATCH_SIZE_ENCODE * 220000 / 3600:.1f}h")

    if mean_ssim < 0.90:
        print(f"\n  WARNING KILL CRITERION: SSIM {mean_ssim:.4f} < 0.90")
    else:
        print(f"\n  PREFLIGHT PASSED: SSIM {mean_ssim:.4f} > 0.90")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "phase": "preflight",
        "n_videos": len(videos),
        "total_videos_found": len(all_files),
        "vae_ssim_mean": float(mean_ssim),
        "vae_ssim_std": float(std_ssim),
        "vae_psnr_mean": float(mean_psnr),
        "latent_shape": list(latent_shapes[0]),
        "encode_time_per_video": float(np.mean(encode_times) / BATCH_SIZE_ENCODE),
        "estimated_full_hours": float(np.mean(encode_times) / BATCH_SIZE_ENCODE * 220000 / 3600),
        "kill_criterion_passed": bool(mean_ssim >= 0.90),
        "ssims": [float(s) for s in all_ssims],
        "psnrs": [float(p) for p in all_psnrs],
    }
    with open(RESULTS_DIR / "ssv2_preflight.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'ssv2_preflight.json'}")


def run_encode(args):
    """Phase 1.2: Full VAE encoding of all SSv2 videos with parallel loading."""
    print("=" * 60)
    print("PHASE 1.2: FULL VAE ENCODING")
    print("=" * 60)

    LATENT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all video files
    all_files = get_video_files()
    total = len(all_files)
    print(f"  Found {total} video files")

    # Check already encoded
    existing = set(p.stem for p in LATENT_DIR.glob("*.npy"))
    remaining = [f for f in all_files if f.stem not in existing]
    print(f"  Already encoded: {len(existing)}")
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("  All videos already encoded!")
        return

    # Load VAE
    print("\n[1/2] Loading CogVideoX VAE...")
    from vae_utils import load_cogvideox_vae
    vae, scaling_factor = load_cogvideox_vae("cuda")

    # Create DataLoader with parallel workers
    dataset = SSv2FileDataset(remaining)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_ENCODE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_filter,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Encode
    print(f"[2/2] Encoding {len(remaining)} videos (BS={BATCH_SIZE_ENCODE}, workers={NUM_WORKERS})...")
    encoded_count = 0
    skipped_count = 0
    t_start = time.time()

    for batch_idx, (batch_frames, batch_ids) in enumerate(loader):
        if batch_frames is None:
            skipped_count += BATCH_SIZE_ENCODE
            continue

        try:
            latents = encode_videos(vae, batch_frames, scaling_factor)

            for j, vid_id in enumerate(batch_ids):
                import numpy as np
                np.save(LATENT_DIR / f"{vid_id}.npy", latents[j].cpu().numpy().astype(np.float16))
                encoded_count += 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM at batch {batch_idx}, clearing cache...")
                torch.cuda.empty_cache()
                skipped_count += len(batch_ids)
            else:
                raise

        torch.cuda.empty_cache()

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = encoded_count / elapsed if elapsed > 0 else 0
            done = len(existing) + encoded_count
            eta = (total - done) / rate / 3600 if rate > 0 else 0
            print(f"  Encoded: {done}/{total} "
                  f"({rate:.1f} vid/s, ~{eta:.1f}h remaining, "
                  f"skipped: {skipped_count})", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Done! Encoded {encoded_count} videos in {elapsed / 3600:.1f}h")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total encoded: {len(existing) + encoded_count}/{total}")
    print(f"  Latents saved to {LATENT_DIR}/")

    # Save encoding stats
    results = {
        "phase": "encode",
        "total_videos": total,
        "total_encoded": len(existing) + encoded_count,
        "newly_encoded": encoded_count,
        "skipped": skipped_count,
        "time_hours": elapsed / 3600,
        "rate_videos_per_sec": encoded_count / elapsed if elapsed > 0 else 0,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ssv2_encoding_stats.json", "w") as f:
        json.dump(results, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSv2 Pipeline")
    parser.add_argument("--phase", choices=["preflight", "encode"], required=True)
    args = parser.parse_args()

    if args.phase == "preflight":
        run_preflight(args)
    elif args.phase == "encode":
        run_encode(args)
