"""Generate structured synthetic videos, encode through CogVideoX VAE.

Creates simple moving-rectangle videos with temporal coherence,
encodes them through the frozen VAE to get ON-MANIFOLD latents.
These have real spatial/temporal structure that slots can capture.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import shutil

DATA_DIR = Path("data/ucf101_latents_phase0")
N_VIDEOS = 50
N_FRAMES = 9

def main():
    from vae_utils import load_cogvideox_vae

    # Clean old data
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)

    vae, sf = load_cogvideox_vae("cuda")
    print(f"VAE loaded (scaling_factor={sf:.4f})")

    for i in range(N_VIDEOS):
        # Create video: moving colored rectangle on gradient background
        frames = torch.zeros(N_FRAMES, 3, 256, 256)

        # Random rectangle params
        cx = torch.randint(40, 216, (1,)).item()
        cy = torch.randint(40, 216, (1,)).item()
        dx = torch.randint(-4, 5, (1,)).item()
        dy = torch.randint(-4, 5, (1,)).item()
        color = torch.rand(3)
        rect_size = torch.randint(10, 30, (1,)).item()

        # Background gradient (varies per video)
        bg_angle = torch.rand(1).item() * 3.14
        bg_color = torch.rand(3) * 0.4

        for t in range(N_FRAMES):
            x = max(rect_size, min(cx + dx * t, 255 - rect_size))
            y = max(rect_size, min(cy + dy * t, 255 - rect_size))

            # Gradient background
            coords = torch.stack(torch.meshgrid(
                torch.linspace(0, 1, 256),
                torch.linspace(0, 1, 256),
                indexing="ij"
            ))
            bg_val = (coords[0] * np.cos(bg_angle) +
                      coords[1] * np.sin(bg_angle))
            for c in range(3):
                frames[t, c] = bg_val * bg_color[c] + 0.1

            # Rectangle
            y0, y1 = max(0, y - rect_size), min(256, y + rect_size)
            x0, x1 = max(0, x - rect_size), min(256, x + rect_size)
            frames[t, :, y0:y1, x0:x1] = color.view(3, 1, 1)

        # Encode through CogVideoX VAE: [1, C, T, H, W]
        x = frames.permute(1, 0, 2, 3).unsqueeze(0).half().cuda()
        with torch.no_grad():
            latent = (vae.encode(x).latent_dist.sample() * sf).float().cpu()

        np.save(DATA_DIR / f"struct_{i:04d}.npy",
                latent[0].numpy().astype(np.float16))

        if (i + 1) % 10 == 0:
            print(f"  Encoded {i+1}/{N_VIDEOS}")

    # Verify
    sample = np.load(sorted(DATA_DIR.glob("*.npy"))[0]).astype(np.float32)
    print(f"\nDone: {N_VIDEOS} videos -> {DATA_DIR}")
    print(f"Shape: {sample.shape}, Mean: {sample.mean():.3f}, Std: {sample.std():.3f}")

    del vae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
