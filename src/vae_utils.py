"""CogVideoX 3D VAE decode utilities.

The CogVideoX VAE expects a temporal dimension, so single frames must
be repeated 9 times along the temporal axis. Decoding takes ~539ms per
frame on A100.
"""

import torch


def decode_single_frame(
    vae,
    latent_frame: torch.Tensor,
    scaling_factor: float,
    n_repeats: int = 9,
) -> torch.Tensor:
    """Decode a single latent frame through CogVideoX 3D VAE.

    The 3D VAE requires temporal context, so we repeat the frame along
    the time axis and extract the middle frame from the decoded output.

    Args:
        vae: CogVideoX VAE model (AutoencoderKLCogVideoX)
        latent_frame: [16, 32, 32] single latent frame
        scaling_factor: VAE scaling factor (from vae.config.scaling_factor)
        n_repeats: number of temporal repeats (default 9, minimum 3)

    Returns:
        pixels: [3, H, W] decoded pixel frame in [0, 1] range
    """
    device = next(vae.parameters()).device
    with torch.no_grad():
        inp = (
            latent_frame.unsqueeze(0)
            .unsqueeze(2)
            .repeat(1, 1, n_repeats, 1, 1)
            .half()
            .to(device)
        )
        decoded = vae.decode(inp / scaling_factor).sample
        mid = decoded.shape[2] // 2
        frame = decoded[0, :, mid].float().clamp(0, 1)
    return frame


def load_cogvideox_vae(device: str = "cuda"):
    """Load CogVideoX-2B VAE decoder.

    Returns:
        (vae, scaling_factor) tuple
    """
    from diffusers import AutoencoderKLCogVideoX

    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.float16
    ).to(device).eval()
    return vae, vae.config.scaling_factor
