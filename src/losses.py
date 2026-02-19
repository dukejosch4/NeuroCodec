"""NeuroCodec loss functions.

Spectral frequency-matching loss is the key contribution:
it recovers 56% of the variance suppressed by pure MSE training,
reducing LPIPS by 21% (0.223 -> 0.177).
"""

import torch
import torch.nn.functional as F


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize differences in FFT magnitude spectrum.

    This is the key loss that recovers suppressed high-frequency
    latent structure caused by MSE regression to the mean.

    Args:
        pred: [B, 16, 32, 32] predicted latent (spatial format)
        target: [B, 16, 32, 32] ground-truth latent (spatial format)

    Returns:
        Scalar loss
    """
    pred_fft = torch.fft.rfft2(pred)
    target_fft = torch.fft.rfft2(target)
    return F.mse_loss(pred_fft.abs(), target_fft.abs())


def variance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize channel-wise variance mismatch.

    Pushes the per-channel std_ratio (pred_std / target_std) toward 1.0.

    Args:
        pred: [B, 16, 32, 32] predicted latent (spatial format)
        target: [B, 16, 32, 32] ground-truth latent (spatial format)

    Returns:
        Scalar loss
    """
    pred_std = pred.std(dim=(0, 2, 3))
    target_std = target.std(dim=(0, 2, 3))
    ratio = pred_std / (target_std + 1e-8)
    return ((ratio - 1.0) ** 2).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize differences in spatial gradients (sharpness).

    Args:
        pred: [B, 16, 32, 32] predicted latent (spatial format)
        target: [B, 16, 32, 32] ground-truth latent (spatial format)

    Returns:
        Scalar loss
    """
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.mse_loss(pred_dx, target_dx) + F.mse_loss(pred_dy, target_dy)


def combined_loss(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    latent_t: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.01,
    gamma: float = 0.0,
    delta_w: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined training loss.

    Recommended: alpha=1.0, beta=0.01 (spectral only).
    This achieves LPIPS 0.177 with minimal MSE trade-off.

    Args:
        pred_delta: [B, 1024, 16] predicted delta tokens
        target_delta: [B, 1024, 16] ground-truth delta tokens
        latent_t: [B, 1024, 16] current frame latent tokens
        alpha: MSE weight (default 1.0)
        beta: spectral loss weight (default 0.01)
        gamma: variance loss weight (default 0.0)
        delta_w: gradient loss weight (default 0.0)

    Returns:
        (total_loss, loss_dict) where loss_dict contains individual components
    """
    l_mse = F.mse_loss(pred_delta, target_delta)

    # Reconstruct full latents in spatial format
    pred_full = latent_t + pred_delta
    target_full = latent_t + target_delta
    B = pred_full.shape[0]
    pred_spatial = pred_full.permute(0, 2, 1).reshape(B, 16, 32, 32)
    target_spatial = target_full.permute(0, 2, 1).reshape(B, 16, 32, 32)

    l_spectral = spectral_loss(pred_spatial, target_spatial) if beta > 0 else torch.tensor(0.0)
    l_variance = variance_loss(pred_spatial, target_spatial) if gamma > 0 else torch.tensor(0.0)
    l_gradient = gradient_loss(pred_spatial, target_spatial) if delta_w > 0 else torch.tensor(0.0)

    total = alpha * l_mse + beta * l_spectral + gamma * l_variance + delta_w * l_gradient

    return total, {
        "mse": l_mse.item(),
        "spectral": l_spectral.item() if isinstance(l_spectral, torch.Tensor) else 0.0,
        "variance": l_variance.item() if isinstance(l_variance, torch.Tensor) else 0.0,
        "gradient": l_gradient.item() if isinstance(l_gradient, torch.Tensor) else 0.0,
        "total": total.item(),
    }
