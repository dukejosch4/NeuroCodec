# NeuroCodec Experiment Summary

All experiments conducted on A100-40GB with UCF-101 (1,315 videos, CogVideoX latents).

## Phase D2: Core System (Multi-Seed, 3 Seeds)

### Latent-Space MSE

| Split | Copy | Residual V2 | Hybrid |
|-------|------|-------------|--------|
| EASY (85%) | 0.285 | 0.190 +/- .0004 | 0.190 +/- .0007 |
| HARD (15%) | 0.666 | 0.366 +/- .003 | 0.388 +/- .002 |
| ALL | 0.342 | **0.216 +/- .0004** | 0.220 +/- .0008 |

### Ablation

| Config | MSE (ALL) | Delta vs Full |
|--------|-----------|---------------|
| Full System | 0.219 | --- |
| w/o Dynamics (oracle) | 0.220 | +0.5% |
| w/o Slots (self-attn) | 0.211 | -3.8% (but 3.16x rollout!) |
| w/o Event Segmentation | 0.216 | -1.4% |
| w/o Residual (V3 only) | 0.711 | +225% |
| Copy Baseline | 0.342 | +56% |

### Speed (A100)

| Metric | Value |
|--------|-------|
| Prediction step (bs=1) | 2.5 ms |
| UNet 50-step (bs=32) | 1,026 ms |
| **Speedup** | **518x** |

---

## E0: Diagnostics

- Oracle SSIM: 0.808 (ceiling for consecutive frames)
- 3 shifts detected: Mean, Std (ratio 0.55-0.70), Correlation
- Prediction RMSE: 0.48 = 38% of latent std

---

## E1: Perceptual Loss Ablation -- **GO**

| Loss | Ratio Dev | MSE (ALL) | LPIPS |
|------|-----------|-----------|-------|
| MSE only (retrained) | 0.260 | 0.193 | 0.223 |
| + Spectral (beta=0.1) | 0.100 | 0.233 | --- |
| + Gradient (delta=0.1) | 0.266 | 0.194 | --- |
| Full Perceptual | 0.094 | 0.234 | 0.173 |

**Key finding**: Spectral loss is the dominant lever. Gradient loss contributes negligibly.

---

## E2: Spectral Weight Sweep -- **Best: beta=0.01**

| Model | SSIM | LPIPS | Ratio Dev | Rollout Stability |
|-------|------|-------|-----------|-------------------|
| Copy | 0.765 | 0.171 | 0.000 | 1.67x |
| MSE-only (D2) | **0.787** | 0.223 | 0.260 | 1.93x |
| **Spectral beta=0.01** | **0.785** | **0.177** | **0.113** | **1.84x** |
| Rollout-aware | 0.782 | 0.186 | 0.112 | 1.98x |

- LPIPS reduced by 21% (0.223 -> 0.177)
- SSIM only -0.2% trade-off
- Rollout-aware fine-tuning helps nothing

---

## E1b: Latent Adapter -- NO-GO

- Simple (32 params): MSE 0.217 -> 0.216 (+0.3%)
- V2 (27K params): MSE 0.217 -> 0.215 (+0.8%), SSIM worsened
- **Insight**: First-order stats shift is too small (~5%)

---

## E1c: Self-Forcing -- NO-GO

- ALL MSE: 0.217 -> 0.217 (-0.2%)
- Rollout: 1.93x -> 1.93x (unchanged)
- **Insight**: Dynamics was never the bottleneck

---

## E5: Variational Residual (Simple VAE) -- NO-GO

- Posterior collapse: z ignored, best_epoch=0, gain 0.04%
- **Insight**: z_proj zero-init + additive injection = model ignores z

---

## E5b: FiLM-VAE + Scheduled Dropout -- NO-GO

| Metric | Det (z=0) | Best-of-5 | Copy |
|--------|-----------|-----------|------|
| SSIM | 0.783 | 0.783 | 0.765 |
| LPIPS | 0.182 | 0.182 | 0.171 |

- Posterior collapse broken (diversity 5x better), best_epoch=110
- But: det LPIPS 0.182 > baseline 0.177 (dropout degrades quality)
- **Insight**: Stochasticity works but is not the lever for LPIPS

---

## E6 Stage 1: Channel-Weighted MSE -- NO-GO

- best_epoch=0 (warm-start already optimal)
- **Insight**: MSE-trained model at local optimum for weighted variants

---

## E6 Stage 2: End-to-End Pixel Fine-Tuning -- NO-GO

- LPIPS backward through VAE = OOM on A100-40GB (>40 GB)
- L1-only pixel loss: LPIPS **worsened** 0.192 -> 0.213
- **Key insight**: Predicted latents lie off the VAE decoder's data manifold.
  No latent-space loss can fix this. Only decoder modification (LoRA) could help.

---

## Summary: What Worked vs What Didn't

| Experiment | Result | Key Insight |
|-----------|--------|-------------|
| Spectral Loss (E2) | **GO** (-21% LPIPS) | Frequency matching recovers suppressed variance |
| Latent Adapter (E1) | NO-GO | First-order stats not the problem |
| Self-Forcing (E1) | NO-GO | Dynamics not the bottleneck |
| Variational (E5/E5b) | NO-GO | Stochasticity not the lever |
| Channel Weights (E6) | NO-GO | Already at MSE optimum |
| Pixel Fine-Tuning (E6) | NO-GO | Off-manifold problem |
