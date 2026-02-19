# NeuroCodec

**Brain-Inspired Residual Dynamics for Efficient Video Latent Prediction**

NeuroCodec translates four neuroscience principles into a video prediction system that operates in the latent space of a pre-trained video VAE (CogVideoX). Instead of generating each frame from scratch, we predict lightweight residual updates: `L_{t+1} = L_t + Delta`.

```
Frame_t --> [VAE Encoder] --> L_t --> [Slot Attention] --> S_t --> [Dynamics] --> S_{t+1}
                               |                                        |
                               +-----> [Boundary Detector] <-----------+
                               |            |           |
                              EASY (85%)              HARD (15%)
                               |                       |
                        L_t + ResDecV2(...)      FullDecoder(S_{t+1})
                               |                       |
                               +-----> L_{t+1} <------+
```

## Key Results

| Metric | Ours | Copy Baseline | Improvement |
|--------|------|---------------|-------------|
| Latent MSE (ALL) | **0.216** | 0.342 | -36.8% |
| LPIPS (single-step) | **0.177** | 0.171 | within 3.5% |
| SSIM (single-step) | **0.785** | 0.765 | +2.6% |
| Inference time | **1.98 ms** | --- | 518x vs UNet-50 |
| Rollout stability (8 frames) | **1.39x** | 1.67x | more stable |

## Four Neuroscience Principles

1. **Sparse Coding** -- Slot Attention compresses 1024 latent tokens into 64 slots (16x reduction)
2. **Predictive Coding** -- Dynamics Transformer forecasts next-step slots (438K params, 0.65ms)
3. **Event Segmentation** -- Boundary Detector routes frames to residual (EASY) or full decode (HARD)
4. **Hierarchical Residual Update** -- Cross-attention decoder predicts Delta instead of full reconstruction

## Installation

```bash
pip install -r requirements.txt
```

Requires PyTorch >= 2.0 with CUDA support. CogVideoX VAE (~4GB) is downloaded automatically for pixel-level evaluation.

## Project Structure

```
NeuroCodec/
├── paper/main.tex              # Full paper with all results
├── src/
│   ├── models.py               # ResidualDecoderV2, DynamicsTransformer, BoundaryDetector
│   ├── losses.py               # Spectral loss, variance loss, combined loss
│   └── vae_utils.py            # CogVideoX decode helpers
├── scripts/
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation pipeline
├── demo/predict.py             # Self-contained prediction demo
└── results/experiment_summary.md  # All experiment results (E0-E6)
```

## Quick Start

```bash
# Train with spectral loss (recommended: beta=0.01)
python scripts/train.py --data-dir /path/to/data --spectral-beta 0.01

# Evaluate
python scripts/evaluate.py --data-dir /path/to/data --checkpoint checkpoints/residual_v2_best.pt

# Demo: single-step prediction
python demo/predict.py --checkpoint residual_v2_best.pt --data-dir /path/to/data
```

## Experiment Timeline

| Phase | Experiment | Result |
|-------|-----------|--------|
| D2 | Core system (3 seeds) | MSE 0.216, 518x speedup |
| E0 | Diagnostics | Identified mean/std/correlation shifts |
| E1 | Perceptual loss ablation | Spectral loss = key lever |
| E2 | Spectral weight sweep | beta=0.01: LPIPS -21% |
| E1b | Latent adapter | NO-GO (stats shift too small) |
| E1c | Self-forcing | NO-GO (dynamics not bottleneck) |
| E5 | Variational residual | NO-GO (posterior collapse) |
| E5b | FiLM-VAE + dropout | Collapse broken, but quality degrades |
| E6 | End-to-end pixel tuning | NO-GO (off-manifold bottleneck) |

See [results/experiment_summary.md](results/experiment_summary.md) for detailed numbers.

## Key Finding: The Off-Manifold Bottleneck

Our systematic evaluation of 6 improvement strategies reveals that predicted latents lie off the VAE decoder's data manifold. No latent-space loss -- including end-to-end pixel supervision through the frozen decoder -- can fully close the remaining LPIPS gap (0.177 vs copy's 0.171). Only modifying the decoder itself (e.g., LoRA fine-tuning) could address this.

## Citation

```bibtex
@article{haertel2026neurocodec,
  title={NeuroCodec: Brain-Inspired Residual Dynamics for Efficient Video Latent Prediction},
  author={H{\"a}rtel, Joscha},
  year={2026}
}
```

## License

MIT
