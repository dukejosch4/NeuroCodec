<p align="center">
  <img src="assets/slot_compression.jpg" width="700">
</p>

<h1 align="center">NeuroCodec</h1>

<p align="center">
  <strong>Efficient Video Prediction via Residual Latent Dynamics</strong>
</p>

<p align="center">
  <a href="https://zenodo.org/records/18860333"><img src="https://img.shields.io/badge/Paper-Zenodo-blue" alt="Paper"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D2.0-red.svg" alt="PyTorch"></a>
</p>

---

NeuroCodec predicts future video frames as **residual updates** in a pre-trained video VAE latent space. Instead of generating each frame from scratch, we predict lightweight deltas: `L_{t+1} = L_t + Delta` — an approach motivated by predictive processing in biological vision.

## Architecture

<p align="center">
  <img src="assets/residual_update.jpg" width="650">
</p>

```
CogVideoX VAE Encoder → Latents [B, 16, T, 32, 32]
  → Slot Attention         → 64 slots × 128d  (16× compression)
  → Dynamics Transformer   → predict S_{t+1}   (438K params)
  → Boundary Detector      → EASY (85%) / HARD (15%)
  → Residual Decoder       → L_{t+1} = L_t + Delta  (1.43M params)
  → Manifold Projector     → correct off-manifold drift  (55K params)
  → CogVideoX VAE Decoder  → pixel frames
```

**Total trainable parameters: ~2.5M** (vs. 215M frozen VAE)

## Results

### UCF-101

| Metric | Ours | Copy Baseline | Improvement |
|--------|------|---------------|-------------|
| Latent MSE | **0.216** | 0.342 | **-36.8%** |
| LPIPS (single-step) | **0.105** | 0.097 | within 8% |
| SSIM (single-step) | **0.849** | 0.847 | +0.2% |
| Inference latency | **0.25 ms** | — | **130-406× faster** than UNet-50 |
| Rollout stability (8 frames) | **2.07×** | 1.83× | stable multi-step |

### Something-Something v2 (Cross-Dataset Generalization)

The architecture generalizes to SSv2 (220K videos, 2.65M frame pairs) **without any modification**:

| Metric | Value |
|--------|-------|
| Slot Variance Explained | **88.6%** |
| Dynamics vs. Copy Baseline | **+32.2%** |

<p align="center">
  <img src="assets/off_manifold.jpg" width="600">
  <br>
  <em>The off-manifold bottleneck: predicted latents drift off the VAE decoder's data manifold,<br>limiting pixel-level quality regardless of latent-space loss design.</em>
</p>

## Visual Examples

<p align="center">
  <img src="results/demo/video_00_comparison.gif" width="700">
  <br>
  <em>Single-step prediction: Ground Truth (top) vs. NeuroCodec prediction (bottom)</em>
</p>

<p align="center">
  <img src="assets/rollout_strip_00.png" width="700">
  <br>
  <em>8-frame autoregressive rollout — the system maintains visual coherence across multiple steps.</em>
</p>

## Key Findings

1. **Residual updates are fundamental** — removing the residual pathway increases MSE by 225%, making it the single most critical component.

2. **Slot compression is robust** — 16× token reduction (1024→64) captures 88.6% of latent variance while enabling 130-406× inference speedup.

3. **The off-manifold bottleneck** — systematic evaluation of 6 improvement strategies reveals that predicted latents lie off the VAE decoder's data manifold. No latent-space loss (including end-to-end pixel supervision) can fully close this gap without modifying the decoder.

4. **Decoupled manifold projection** — gradient feedback from the projector must be decoupled from the main prediction pathway to prevent training interference. This is a genuinely novel contribution.

## Design Principles

| Principle | Implementation | Effect |
|-----------|---------------|--------|
| Predictive Coding | Residual updates (`L_t + Delta`) | -36.8% MSE vs. full decoding |
| Sparse Coding | Slot Attention (1024→64 tokens) | 16× compression, 88.6% variance retained |
| Event Segmentation | Boundary Detector (EASY/HARD routing) | Allocates capacity where needed |
| Spectral Loss | Frequency-domain matching | -55% variance-ratio deviation |

## Ablation Study

| Configuration | MSE | vs. Full System |
|---------------|-----|-----------------|
| Full System | 0.219 | — |
| w/o Residual | 0.711 | +225% |
| w/o Dynamics | 0.220 | +0.5% |
| w/o Slots | 0.211 | -3.8% (but 3.16× rollout instability) |
| w/o Event Segmentation | 0.216 | -1.4% |
| Copy Baseline | 0.342 | +56% |

## Installation

```bash
git clone https://github.com/dukejosch4/NeuroCodec.git
cd NeuroCodec
pip install -r requirements.txt
```

Requires **PyTorch >= 2.0** with CUDA support. The CogVideoX VAE (~4GB) is downloaded automatically.

## Quick Start

```bash
# Train with spectral loss
python scripts/train.py --data-dir /path/to/latents --spectral-beta 0.01

# Evaluate
python scripts/evaluate.py --data-dir /path/to/latents --checkpoint checkpoints/best.pt

# Demo: single-step prediction with visualization
python demo/predict.py --checkpoint best.pt --data-dir /path/to/latents
```

## Project Structure

```
NeuroCodec/
├── paper/main.tex          # Full paper (LaTeX source)
├── src/
│   ├── models.py           # All model components
│   ├── losses.py           # Spectral + variance losses
│   └── vae_utils.py        # CogVideoX VAE helpers
├── scripts/
│   ├── train.py            # Training loop
│   └── evaluate.py         # Evaluation pipeline
├── results/
│   ├── json/               # Experiment results (JSON)
│   ├── figures/            # Analysis plots
│   └── demo/               # Visual comparisons + GIFs
└── demo/predict.py         # Self-contained demo
```

## Experiment Log

| ID | Experiment | Result |
|----|-----------|--------|
| D2 | Core system (3 seeds) | MSE 0.216 ± 0.0004, 130-406× speedup |
| E1 | Spectral loss ablation | LPIPS -21%, key quality lever |
| E2 | Spectral weight sweep | beta=0.01 optimal |
| E3 | Latent adapter | NO-GO (stats shift too small) |
| E4 | Channel-weighted MSE | NO-GO (marginal gains) |
| E5 | Variational residual | NO-GO (posterior collapse) |
| E6 | End-to-end pixel tuning | NO-GO (off-manifold bottleneck) |
| SSv2 | Cross-dataset scaling | 88.6% VarExp, +32.2% dynamics |
| Diff-P0 | Diffusion feasibility | +11.9% slot benefit (PASSED) |

## Citation

```bibtex
@article{haertel2026neurocodec,
  title   = {NeuroCodec: Efficient Video Prediction via Residual Latent Dynamics},
  author  = {H{\"a}rtel, Joscha},
  year    = {2026},
  doi     = {10.5281/zenodo.18860333},
  url     = {https://zenodo.org/records/18860333}
}
```

## License

MIT
