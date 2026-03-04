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

<p align="center">
  <b>36.8% lower MSE</b> &nbsp;|&nbsp; <b>130-406x faster</b> than UNet-50 &nbsp;|&nbsp; <b>~2.5M trainable params</b> &nbsp;|&nbsp; <b>cross-dataset transfer</b>
</p>

---

NeuroCodec predicts future video frames as **residual updates** in a frozen video VAE latent space: `L_{t+1} = L_t + Delta`. Instead of generating each frame from scratch, we predict only what changes — motivated by predictive processing in biological vision.

## Architecture

<p align="center">
  <img src="assets/residual_update.jpg" width="650">
</p>

```
CogVideoX VAE Encoder → Latents [B, 16, T, 32, 32]
  → Slot Attention         → 64 slots × 128d  (16x compression)
  → Dynamics Transformer   → predict S_{t+1}   (438K params)
  → Boundary Detector      → EASY (85%) / HARD (15%)
  → Residual Decoder       → L_{t+1} = L_t + Delta  (1.43M params)
  → Manifold Projector     → correct off-manifold drift  (55K params)
  → CogVideoX VAE Decoder  → pixel frames
```

## Results

### UCF-101

| Metric | Ours | Copy Baseline | Improvement |
|--------|------|---------------|-------------|
| Latent MSE | **0.216** | 0.342 | **-36.8%** |
| LPIPS (single-step) | **0.105** | 0.097 | within 8% |
| SSIM (single-step) | **0.849** | 0.847 | +0.2% |
| Inference latency | **0.25 ms** | — | **130-406x faster** than UNet-50 |
| Rollout stability (8 frames) | **2.07x** | 1.83x | stable multi-step |

### Something-Something v2 (Cross-Dataset Transfer)

The architecture generalizes to SSv2 (220K videos, 2.65M frame pairs) **without any modification**:

| Metric | Value |
|--------|-------|
| Slot Variance Explained | **88.6%** |
| Dynamics vs. Copy Baseline | **+32.2%** |

### Ablation

| Configuration | MSE | vs. Full System |
|---------------|-----|-----------------|
| Full System | 0.219 | — |
| w/o Residual | 0.711 | +225% |
| w/o Slots | 0.211 | -3.8% (but 3.16x rollout instability) |
| w/o Event Segmentation | 0.216 | -1.4% |
| Copy Baseline | 0.342 | +56% |

## Visual Examples

<p align="center">
  <img src="results/demo/video_00_comparison.gif" width="700">
  <br>
  <em>Ground Truth (top) vs. NeuroCodec prediction (bottom). Predictions achieve strong latent-space metrics<br>(36.8% lower MSE), but pixel reconstructions show progressive blurring due to the off-manifold bottleneck (see below).</em>
</p>

<p align="center">
  <img src="assets/rollout_strip_00.png" width="700">
  <br>
  <em>8-frame autoregressive rollout. The system maintains structural coherence, but accumulated<br>off-manifold drift causes softening — <b>a fundamental limitation of frozen VAE decoders, not of the prediction itself.</b></em>
</p>

## Key Findings

<p align="center">
  <img src="assets/off_manifold.jpg" width="600">
</p>

1. **Residual updates are fundamental** — removing the residual pathway increases MSE by 225%.

2. **Slot compression is robust** — 16x token reduction (1024 to 64) captures 88.6% of latent variance while enabling 130-406x inference speedup.

3. **The off-manifold bottleneck** — systematic evaluation of 6 improvement strategies reveals that predicted latents lie off the VAE decoder's data manifold. No latent-space loss can fully close this gap without modifying the decoder.

4. **Decoupled manifold projection** — gradient feedback from the projector must be decoupled from the main prediction pathway to prevent training interference.

## Getting Started

```bash
git clone https://github.com/dukejosch4/NeuroCodec.git
cd NeuroCodec
pip install -r requirements.txt
```

Requires **PyTorch >= 2.0** with CUDA. The CogVideoX VAE (~4GB) downloads automatically.

```bash
# Train
python scripts/train.py --data-dir /path/to/latents --spectral-beta 0.01

# Evaluate
python scripts/evaluate.py --data-dir /path/to/latents --checkpoint checkpoints/best.pt

# Demo
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

<details>
<summary><b>Full Experiment Log</b></summary>

| ID | Experiment | Result |
|----|-----------|--------|
| D2 | Core system (3 seeds) | MSE 0.216 +/- 0.0004, 130-406x speedup |
| E1 | Spectral loss ablation | LPIPS -21%, key quality lever |
| E2 | Spectral weight sweep | beta=0.01 optimal |
| E3 | Latent adapter | NO-GO (stats shift too small) |
| E4 | Channel-weighted MSE | NO-GO (marginal gains) |
| E5 | Variational residual | NO-GO (posterior collapse) |
| E6 | End-to-end pixel tuning | NO-GO (off-manifold bottleneck) |
| SSv2 | Cross-dataset scaling | 88.6% VarExp, +32.2% dynamics |
| Diff-P0 | Diffusion feasibility | +11.9% slot benefit (PASSED) |

</details>

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

## Acknowledgments

Built with [Research Brain](https://github.com/dukejosch4/research-brain) — a persistent cognitive architecture for AI-assisted research with Claude Code.

## License

MIT
