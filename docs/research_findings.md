# NeuroCodec: Comprehensive Research & Improvement Strategy

## Status Quo (Phase D2 Complete)

### What Works
- **Residual Prediction**: 36.8% MSE reduction vs. copy baseline (0.216 vs 0.342)
- **Speed**: 2.5ms per frame prediction (~393 FPS), 12.7x faster than 50-step diffusion
- **Rollout Stability**: 1.35x degradation over 8 frames (vs. copy's 1.71x)
- **Multi-Seed Reproducibility**: MSE 0.216 +/- 0.0004 across 3 seeds

### What Doesn't Work
1. **Pixel quality worse than copy**: LPIPS 0.308 (ours) vs. 0.154 (copy), FID 146 vs. 91
2. **2 of 4 principles nearly irrelevant**: w/o Dynamics = 0.220 MSE (only +2%), w/o Event Seg = 0.216 MSE (~0%)
3. **Rollout error accumulation**: LPIPS degrades from 0.10 (frame 1) to 0.44 (frame 8)
4. **VAE decode bottleneck**: 539ms per frame (prediction is 2.5ms, but decode is 539ms)
5. **Deterministic decoder**: Averages over futures, producing blurry predictions

---

## Part I: Root Cause Analysis

### The Latent-Pixel Disconnect (Core Problem)
The CogVideoX 3D VAE was trained exclusively on real encoder outputs. Our predicted latents have distributional shifts that the VAE **nonlinearly amplifies** into perceptual artifacts. Key insight from Sander Dieleman (April 2025): *"Highly nonlinear VAE mappings fundamentally make MSE a poor proxy for perceptual quality."*

This means:
- MSE in latent space is a **poor proxy** for perceptual quality after decoding
- Small latent perturbations can cross nonlinear decision boundaries in the decoder
- The problem is **structural**, not just a matter of training harder with MSE

### Why 2 of 4 Principles Are "Irrelevant"
- **Dynamics Transformer**: Trained on GT slots; at eval uses its own predictions. The +2% degradation reflects exposure bias (train-test gap), not principle failure. **Self-Forcing training would fix this.**
- **Event Segmentation**: The boundary detector is very conservative (recall 26%). It almost never triggers, so removing it changes nothing. **Per-slot gating would provide finer-grained, continuous event detection.**

---

## Part I.b: Diagnostics First (Mentor-Feedback, Feb 19)

**Bevor wir irgendwas umbauen: 3 schnelle Checks zur Root-Cause-Verifikation.**

Script: `run_phase_e0_diagnostics.py` (~20-30 min, ~$0.50)

### Check 1: Decoder Oracle
Decode echte Encoder-Latents -> Pixel. Wenn Qualitat GUT: Decoder ok, Problem ist Distribution Shift.

### Check 2: Statistik-Vergleich
Per-Kanal Mean/Std von predicted vs. echten Latents + Korrelationsmatrix. Sichtbare Abweichung = Smoking Gun.

### Check 3: Sensitivity-Test
Gauss-Noise (sigma=0.01-0.20 relativ) auf echte Latents -> decode. Zeigt wie steil die "Klippe" des Decoders ist.

### Interpretation:
- **Mean/Std Shift vorhanden** -> Latent Adapter (1x1 Conv + affine pro Kanal) als direktester Fix
- **Decoder hypersensitiv** (kleine Noise = grosse LPIPS-Degradation) -> Decoder-Robustification (LoRA + Noise-Augmentation)
- **Statistik OK, aber trotzdem schlechte Pixel** -> Problem ist higher-order (spektral/spatial), braucht LPL oder Pixel-Loss

---

## Part II: Prioritized Solutions (Revised after Mentor-Feedback)

**Prinzip: Sequentielle Ablation, nicht alles auf einmal.**
**Reihenfolge: Max. Lernwert pro GPU-Stunde.**

### Phase E0: Diagnostics (~30 min)
- Run `run_phase_e0_diagnostics.py`
- Ergebnis bestimmt welcher Fix zuerst kommt

### Phase E1: Erster Fix (basierend auf E0)

**Falls Distribution Shift (Mean/Std/Korrelation):**
-> Latent Adapter (kleines Netz vor Decoder, projiziert in richtige Verteilung)

**Falls Decoder hypersensitiv:**
-> Decoder-Robustification (LoRA + Latent-Noise-Augmentation beim Fine-Tuning)

**Falls weder noch (higher-order):**
-> Scheduled Self-Forcing allein (Exposure Bias eliminieren)

### Phase E2: Zweiter Fix (nach E1 Evaluation)
-> Latent Perceptual Loss (Pixel-LPIPS nach Decode)

### Phase E3: Dritter Fix (falls Konsistenz noch Problem)
-> Temporal Contrastive Slot Loss + Per-Slot Gating

---

## Part II.b: Alle Techniken im Detail (Research-Ergebnisse)

### Tier 1: High Impact, Low Effort

#### 1. Latent Perceptual Loss (LPL)
**Source**: Agent D (Latent Quality), "Boosting Latent Diffusion with Perceptual Objectives" (Berrada et al., Nov 2024)

**What**: Instead of `F.mse_loss(pred_delta, gt_delta)`, backpropagate through frozen VAE decoder features:
```python
L_LPL = sum_l  w_l * ||phi_l(z_pred) - phi_l(z_gt)||_2^2
```
where phi_l are intermediate decoder features.

**Why**: Directly addresses the root cause -- training loss now correlates with perceptual quality after decoding.

**Impact**: 6-22% FID improvement (paper). For our off-manifold predictions, gains likely larger.
**Effort**: Medium (hook into VAE decoder intermediate layers).
**Speed**: Zero inference cost. Training ~2-3x slower.

---

#### 2. Scheduled Self-Forcing (Training Procedure Change)
**Source**: Agent A (Neuroscience -- cerebellar forward model), Self-Forcing (ICLR 2025), Rolling Forcing (2025)

**What**: During DynamicsTransformer training, with increasing probability feed the model its own predictions instead of GT:
```python
p_self = min(0.5, epoch / n_epochs)
if random.random() < p_self and t > 1:
    slots_input = slots_pred_prev.detach()  # own prediction
else:
    slots_input = slots_gt[t-1]             # teacher forcing
```

**Why**: Eliminates exposure bias -- the model learns to handle its own imperfect predictions. This is why the Dynamics ablation shows only +2%: the model was never trained on noisy inputs.

**Impact**: Rollout degradation from 1.35x to potentially <1.15x. Makes Dynamics Transformer genuinely useful.
**Effort**: Very low -- training procedure change only, no new parameters.
**Speed**: Zero inference cost.

---

#### 3. SRVP-Style Variational Residual
**Source**: Agent B (Video SOTA), "Stochastic Latent Residual Video Prediction" (ICML 2020)

**What**: ResidualDecoderV2 outputs both mean AND variance. Sample delta from learned distribution:
```python
mu = self.out_proj(h)
log_var = self.var_proj(h)  # new: ~100K params
delta = mu + sigma * epsilon  # reparameterization trick
```
Train with ELBO: reconstruction + KL divergence.

**Why**: Deterministic decoder averages over futures, producing blurry/off-manifold predictions. Stochastic sampling produces sharper, on-manifold outputs.

**Impact**: LPIPS improvement 15-30%. Addresses Problem 5 (deterministic decoder).
**Effort**: Low (~100K extra parameters, loss function change).
**Speed**: ~0.1ms extra.

---

#### 4. Temporal Contrastive Slot Loss (SlotContrast)
**Source**: Agent A (Neuroscience -- temporal continuity), CVPR 2025 Oral (arxiv 2412.14295)

**What**: Add contrastive loss to DynamicsTransformer training:
```python
sim_matrix = cosine_similarity(slots_pred[:,:,None,:], slots_gt[:,None,:,:])
labels = torch.arange(64)
contrast_loss = cross_entropy(sim_matrix / 0.07, labels)
total_loss = pred_loss + 0.1 * contrast_loss
```

**Why**: Preserves per-slot identity during dynamics prediction, preventing slot-swapping that corrupts rollout.

**Impact**: Slot temporal consistency (FG-ARI: 22.2 -> 69.3 in paper). Improved rollout stability.
**Effort**: Very low -- pure loss function change, zero new parameters.
**Speed**: Zero inference cost.

---

#### 5. Periodic Feature Refresh (from Neural Video Compression)
**Source**: Agent B (Video SOTA), DCVC-FM (CVPR 2024)

**What**: Every K frames during rollout, replace accumulated latent with a fresh V3 full decode:
```python
if t % K == 0:  # e.g., K=4
    L_current = V3_decoder(current_slots)  # reset
else:
    L_current = L_prev + residual_decoder(...)  # continue
```

**Why**: Periodic resets prevent indefinite error accumulation, creating a "sawtooth" error pattern with bounded maximum.

**Impact**: Worst-case rollout error reduced 30-50%.
**Effort**: Very low -- one-line code change in rollout loop.
**Speed**: One extra V3 decode every K frames.

---

#### NEW: Latent Adapter (Mentor-Empfehlung -- "Missing Fix")
**Source**: Mentor-Feedback (Feb 19, 2026)

**What**: Ein minimales Netz VOR dem Decoder, das predicted Latents in die decoder-kompatible Verteilung projiziert:
```python
class LatentAdapter(nn.Module):
    """Maps predicted latents -> decoder-compatible latents.
    Option A: Per-channel affine (16 params)
    Option B: 1x1 Conv + residual blocks (~10-50K params)"""
    def __init__(self, n_channels=16):
        super().__init__()
        # Option A: Simplest -- per-channel affine
        self.scale = nn.Parameter(torch.ones(n_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(n_channels, 1, 1))

    def forward(self, z):
        return z * self.scale + self.bias

# Option B: Richer adapter with residual blocks
class LatentAdapterV2(nn.Module):
    def __init__(self, n_channels=16, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, n_channels, 1),
        )
        # Zero-init for residual connection
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return z + self.net(z)
```

Train on pairs: `(predicted_latent, real_latent)` with MSE + optional perceptual loss.

**Why**: Direktester Fix fur das Latent-Pixel-Disconnect. Projiziert predicted Latents in die richtige Verteilung, ohne den Predictor zu andern.

**Impact**: Hoch wenn das Hauptproblem statistischer Shift ist (Check 2 wird das zeigen).
**Effort**: Sehr gering (~16 params fur Option A, ~50K fur Option B). Training: Minuten.
**Speed**: ~0.01ms extra (negligible).

---

### Tier 2: High Impact, Medium Effort

#### 6. Precision-Weighted Residuals (Per-Token Confidence)
**Source**: Agent A (Neuroscience -- Free Energy Principle, Friston)

**What**: ResidualDecoder outputs delta AND per-token precision (confidence):
```python
delta = self.out_proj(h)
precision = torch.sigmoid(self.precision_proj(h))  # [B, 1024, 16]
weighted_delta = delta * precision  # uncertain tokens get small updates
```
Loss: `precision * (pred - gt)^2 + lambda * (1 - precision)` (penalize being too conservative).

**Why**: Tokens where the model is uncertain get small updates (closer to copy), preventing error injection. The brain uses exactly this mechanism (precision-weighted prediction errors).

**Impact**: Significantly reduced error accumulation. Interpretable confidence maps.
**Effort**: Low-medium (~6K extra params, modified loss).
**Speed**: ~0.1ms extra.

---

#### 7. Per-Slot Gating (Replace Binary Boundary Detector)
**Source**: Agent A (Neuroscience -- thalamic gating, RECOLLECT model)

**What**: Instead of one binary boundary signal per frame, compute per-slot gate:
```python
gate = sigmoid(gate_net(cat([slots_t, slots_t1], dim=-1)))  # [B, 64, 1]
slots_effective = gate * st1_pred + (1 - gate) * st  # per-slot blend
```

**Why**: In a 30fps video, most objects don't change frame-to-frame. Freezing unchanged slots reduces noise sources by ~80%. This makes Event Segmentation genuinely useful (current binary detector is too coarse).

**Impact**: Addresses Problem 2 (irrelevant principles) and reduces rollout error.
**Effort**: Low (~25K extra params).
**Speed**: ~0.1ms extra.

---

#### 8. VAE Decoder LoRA Adaptation
**Source**: Agent D (Latent Quality), SRL-VAE (2025)

**What**: Add LoRA layers (rank 8) to CogVideoX decoder, fine-tune on pairs of (predicted_latent, ground_truth_pixel):
```python
adapted_decoder = LoRA(vae.decoder, r=8, alpha=16)
# Fine-tune on your predicted latents
loss = mse(adapted_decoder(pred_latent), gt_pixel) + lpips(adapted_decoder(pred_latent), gt_pixel)
```

**Why**: Makes the decoder tolerant of our specific prediction distribution. Attacks the latent-pixel disconnect from the decoder side.

**Impact**: PSNR +0.8dB, rFID 3.71 -> 3.09 (paper). For our system, potentially dramatic.
**Effort**: Medium (need paired data from trained residual decoder + LoRA training).
**Speed**: ~1-2% inference overhead.

---

#### 9. Latent Flow Warping + Refinement
**Source**: Agent B (Video SOTA), LatentWarp (2024)

**What**: Decompose prediction into motion (warping) + appearance change (small residual):
```python
flow = flow_head(slots_t, slots_t1)  # predict latent-space flow [2, 32, 32]
L_warped = grid_sample(L_t, flow)    # warp preserves structure
L_t1 = L_warped + small_residual     # refine
```

**Why**: For most frames, motion accounts for 70-80% of the change. Warping preserves texture/structure much better than additive residuals. The refinement residual becomes much smaller and easier to predict.

**Impact**: MSE reduction 30-50%, dramatic rollout stability improvement.
**Effort**: Medium (flow prediction head ~50K params, differentiable warping, flow supervision via SEA-RAFT).
**Speed**: ~1-2ms extra.

---

#### 10. Step-Index Embedding for Dynamics
**Source**: Agent A (Neuroscience -- oscillatory phase coding), VideoMAR (2025) progressive temperature

**What**: Add a step embedding to DynamicsTransformer so it knows "I'm at step 5 of a rollout":
```python
phase = self.step_embed(torch.tensor([step_idx]))  # learned embedding
h = self.input_proj(x) + self.pos_embed + phase.unsqueeze(1)
```

**Why**: Without this, the model uses identical weights at step 1 and step 8, despite very different error characteristics. Step-aware dynamics can learn compensatory behavior for later steps.

**Impact**: Medium (10-20% rollout stability improvement).
**Effort**: Very low (one embedding layer).
**Speed**: Negligible.

---

### Tier 3: Exploratory / Longer-Term

#### 11. LCM-Style Consistency Corrector (1-2 step manifold projection)
**Source**: Agent B (Video SOTA), Latent Consistency Models (2023), rCM (ICLR 2026)

**What**: Train a small (2-5M param) denoiser that takes our predicted latent + light noise and projects it back onto the valid latent manifold in 1-2 steps.

**Impact**: LPIPS -25-40%. But adds 3-5ms per frame.
**Effort**: Medium-high (consistency distillation training).

#### 12. Slot Attractor Regularizer (VQ-style manifold projection for slots)
**Source**: Agent A (Neuroscience -- attractor dynamics, Nature Neuroscience 2024)

**What**: Learn a codebook of valid slot states. After dynamics prediction, softly project predicted slots toward nearest codebook entries.

**Impact**: Rollout stability improvement, prevents slot drift.
**Effort**: Medium (codebook training + VQ machinery).

#### 13. Mamba-Based Dynamics (Replace Transformer)
**Source**: Agent C (Perception), VideoMamba (ECCV 2024)

**What**: Replace DynamicsTransformer with Mamba SSM for O(n) complexity. Selective state updates naturally implement "what to remember/forget" per slot.

**Impact**: Better scaling to longer rollouts, potentially lower latency.
**Effort**: Medium-high (new architecture, retraining).

#### 14. Dreamer-4 Style: Predict Clean States, Not Deltas
**Source**: Agent C (Perception), Dreamer 4 (2025)

**What**: Instead of `L_{t+1} = L_t + Delta`, predict `L_{t+1}` directly but train with shortcut forcing. Dreamer 4 found this reduces error accumulation vs. delta prediction.

**Impact**: Could fundamentally change rollout characteristics.
**Effort**: High (architectural rethink of residual paradigm).
**Caution**: Our D2 ablation shows residual IS the key principle. Need careful A/B testing.

---

## Part III: Alternative Use Cases

### Ranked by Technical Fit

| Rank | Use Case | Key Insight | Time to MVP |
|------|----------|-------------|-------------|
| **1** | **Video Anomaly Detection** | Prediction error IS the product; single-step only (no rollout); slot = per-object localization; $12B market | 3-4 months |
| **2** | **Diffusion Acceleration** | Use our 2ms prediction as warm-start init for diffusion; skip diffusion on easy frames | 4-6 months |
| **3** | **Sports Analytics** | Boundary detector = highlight detection; prediction error = "interestingness" score; UCF-101 training relevant | 4-5 months |
| **4** | **Robotics World Model** | 2ms fits MPC control loops; slots = objects for planning; validated by SOLD (ICML 2025) | 6-9 months |

### Why Anomaly Detection Is the Best Pivot
Our biggest weakness -- **pixel quality degrades over rollout** -- becomes irrelevant:
- Only need **single-step prediction** (compare frame t+1 prediction with actual frame t+1)
- Prediction **error IS the signal** (high error = anomaly)
- **Slot-level decomposition** provides per-object anomaly localization (unique differentiator vs. competitors)
- **2ms latency** enables multi-camera real-time processing
- **Boundary detector** = automatic temporal event segmentation

---

## Part IV: Recommended Implementation Roadmap (Revised)

**Prinzip: Ein Fix pro Experiment. Sequentielle Ablation. Max. Lernwert pro GPU-Stunde.**

### Phase E0: Diagnostics (~30 min, ~$0.50)
- `run_phase_e0_diagnostics.py`
- 3 Checks: Decoder Oracle, Statistik-Vergleich, Sensitivity-Test
- **Ergebnis bestimmt die Reihenfolge der Fixes**

### Phase E1: Erster Fix (~2-4h, ~$3-5)
- **Wenn Distribution Shift**: Latent Adapter (per-channel affine oder 1x1 Conv)
- **Wenn Decoder hypersensitiv**: Decoder-Robustification (LoRA + Noise-Augmentation)
- **Wenn weder noch**: Scheduled Self-Forcing allein
- Evaluieren: Latent MSE, Pixel SSIM, LPIPS, FID

### Phase E2: Zweiter Fix (~2-4h, ~$3-5)
- Scheduled Self-Forcing (wenn nicht in E1 gemacht)
- ODER Latent Perceptual Loss (Pixel-LPIPS nach Decode)
- Evaluieren: gleiche Metriken + Rollout-Stabilitat

### Phase E3: Dritter Fix (~2-4h, ~$3-5)
- Temporal Contrastive Slot Loss (wenn Konsistenz noch Problem)
- ODER SRVP Variational Residual (wenn Blurriness noch Problem)
- Evaluieren

### Phase E4: Decision Point
Basierend auf E0-E3 Ergebnissen:
- **Pixel-Qualitat kompetitiv** -> weiter als Video-Prediction System, Paper finalisieren
- **Pixel-Qualitat-Gap bleibt** -> Pivot zu Anomaly Detection (Schwache = Starke)
- **Parallel**: Diffusion Acceleration als Infrastructure-Play explorieren

### Geschatzte Gesamtkosten: ~$15-25 (15-25h Lambda A100)

---

## Part V: Key References by Domain

### Neuroscience
- Precision-Weighted Prediction Errors: [arxiv 2506.23800](https://arxiv.org/abs/2506.23800)
- Thalamic Gating / RECOLLECT: [PLOS One 2024](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0316453)
- Attractor Dynamics: [Nature Neuroscience 2024](https://www.nature.com/articles/s41593-024-01766-5)
- Cerebellar Forward Model: [Frontiers 2020](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2020.00019/full)
- SlotContrast: [CVPR 2025 Oral, arxiv 2412.14295](https://arxiv.org/abs/2412.14295)
- Self-Forcing: [ICLR 2025](https://self-forcing.github.io/)
- FramePack: [NeurIPS 2025, arxiv 2504.12626](https://arxiv.org/abs/2504.12626)

### Video Generation & Compression
- SRVP Variational Residual: [ICML 2020](http://proceedings.mlr.press/v119/franceschi20a/franceschi20a.pdf)
- DCVC-FM Periodic Refresh: [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Neural_Video_Compression_with_Feature_Modulation_CVPR_2024_paper.pdf)
- Latent Consistency Models: [arxiv 2310.04378](https://arxiv.org/abs/2310.04378)
- rCM Distillation: [ICLR 2026, arxiv 2510.08431](https://arxiv.org/abs/2510.08431)
- LatentWarp: [OpenReview](https://openreview.net/forum?id=ZJHdiYDD5k)
- MAGVIT-v2 Discrete Tokens: [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/036912a83bdbb1fd792baf6532f102d8-Paper-Conference.pdf)

### Perception & World Models
- Dreamer 4: [arxiv 2509.24527](https://arxiv.org/pdf/2509.24527)
- V-JEPA 2: [arxiv 2506.09985](https://arxiv.org/abs/2506.09985)
- VideoMamba: [ECCV 2024, arxiv 2403.06977](https://arxiv.org/abs/2403.06977)
- PlaySlot: [ICML 2025, arxiv 2502.07600](https://arxiv.org/abs/2502.07600)
- SOLD: [ICML 2025, arxiv 2410.08822](https://arxiv.org/html/2410.08822v2)
- ReDRAW: [arxiv 2504.02252](https://arxiv.org/abs/2504.02252)
- 4D Gaussian Splatting: [CVPR 2024](https://guanjunwu.github.io/4dgs/)
- DINO-world: [arxiv 2507.19468](https://arxiv.org/abs/2507.19468)

### Latent Space Quality
- Latent Perceptual Loss: [arxiv 2411.04873](https://arxiv.org/abs/2411.04873)
- EQ-VAE: [ICML 2025, arxiv 2502.09509](https://arxiv.org/abs/2502.09509)
- SRL-VAE: [arxiv 2504.17219](https://arxiv.org/html/2504.17219v1)
- SDEdit: [arxiv 2108.01073](https://arxiv.org/abs/2108.01073)
- Sander Dieleman on Latent Space: [sander.ai/2025/04/15/latents](https://sander.ai/2025/04/15/latents.html)

### Alternative Use Cases
- Video Anomaly Detection Survey: [arxiv 2405.19387](https://arxiv.org/html/2405.19387v1)
- AI Industrial Defect Detection Market: [Future Market Insights](https://www.futuremarketinsights.com/reports/ai-industrial-defect-detection-market)
- NVIDIA Cosmos World Models: [nvidia.com/cosmos](https://www.nvidia.com/en-us/ai/cosmos/)

---

*Generated: 2026-02-19. Based on 5 parallel research agents covering neuroscience, video generation SOTA, real-time perception, latent space quality, and alternative use cases.*
