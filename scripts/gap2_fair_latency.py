#!/usr/bin/env python3
"""Gap 2 Fix: Fair Latency Benchmark at Equal Batch Sizes.

Benchmarks ALL models (Ours, UNet 20-step, UNet 50-step) at BOTH
batch_size=1 and batch_size=32 to enable a fair comparison.

Uses CUDA events for precise GPU timing (not wall-clock).
Reports: per-sample latency, throughput, and speedup ratios.

GPU time: ~5 minutes on A100.

Usage:
    python scripts/gap2_fair_latency.py --output results/json/gap2_latency.json
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import ResidualDecoderV2, DynamicsTransformer, BoundaryDetector


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def benchmark_fn(fn, n_warmup=10, n_iters=100):
    """Benchmark a function using CUDA events for precise GPU timing.

    Returns per-call latency in milliseconds.
    """
    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Measure with CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]

    for i in range(n_iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Drop top/bottom 10% for robustness
    times_ms.sort()
    trim = max(1, n_iters // 10)
    trimmed = times_ms[trim:-trim]

    return {
        "mean_ms": sum(trimmed) / len(trimmed),
        "std_ms": (sum((t - sum(trimmed)/len(trimmed))**2 for t in trimmed) / len(trimmed)) ** 0.5,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "median_ms": times_ms[n_iters // 2],
        "n_iters": n_iters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/json/gap2_latency.json")
    parser.add_argument("--n-iters", type=int, default=200,
                        help="Number of timed iterations per benchmark")
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required"
    log(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load Models ──
    log("Loading models...")
    res_decoder = ResidualDecoderV2().to(device).eval()
    dynamics = DynamicsTransformer().to(device).eval()
    boundary = BoundaryDetector().to(device).eval()

    # UNet baseline (matched architecture from D2 experiment)
    from diffusers import UNet2DModel
    unet = UNet2DModel(
        sample_size=32,
        in_channels=16,
        out_channels=16,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device).eval()

    unet_params = sum(p.numel() for p in unet.parameters()) / 1e6
    ours_params = (
        sum(p.numel() for p in res_decoder.parameters()) +
        sum(p.numel() for p in dynamics.parameters()) +
        sum(p.numel() for p in boundary.parameters())
    ) / 1e6
    log(f"UNet: {unet_params:.1f}M params | Ours: {ours_params:.2f}M params")

    # ── Benchmark at each batch size ──
    batch_sizes = [1, 8, 32]
    results = {"gpu": torch.cuda.get_device_name(0), "models": {}, "comparison": {}}

    for bs in batch_sizes:
        log(f"\n--- Batch Size {bs} ---")

        # Create dummy inputs
        dummy_x = torch.randn(bs, 16, 32, 32, device=device)
        dummy_t = torch.randint(0, 1000, (bs,), device=device)
        dummy_tokens = torch.randn(bs, 1024, 16, device=device)
        dummy_slots = torch.randn(bs, 64, 128, device=device)
        dummy_bd_feats = torch.randn(bs, 3, device=device)

        # --- UNet single forward pass ---
        def unet_single():
            with torch.no_grad():
                unet(dummy_x, dummy_t)

        unet_1step = benchmark_fn(unet_single, n_iters=args.n_iters)
        log(f"  UNet 1-step:    {unet_1step['mean_ms']:8.3f} ms (per batch)")

        # --- UNet 20-step denoising ---
        def unet_20step():
            with torch.no_grad():
                x = dummy_x.clone()
                for step in range(20):
                    t = torch.full((bs,), step * 50, device=device, dtype=torch.long)
                    x = unet(x, t).sample

        unet_20 = benchmark_fn(unet_20step, n_warmup=5, n_iters=min(50, args.n_iters))
        log(f"  UNet 20-step:   {unet_20['mean_ms']:8.3f} ms (per batch)")

        # --- UNet 50-step denoising ---
        def unet_50step():
            with torch.no_grad():
                x = dummy_x.clone()
                for step in range(50):
                    t = torch.full((bs,), step * 20, device=device, dtype=torch.long)
                    x = unet(x, t).sample

        unet_50 = benchmark_fn(unet_50step, n_warmup=3, n_iters=min(30, args.n_iters))
        log(f"  UNet 50-step:   {unet_50['mean_ms']:8.3f} ms (per batch)")

        # --- Our pipeline: dynamics + residual + boundary ---
        def ours_residual():
            with torch.no_grad():
                pred_slots = dynamics(dummy_slots)
                delta = res_decoder(dummy_tokens, dummy_slots, pred_slots)
                _ = dummy_tokens + delta

        ours_res = benchmark_fn(ours_residual, n_iters=args.n_iters)
        log(f"  Ours (residual): {ours_res['mean_ms']:7.3f} ms (per batch)")

        # --- Our pipeline: dynamics + boundary check ---
        def ours_with_boundary():
            with torch.no_grad():
                pred_slots = dynamics(dummy_slots)
                feats = boundary.compute_features(dummy_slots, pred_slots)
                logits = boundary(feats)
                delta = res_decoder(dummy_tokens, dummy_slots, pred_slots)
                _ = dummy_tokens + delta

        ours_full = benchmark_fn(ours_with_boundary, n_iters=args.n_iters)
        log(f"  Ours (full):    {ours_full['mean_ms']:8.3f} ms (per batch)")

        # --- Per-sample latency ---
        per_sample = {
            "unet_1step": unet_1step["mean_ms"] / bs,
            "unet_20step": unet_20["mean_ms"] / bs,
            "unet_50step": unet_50["mean_ms"] / bs,
            "ours_residual": ours_res["mean_ms"] / bs,
            "ours_full": ours_full["mean_ms"] / bs,
        }

        speedup_vs_50 = per_sample["unet_50step"] / per_sample["ours_full"]
        speedup_vs_20 = per_sample["unet_20step"] / per_sample["ours_full"]

        log(f"  Per-sample latency:")
        log(f"    UNet 50-step:  {per_sample['unet_50step']:8.3f} ms")
        log(f"    UNet 20-step:  {per_sample['unet_20step']:8.3f} ms")
        log(f"    Ours (full):   {per_sample['ours_full']:8.3f} ms")
        log(f"    Speedup vs 50: {speedup_vs_50:.1f}x")
        log(f"    Speedup vs 20: {speedup_vs_20:.1f}x")

        results["comparison"][f"bs{bs}"] = {
            "batch_size": bs,
            "batch_latency_ms": {
                "unet_1step": unet_1step,
                "unet_20step": unet_20,
                "unet_50step": unet_50,
                "ours_residual": ours_res,
                "ours_full": ours_full,
            },
            "per_sample_ms": per_sample,
            "speedup_vs_50step": speedup_vs_50,
            "speedup_vs_20step": speedup_vs_20,
        }

    # ── Summary Table (for paper) ──
    log("\n" + "=" * 70)
    log("SUMMARY TABLE (for paper Table 5 replacement)")
    log("=" * 70)
    header = f"{'Method':<22} {'BS=1 (ms)':<12} {'BS=8 (ms)':<12} {'BS=32 (ms)':<12}"
    log(header)
    log("-" * 58)

    for method in ["unet_50step", "unet_20step", "ours_full"]:
        label = {"unet_50step": "UNet 50-step", "unet_20step": "UNet 20-step", "ours_full": "Hybrid (Ours)"}[method]
        vals = []
        for bs in batch_sizes:
            v = results["comparison"][f"bs{bs}"]["per_sample_ms"][method]
            vals.append(f"{v:.3f}")
        log(f"{label:<22} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")

    log("")
    for bs in batch_sizes:
        s = results["comparison"][f"bs{bs}"]["speedup_vs_50step"]
        log(f"BS={bs}: {s:.1f}x speedup vs UNet 50-step")

    # ── Model info ──
    results["models"] = {
        "unet": {"params_M": unet_params, "architecture": "UNet2DModel (128,256,256,512)"},
        "ours": {
            "params_M": ours_params,
            "residual_decoder_params": sum(p.numel() for p in res_decoder.parameters()),
            "dynamics_params": sum(p.numel() for p in dynamics.parameters()),
            "boundary_params": sum(p.numel() for p in boundary.parameters()),
        },
    }

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
