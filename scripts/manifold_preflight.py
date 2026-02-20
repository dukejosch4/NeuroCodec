#!/usr/bin/env python3
"""Manifold Projector Pre-Flight Check.

Validates everything needed BEFORE running the manifold projector pipeline.
Zero GPU compute cost. Run this first to catch issues early.

Usage:
    python scripts/manifold_preflight.py --data-dir ~/
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def log(msg, level="INFO"):
    symbol = {"INFO": "  ", "OK": "\u2713 ", "WARN": "! ", "FAIL": "\u2717 "}
    print(f"[{symbol.get(level, '  ')}{level:4s}] {msg}", flush=True)


def check_section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.path.expanduser("~"))
    args = parser.parse_args()

    errors = []
    warnings = []

    # ── 1. Dependencies ──
    check_section("1. Dependencies")
    for pkg in ["torch", "numpy", "lpips", "scipy"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            log(f"{pkg} {ver}", "OK")
        except ImportError:
            log(f"{pkg} NOT FOUND", "FAIL")
            errors.append(f"Missing: {pkg}")

    # ── 2. GPU ──
    check_section("2. GPU")
    try:
        import torch
        if not torch.cuda.is_available():
            log("No CUDA GPU", "FAIL")
            errors.append("No CUDA")
        else:
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            log(f"{name} ({mem_gb:.1f} GB)", "OK")
            x = torch.randn(2, 2, device="cuda") @ torch.randn(2, 2, device="cuda")
            log("CUDA compute OK", "OK")
    except Exception as e:
        log(f"GPU failed: {e}", "FAIL")
        errors.append(str(e))

    # ── 3. Data Files ──
    check_section("3. Data Files")
    import torch

    for fname, req in [
        ("latents_2000videos.pt", True),
        ("aligned_video_slots.pt", True),
    ]:
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.exists(fpath):
            log(f"{fname}: NOT FOUND", "FAIL" if req else "WARN")
            if req:
                errors.append(f"Missing: {fname}")
            continue
        size_mb = os.path.getsize(fpath) / 1e6
        log(f"{fname}: {size_mb:.0f} MB", "OK")

        data = torch.load(fpath, map_location="cpu", weights_only=False)
        log(f"  Shape: {tuple(data.shape)}", "OK")
        if torch.isnan(data).any():
            log("  Contains NaN!", "FAIL")
            errors.append(f"{fname} NaN")
        if torch.isinf(data).any():
            log("  Contains Inf!", "FAIL")
            errors.append(f"{fname} Inf")
        log(f"  Range: [{data.float().min():.3f}, {data.float().max():.3f}]", "OK")
        del data

    # ── 4. Checkpoints ──
    check_section("4. Required Checkpoints")
    required_ckpts = [
        "residual_v2_spectral_beta0.01_best.pt",
        "dynamics_best_d2.pt",
        "slot_v2_64slots_2k_best.pt",
    ]
    for fname in required_ckpts:
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.exists(fpath):
            fpath = os.path.join(args.data_dir, "checkpoints", fname)
        if not os.path.exists(fpath):
            log(f"{fname}: NOT FOUND", "FAIL")
            errors.append(f"Missing checkpoint: {fname}")
        else:
            size_mb = os.path.getsize(fpath) / 1e6
            state = torch.load(fpath, map_location="cpu", weights_only=False)
            n_keys = len(state) if isinstance(state, dict) else "N/A"
            log(f"{fname}: {size_mb:.1f} MB, {n_keys} keys", "OK")
            del state

    # ── 5. Model Instantiation ──
    check_section("5. Model Instantiation")
    try:
        from src.models import ResidualDecoderV2, DynamicsTransformer, ManifoldProjector

        rd = ResidualDecoderV2()
        rd_params = sum(p.numel() for p in rd.parameters())
        log(f"ResidualDecoderV2: {rd_params:,} params", "OK")

        dt = DynamicsTransformer()
        dt_params = sum(p.numel() for p in dt.parameters())
        log(f"DynamicsTransformer: {dt_params:,} params", "OK")

        mp = ManifoldProjector()
        mp_params = sum(p.numel() for p in mp.parameters())
        log(f"ManifoldProjector: {mp_params:,} params", "OK")

        # Forward pass test
        with torch.no_grad():
            dummy_lt = torch.randn(2, 1024, 16)
            dummy_st = torch.randn(2, 64, 128)
            delta = rd(dummy_lt, dummy_st, dummy_st)
            assert delta.shape == (2, 1024, 16), f"ResDecoder shape: {delta.shape}"

            pred_latent = (dummy_lt + delta).permute(0, 2, 1).reshape(2, 16, 32, 32)
            proj = mp(pred_latent)
            assert proj.shape == (2, 16, 32, 32), f"Projector shape: {proj.shape}"

            # Verify zero-init: projector should be identity at init
            diff = (proj - pred_latent).abs().max().item()
            assert diff < 1e-6, f"Zero-init broken: max diff = {diff}"
            log("Forward pass + zero-init: OK", "OK")

            slots_next = dt(dummy_st)
            assert slots_next.shape == (2, 64, 128)
            log("DynamicsTransformer: OK", "OK")

        del rd, dt, mp
    except Exception as e:
        log(f"Model test failed: {e}", "FAIL")
        errors.append(f"Model: {e}")

    # ── 6. Checkpoint Loading ──
    check_section("6. Checkpoint Compatibility")
    try:
        from src.models import ResidualDecoderV2, DynamicsTransformer

        for ckpt_name, ModelClass in [
            ("residual_v2_spectral_beta0.01_best.pt", ResidualDecoderV2),
            ("dynamics_best_d2.pt", DynamicsTransformer),
        ]:
            fpath = os.path.join(args.data_dir, ckpt_name)
            if not os.path.exists(fpath):
                continue
            model = ModelClass()
            state = torch.load(fpath, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state)
            log(f"{ckpt_name} -> {ModelClass.__name__}: OK", "OK")
            del model, state
    except Exception as e:
        log(f"Checkpoint loading failed: {e}", "FAIL")
        errors.append(f"Ckpt load: {e}")

    # ── 7. Quick Pair Generation Test ──
    check_section("7. Pair Generation Dry Run (2 videos)")
    try:
        from src.models import ResidualDecoderV2, DynamicsTransformer

        latents_raw = torch.load(
            os.path.join(args.data_dir, "latents_2000videos.pt"),
            map_location="cpu", weights_only=False,
        )
        video_slots = torch.load(
            os.path.join(args.data_dir, "aligned_video_slots.pt"),
            map_location="cpu", weights_only=False,
        )

        if latents_raw.shape[1] == 16 and latents_raw.shape[2] == 9:
            latent_frames = latents_raw.permute(0, 2, 1, 3, 4)
        else:
            latent_frames = latents_raw

        N, T, C, H, W = latent_frames.shape
        log(f"Data: {N} videos, {T} frames, latent [{C},{H},{W}]", "OK")

        n_pairs_total = N * (T - 1)
        pair_size_mb = n_pairs_total * C * H * W * 4 / 1e6  # float32
        log(f"Training pairs: {n_pairs_total} ({pair_size_mb:.0f} MB per tensor)", "OK")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        rd = ResidualDecoderV2().to(device)
        state = torch.load(
            os.path.join(args.data_dir, "residual_v2_spectral_beta0.01_best.pt"),
            map_location="cpu", weights_only=False,
        )
        rd.load_state_dict(state)
        rd.eval()

        dt = DynamicsTransformer().to(device)
        dyn_state = torch.load(
            os.path.join(args.data_dir, "dynamics_best_d2.pt"),
            map_location="cpu", weights_only=False,
        )
        dt.load_state_dict(dyn_state)
        dt.eval()

        # Test 2 pairs
        lt = latent_frames[0, 0].to(device)  # [C, H, W]
        lt_tok = lt.flatten(1).unsqueeze(0).permute(0, 2, 1)  # [1, 1024, 16]
        st = video_slots[0, 0:1].to(device)  # [1, 64, 128]
        st1_dyn = dt(st)

        with torch.no_grad():
            delta = rd(lt_tok, st, st1_dyn)
            z_pred = (lt_tok + delta).permute(0, 2, 1).reshape(1, C, H, W)
            z_gt = latent_frames[0, 1:2].to(device)

        mse = torch.nn.functional.mse_loss(z_pred, z_gt).item()
        log(f"Test pair MSE: {mse:.6f}", "OK")
        log(f"z_pred range: [{z_pred.min():.3f}, {z_pred.max():.3f}]", "OK")
        log(f"z_gt   range: [{z_gt.min():.3f}, {z_gt.max():.3f}]", "OK")
        log(f"Residual (z_pred - z_gt) std: {(z_pred - z_gt).std():.4f}", "OK")

        del rd, dt, latents_raw, video_slots, latent_frames
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        log(f"Pair generation test failed: {e}", "FAIL")
        errors.append(f"Pair gen: {e}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  MANIFOLD PROJECTOR PRE-FLIGHT SUMMARY")
    print(f"{'='*60}")
    if errors:
        print(f"\n  BLOCKING ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    \u2717 {e}")
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    ! {w}")
    if not errors:
        print("\n  ALL CHECKS PASSED — ready to run pipeline")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
