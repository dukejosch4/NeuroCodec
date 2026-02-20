#!/usr/bin/env python3
"""NeuroCodec Pre-Flight Check — run BEFORE any GPU experiments.

Validates all data, checkpoints, dependencies, and GPU state.
Zero compute cost. Catches problems before they waste A100 time.

Usage (run on Lambda BEFORE starting experiments):
    python scripts/preflight.py --data-dir ~/

Exit code 0 = all clear, 1 = blocking issue found.
"""

import argparse
import os
import sys
import time
import json

def log(msg, level="INFO"):
    symbol = {"INFO": "  ", "OK": "\u2713 ", "WARN": "! ", "FAIL": "\u2717 "}
    print(f"[{symbol.get(level, '  ')}{level:4s}] {msg}", flush=True)

def check_section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.path.expanduser("~"),
                        help="Directory containing .pt data files")
    args = parser.parse_args()

    errors = []
    warnings = []

    # ── 1. Python & Dependencies ──
    check_section("1. Python & Dependencies")
    log(f"Python: {sys.version.split()[0]}")

    required = [
        ("torch", "2.0"),
        ("numpy", None),
        ("diffusers", None),
        ("transformers", None),
        ("lpips", None),
        ("scipy", None),
    ]
    optional = [
        ("skimage", None),
        ("matplotlib", None),
        ("PIL", None),
    ]

    for pkg, min_ver in required:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            log(f"{pkg} {ver}", "OK")
        except ImportError:
            log(f"{pkg} NOT FOUND (required)", "FAIL")
            errors.append(f"Missing required package: {pkg}")

    for pkg, _ in optional:
        try:
            __import__(pkg)
            log(f"{pkg} available", "OK")
        except ImportError:
            log(f"{pkg} not found (optional, will install)", "WARN")
            warnings.append(f"Optional package missing: {pkg}")

    # ── 2. GPU & CUDA ──
    check_section("2. GPU & CUDA")
    try:
        import torch
        if not torch.cuda.is_available():
            log("CUDA not available!", "FAIL")
            errors.append("No CUDA GPU")
        else:
            n_gpus = torch.cuda.device_count()
            log(f"CUDA available, {n_gpus} GPU(s)", "OK")
            for i in range(n_gpus):
                name = torch.cuda.get_device_name(i)
                mem_gb = torch.cuda.get_device_properties(i).total_mem / 1e9
                log(f"  GPU {i}: {name} ({mem_gb:.1f} GB)", "OK")
                if mem_gb < 38:
                    log(f"  GPU {i} has < 40GB — VAE decode may OOM", "WARN")
                    warnings.append(f"GPU {i} has only {mem_gb:.1f} GB")

            # Quick CUDA sanity
            x = torch.randn(2, 2, device="cuda")
            y = x @ x.T
            assert y.shape == (2, 2)
            log("CUDA compute: OK", "OK")
    except Exception as e:
        log(f"GPU check failed: {e}", "FAIL")
        errors.append(f"GPU check: {e}")

    # ── 3. Data Files ──
    check_section("3. Data Files")
    data_files = {
        "latents_2000videos.pt": {
            "required": True,
            "expected_shapes": [(2000, 16, 9, 32, 32), (2000, 9, 16, 32, 32)],
        },
        "aligned_video_slots.pt": {
            "required": True,
            "expected_shapes": [(2000, 9, 64, 128)],
        },
    }

    for fname, spec in data_files.items():
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.exists(fpath):
            if spec["required"]:
                log(f"{fname}: NOT FOUND", "FAIL")
                errors.append(f"Missing data: {fname}")
            else:
                log(f"{fname}: not found (optional)", "WARN")
            continue

        size_mb = os.path.getsize(fpath) / 1e6
        log(f"{fname}: {size_mb:.0f} MB", "OK")

        try:
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            shape = tuple(data.shape)
            if shape in spec["expected_shapes"]:
                log(f"  Shape: {shape}", "OK")
            else:
                log(f"  Shape: {shape} (unexpected, expected one of {spec['expected_shapes']})", "WARN")
                warnings.append(f"{fname} shape mismatch: {shape}")

            # Basic value sanity
            if torch.isnan(data).any():
                log(f"  Contains NaN!", "FAIL")
                errors.append(f"{fname} contains NaN")
            elif torch.isinf(data).any():
                log(f"  Contains Inf!", "FAIL")
                errors.append(f"{fname} contains Inf")
            else:
                log(f"  Values: mean={data.float().mean():.4f}, std={data.float().std():.4f}", "OK")
            del data
        except Exception as e:
            log(f"  Failed to load: {e}", "FAIL")
            errors.append(f"Cannot load {fname}: {e}")

    # ── 4. Checkpoints ──
    check_section("4. Model Checkpoints")
    checkpoints = {
        "residual_decoder_v2_best.pt": {"required": True, "desc": "MSE-only residual decoder"},
        "residual_v2_spectral_beta0.01_best.pt": {"required": True, "desc": "Spectral loss variant"},
        "dynamics_best_d2.pt": {"required": True, "desc": "Dynamics transformer"},
        "dynamics_best.pt": {"required": False, "desc": "Dynamics (alt name)"},
        "boundary_detector_best_d2.pt": {"required": True, "desc": "Boundary detector"},
        "slot_v2_64slots_2k_best.pt": {"required": True, "desc": "Slot encoder (for Gap 3)"},
    }

    for fname, spec in checkpoints.items():
        # Check both data-dir and checkpoints subdir
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.exists(fpath):
            fpath = os.path.join(args.data_dir, "checkpoints", fname)
        if not os.path.exists(fpath):
            if spec["required"]:
                log(f"{fname}: NOT FOUND — {spec['desc']}", "FAIL")
                errors.append(f"Missing checkpoint: {fname}")
            else:
                log(f"{fname}: not found (optional) — {spec['desc']}", "WARN")
            continue

        size_mb = os.path.getsize(fpath) / 1e6
        log(f"{fname}: {size_mb:.1f} MB — {spec['desc']}", "OK")

        # Try loading
        try:
            state = torch.load(fpath, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                n_keys = len(state)
                log(f"  State dict with {n_keys} keys", "OK")
            else:
                log(f"  Type: {type(state).__name__}", "OK")
            del state
        except Exception as e:
            log(f"  Failed to load: {e}", "FAIL")
            errors.append(f"Cannot load {fname}: {e}")

    # ── 5. Model Instantiation ──
    check_section("5. Model Instantiation (CPU)")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from src.models import ResidualDecoderV2, DynamicsTransformer, BoundaryDetector

        rd = ResidualDecoderV2()
        n_params = sum(p.numel() for p in rd.parameters())
        log(f"ResidualDecoderV2: {n_params:,} params", "OK")

        dt = DynamicsTransformer()
        n_params = sum(p.numel() for p in dt.parameters())
        log(f"DynamicsTransformer: {n_params:,} params", "OK")

        bd = BoundaryDetector()
        n_params = sum(p.numel() for p in bd.parameters())
        log(f"BoundaryDetector: {n_params:,} params", "OK")

        # Quick forward pass sanity
        with torch.no_grad():
            dummy_lt = torch.randn(2, 1024, 16)
            dummy_st = torch.randn(2, 64, 128)
            out = rd(dummy_lt, dummy_st, dummy_st)
            assert out.shape == (2, 1024, 16), f"Wrong output shape: {out.shape}"
            log("Forward pass: OK", "OK")

        del rd, dt, bd
    except Exception as e:
        log(f"Model instantiation failed: {e}", "FAIL")
        errors.append(f"Model init: {e}")

    # ── 6. Checkpoint Loading Test ──
    check_section("6. Checkpoint Loading (weight compatibility)")
    try:
        from src.models import ResidualDecoderV2, DynamicsTransformer

        for ckpt_name, ModelClass in [
            ("residual_decoder_v2_best.pt", ResidualDecoderV2),
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
            log(f"{ckpt_name} -> {ModelClass.__name__}: loaded OK", "OK")
            del model, state
    except Exception as e:
        log(f"Checkpoint loading failed: {e}", "FAIL")
        errors.append(f"Checkpoint load: {e}")

    # ── 7. GPU Memory Estimate ──
    check_section("7. GPU Memory Estimate")
    try:
        import torch
        # CogVideoX VAE: ~5GB fp16
        # Slot encoder: ~50MB
        # Our models: ~10MB
        # Data batch (48 pairs): ~50MB
        # VAE decode workspace: ~8GB
        total_est_gb = 5.0 + 0.05 + 0.01 + 0.05 + 8.0
        log(f"Estimated peak VRAM: ~{total_est_gb:.1f} GB", "INFO")
        if torch.cuda.is_available():
            avail_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            if avail_gb >= total_est_gb + 2:
                log(f"GPU has {avail_gb:.1f} GB — sufficient", "OK")
            else:
                log(f"GPU has {avail_gb:.1f} GB — tight, may need smaller batches", "WARN")
                warnings.append("GPU memory may be tight")
    except Exception:
        pass

    # ── 8. Disk Space ──
    check_section("8. Disk Space")
    try:
        import shutil
        total, used, free = shutil.disk_usage(args.data_dir)
        free_gb = free / 1e9
        log(f"Free disk: {free_gb:.1f} GB", "OK" if free_gb > 5 else "WARN")
        if free_gb < 2:
            warnings.append(f"Low disk space: {free_gb:.1f} GB")
    except Exception:
        pass

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  PRE-FLIGHT SUMMARY")
    print(f"{'='*60}")
    if errors:
        print(f"\n  BLOCKING ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    \u2717 {e}")
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    ! {w}")
    if not errors and not warnings:
        print("\n  ALL CHECKS PASSED")

    if errors:
        print(f"\n  STATUS: BLOCKED — fix {len(errors)} error(s) before running experiments")
        sys.exit(1)
    else:
        print(f"\n  STATUS: READY TO GO")
        sys.exit(0)


if __name__ == "__main__":
    main()
