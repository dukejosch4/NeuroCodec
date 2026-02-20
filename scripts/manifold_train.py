#!/usr/bin/env python3
"""Train the ManifoldProjector on (z_pred, z_gt) pairs.

Uses MSE + Spectral loss (same as ResidualDecoder training).
Tiny model (~46K params), converges fast (~5-10 min on A100).

Usage:
    python scripts/manifold_train.py \
        --pairs ~/manifold_pairs.pt \
        --output ~/manifold_projector_best.pt \
        --epochs 100
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import ManifoldProjector
from src.losses import spectral_loss


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, required=True, help="Path to manifold_pairs.pt")
    parser.add_argument("--output", type=str, default="manifold_projector_best.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--spectral-beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    assert torch.cuda.is_available()
    log(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load Pairs ──
    log(f"Loading pairs from {args.pairs}...")
    data = torch.load(args.pairs, map_location="cpu", weights_only=False)
    z_pred = data["z_pred"]   # [N, 16, 32, 32]
    z_gt = data["z_gt"]       # [N, 16, 32, 32]
    split_idx = data["split_idx"]

    train_pred, val_pred = z_pred[:split_idx], z_pred[split_idx:]
    train_gt, val_gt = z_gt[:split_idx], z_gt[split_idx:]
    log(f"Train: {len(train_pred)} pairs, Val: {len(val_pred)} pairs")
    log(f"Baseline MSE (no projector): {data.get('avg_mse', 'N/A')}")

    del data, z_pred, z_gt

    # ── Baseline Metrics ──
    log("\nComputing baseline (identity projector)...")
    with torch.no_grad():
        # Compute on val set
        val_mse_baseline = F.mse_loss(val_pred, val_gt).item()
        val_spec_baseline = spectral_loss(val_pred[:256], val_gt[:256]).item()

        # Variance ratio baseline
        pred_std = val_pred.std(dim=(0, 2, 3))
        gt_std = val_gt.std(dim=(0, 2, 3))
        var_ratio_baseline = ((pred_std / (gt_std + 1e-8) - 1) ** 2).mean().item()

    log(f"  Val MSE (baseline):      {val_mse_baseline:.6f}")
    log(f"  Val Spectral (baseline): {val_spec_baseline:.6f}")
    log(f"  Val VarRatio (baseline): {var_ratio_baseline:.6f}")

    # ── Model ──
    model = ManifoldProjector().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"\nManifoldProjector: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * (len(train_pred) // args.batch_size + 1)

    def lr_lambda(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──
    log(f"\nTraining for {args.epochs} epochs (beta={args.spectral_beta})...")
    best_val = float("inf")
    history = []
    step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_pred))
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_spec = 0.0
        n_batches = 0

        for i in range(0, len(train_pred), args.batch_size):
            idx = perm[i:i + args.batch_size]
            zp = train_pred[idx].to(device)
            zg = train_gt[idx].to(device)

            z_proj = model(zp)

            l_mse = F.mse_loss(z_proj, zg)
            l_spec = spectral_loss(z_proj, zg) if args.spectral_beta > 0 else torch.tensor(0.0)
            loss = l_mse + args.spectral_beta * l_spec

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            epoch_loss += loss.item()
            epoch_mse += l_mse.item()
            epoch_spec += l_spec.item() if isinstance(l_spec, torch.Tensor) else 0.0
            n_batches += 1

        # Validation
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            val_mse = 0.0
            val_spec = 0.0
            val_n = 0
            with torch.no_grad():
                for i in range(0, len(val_pred), args.batch_size):
                    zp = val_pred[i:i + args.batch_size].to(device)
                    zg = val_gt[i:i + args.batch_size].to(device)
                    z_proj = model(zp)
                    val_mse += F.mse_loss(z_proj, zg).item()
                    val_spec += spectral_loss(z_proj, zg).item()
                    val_n += 1

            val_avg_mse = val_mse / val_n
            val_avg_spec = val_spec / val_n
            val_combined = val_avg_mse + args.spectral_beta * val_avg_spec

            improved = val_combined < best_val
            if improved:
                best_val = val_combined
                torch.save(model.state_dict(), args.output)

            # MSE improvement over baseline
            mse_improvement = (1 - val_avg_mse / val_mse_baseline) * 100

            entry = {
                "epoch": epoch,
                "train_loss": epoch_loss / n_batches,
                "train_mse": epoch_mse / n_batches,
                "train_spec": epoch_spec / n_batches,
                "val_mse": val_avg_mse,
                "val_spec": val_avg_spec,
                "val_combined": val_combined,
                "mse_improvement_pct": mse_improvement,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(entry)

            elapsed = time.time() - t_start
            log(
                f"Epoch {epoch:3d}: "
                f"val_mse={val_avg_mse:.6f} "
                f"val_spec={val_avg_spec:.4f} "
                f"MSE_improv={mse_improvement:+.1f}% "
                f"{'*' if improved else ' '} "
                f"[{elapsed:.0f}s]"
            )

    # ── Final Summary ──
    elapsed = time.time() - t_start
    log(f"\nTraining complete in {elapsed:.1f}s")
    log(f"Best val combined loss: {best_val:.6f}")

    # Reload best and compute final metrics
    model.load_state_dict(torch.load(args.output, map_location="cpu", weights_only=False))
    model.eval()

    with torch.no_grad():
        # Final val metrics
        all_proj = []
        for i in range(0, len(val_pred), args.batch_size):
            zp = val_pred[i:i + args.batch_size].to(device)
            all_proj.append(model(zp).cpu())
        val_projected = torch.cat(all_proj, dim=0)

        final_mse = F.mse_loss(val_projected, val_gt).item()
        final_spec = spectral_loss(val_projected[:256], val_gt[:256]).item()

        # Variance ratio
        proj_std = val_projected.std(dim=(0, 2, 3))
        gt_std = val_gt.std(dim=(0, 2, 3))
        var_ratio_proj = ((proj_std / (gt_std + 1e-8) - 1) ** 2).mean().item()

    log(f"\n{'='*60}")
    log(f"RESULTS COMPARISON")
    log(f"{'='*60}")
    log(f"{'Metric':<25s} {'Baseline':>12s} {'Projected':>12s} {'Change':>10s}")
    log(f"{'-'*60}")
    log(f"{'Val MSE':<25s} {val_mse_baseline:>12.6f} {final_mse:>12.6f} {(1-final_mse/val_mse_baseline)*100:>+9.1f}%")
    log(f"{'Val Spectral':<25s} {val_spec_baseline:>12.6f} {final_spec:>12.6f} {(1-final_spec/val_spec_baseline)*100:>+9.1f}%")
    log(f"{'Val Var Ratio Dev':<25s} {var_ratio_baseline:>12.6f} {var_ratio_proj:>12.6f} {(1-var_ratio_proj/var_ratio_baseline)*100:>+9.1f}%")

    # Save training log
    log_path = args.output.replace(".pt", "_log.json")
    summary = {
        "baseline": {
            "val_mse": val_mse_baseline,
            "val_spectral": val_spec_baseline,
            "val_var_ratio": var_ratio_baseline,
        },
        "projected": {
            "val_mse": final_mse,
            "val_spectral": final_spec,
            "val_var_ratio": var_ratio_proj,
        },
        "improvement": {
            "mse_pct": (1 - final_mse / val_mse_baseline) * 100,
            "spectral_pct": (1 - final_spec / val_spec_baseline) * 100,
            "var_ratio_pct": (1 - var_ratio_proj / var_ratio_baseline) * 100,
        },
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "spectral_beta": args.spectral_beta,
            "n_params": n_params,
        },
        "history": history,
        "training_time_s": elapsed,
    }
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
