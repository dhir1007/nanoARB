#!/usr/bin/env python3
"""Training script for NanoARB models."""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from config import Config, DEFAULT_CONFIG
from models.mamba_lob import MambaLOBModel, TransactionCostAwareLoss, export_to_onnx


def create_synthetic_data(
    num_samples: int = 10000,
    seq_len: int = 100,
    input_dim: int = 40,
    num_horizons: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic LOB data for testing.

    In production, this would load real CME/NASDAQ tick data.
    """
    # Synthetic features
    X = torch.randn(num_samples, seq_len, input_dim)

    # Synthetic targets (0: down, 1: neutral, 2: up)
    # Create correlated targets based on features
    feature_mean = X.mean(dim=(1, 2))
    targets = torch.zeros(num_samples, num_horizons, dtype=torch.long)

    for h in range(num_horizons):
        noise = torch.randn(num_samples) * 0.5
        scores = feature_mean + noise
        targets[:, h] = (scores > 0.3).long() + (scores > -0.3).long()

    return X, targets


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (X, y) in enumerate(tqdm(loader, desc="Training")):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(X)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for X, y in tqdm(loader, desc="Evaluating"):
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Per-horizon accuracy
    horizon_acc = []
    for h in range(all_preds.shape[1]):
        acc = (all_preds[:, h] == all_targets[:, h]).float().mean().item()
        horizon_acc.append(acc)

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
        "horizon_accuracy": horizon_acc,
    }


def benchmark_latency(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
    num_iterations: int = 1000,
) -> dict:
    """Benchmark model inference latency."""
    model.eval()
    X = torch.randn(1, *input_shape).to(device)

    # Warmup
    for _ in range(100):
        _ = model(X)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    import time
    latencies = []

    for _ in range(num_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        _ = model(X)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    latencies = sorted(latencies)

    return {
        "mean_ns": sum(latencies) / len(latencies),
        "median_ns": latencies[len(latencies) // 2],
        "p50_ns": latencies[int(len(latencies) * 0.50)],
        "p95_ns": latencies[int(len(latencies) * 0.95)],
        "p99_ns": latencies[int(len(latencies) * 0.99)],
        "min_ns": min(latencies),
        "max_ns": max(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="Train NanoARB models")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = DEFAULT_CONFIG

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic data (replace with real data loading)
    print("Creating synthetic data...")
    X_train, y_train = create_synthetic_data(8000, config.data.sequence_length)
    X_val, y_val = create_synthetic_data(1000, config.data.sequence_length)
    X_test, y_test = create_synthetic_data(1000, config.data.sequence_length)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    print("Creating model...")
    model = MambaLOBModel(
        input_dim=config.data.num_levels * config.data.features_per_level,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        d_state=config.model.d_state,
        d_conv=config.model.d_conv,
        expand=config.model.expand,
        num_horizons=len(config.data.prediction_horizons),
        dropout=config.model.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Setup training
    criterion = TransactionCostAwareLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Training loop
    best_val_loss = float("inf")
    best_model_path = output_dir / "best_model.pt"

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Horizon Acc: {[f'{a:.4f}' for a in val_metrics['horizon_accuracy']]}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, best_model_path)
            print(f"  Saved best model to {best_model_path}")

    # Final evaluation
    print("\nFinal evaluation on test set...")
    # weights_only=False needed because checkpoint includes config object
    model.load_state_dict(torch.load(best_model_path, weights_only=False)["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    print(f"Horizon Acc: {[f'{a:.4f}' for a in test_metrics['horizon_accuracy']]}")

    # Benchmark latency
    if args.benchmark:
        print("\nBenchmarking inference latency...")
        latency = benchmark_latency(
            model,
            (config.data.sequence_length, config.data.num_levels * config.data.features_per_level),
            device,
        )
        print("Latency (ns):")
        print(f"  Mean: {latency['mean_ns']:.0f}")
        print(f"  Median: {latency['median_ns']:.0f}")
        print(f"  P95: {latency['p95_ns']:.0f}")
        print(f"  P99: {latency['p99_ns']:.0f}")

    # Export to ONNX
    if args.export_onnx:
        print("\nExporting to ONNX...")
        onnx_path = output_dir / "model.onnx"
        export_to_onnx(model, str(onnx_path), config.data.sequence_length)
        print(f"Model exported to {onnx_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

