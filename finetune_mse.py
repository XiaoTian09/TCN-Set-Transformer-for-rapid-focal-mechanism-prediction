#!/usr/bin/env python3
"""Transfer the raw-output MSE theoretical model to labeled INSTANCE data."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def early_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "finetune.json")
    known, _ = parser.parse_known_args()
    cfg = json.loads(known.config.read_text(encoding="utf-8"))
    gpu = str(cfg.get("gpu", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu.lower() == "cpu" else gpu
    return known.config, cfg


CONFIG_PATH, EARLY_CFG = early_config()
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from torch.utils.data import DataLoader
from train_tcn_settransformer_beachball_mse_pytorch import (
    H5BeachballDataset,
    MSEBeachballLoss,
    TCNSetTransformerBeachball,
    run_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(path, shuffle, cfg):
    dataset = H5BeachballDataset(path)
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + int(not shuffle))
    workers = int(cfg["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )
    return dataset, loader


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("train_h5", "validation_h5", "test_h5", "pretrained_checkpoint", "output_dir"):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str(value if value.is_absolute() else HERE / value)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    seed_all(int(cfg["seed"]))
    device = torch.device(
        "cpu" if args.cpu or str(cfg.get("gpu", "0")).lower() == "cpu" else "cuda"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；正式迁移训练需要GPU")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds, train_loader = make_loader(cfg["train_h5"], True, cfg)
    val_ds, val_loader = make_loader(cfg["validation_h5"], False, cfg)
    test_ds, test_loader = make_loader(cfg["test_h5"], False, cfg)

    model = TCNSetTransformerBeachball(dropout=float(cfg["dropout"])).to(device)
    source = load_checkpoint(model, cfg["pretrained_checkpoint"], device)
    criterion = MSEBeachballLoss().to(device)
    precision = cfg["precision"] if device.type == "cuda" else "float32"
    baseline = run_epoch(model, test_loader, criterion, device, precision)
    write_json(output_dir / "baseline_test_metrics.json", baseline)

    if cfg.get("freeze_tcn", True):
        for parameter in model.station_encoder.parameters():
            parameter.requires_grad = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=int(cfg["lr_patience"]),
        min_lr=float(cfg["min_learning_rate"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "float16"
    )
    best = float("inf")
    without_improvement = 0
    rows = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, precision,
            optimizer, scaler, log_interval=20, epoch_label=f"epoch {epoch}",
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, precision)
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        rows.append(row)
        with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        if val_metrics["loss"] < best - 1e-8:
            best = val_metrics["loss"]
            without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best,
                    "source_checkpoint": cfg["pretrained_checkpoint"],
                    "args": cfg,
                    "output_mode": "raw",
                    "loss": "unweighted_pixel_mse",
                },
                output_dir / "best_checkpoint.pt",
            )
        else:
            without_improvement += 1
            if without_improvement >= int(cfg["early_stopping_patience"]):
                break

    best_checkpoint = load_checkpoint(model, output_dir / "best_checkpoint.pt", device)
    transferred = run_epoch(model, test_loader, criterion, device, precision)
    comparison = {
        "baseline": baseline,
        "transferred": transferred,
        "delta_transferred_minus_baseline": {
            key: transferred[key] - baseline[key] for key in ("loss", "mae", "ssim")
        },
        "sample_counts": {
            "train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)
        },
        "best_epoch": int(best_checkpoint["epoch"]),
        "source_epoch": int(source.get("epoch", -1)),
        "test_used_for_model_selection": False,
        "output_mode": "raw",
        "loss": "unweighted_pixel_mse",
    }
    write_json(output_dir / "test_comparison.json", comparison)
    write_json(output_dir / "config.json", cfg)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

