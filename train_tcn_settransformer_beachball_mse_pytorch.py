#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyTorch版：共享TCN + Set Transformer + 沙滩球图像解码器（MSE对照实验）。

直接读取现有HDF5：

    waveform      (20, T, 3)，T 从 HDF5 的 window_length 属性读取
    geometry      (20, 4)
    station_mask  (20,)
    label         (128, 128)

本脚本仅把原复合BeachballLoss替换为原始逐像素MSE；模型、数据与训练配置
保持不变，用于区分模型框架与复合损失函数带来的改善。

默认使用BF16混合精度、多进程HDF5预取，并保存可恢复的完整检查点。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path


# =============================================================================
# 用户配置区：通常只需要修改这里，然后直接运行本脚本。
# 命令行参数仍可选地覆盖这些默认值。
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_TRAIN_H5 = SOURCE_DATA_DIR / "Traindata_tcn_settransformer.h5"
DEFAULT_VAL_H5 = SOURCE_DATA_DIR / "Validata_tcn_settransformer.h5"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "pretrain"
DEFAULT_GPU = "0"
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 1.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_DROPOUT = 0.1
DEFAULT_NUM_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_LOG_INTERVAL = 100
DEFAULT_PRECISION = "bfloat16"
DEFAULT_SEED = 42


def early_gpu_argument() -> str:
    """在import torch前读取--gpu，使CUDA_VISIBLE_DEVICES及时生效。"""
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--gpu":
            return sys.argv[index + 1]
    return os.environ.get("FCN_GPU_INDEX", DEFAULT_GPU)


EARLY_GPU = early_gpu_argument()
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if EARLY_GPU.strip().lower() == "cpu" else EARLY_GPU

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用原始MSE训练PyTorch TCN-SetTransformer沙滩球模型"
    )
    parser.add_argument(
        "--train-h5",
        default=str(DEFAULT_TRAIN_H5),
    )
    parser.add_argument(
        "--val-h5",
        default=str(DEFAULT_VAL_H5),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE
    )
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY
    )
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=DEFAULT_PREFETCH_FACTOR,
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help="每多少个batch输出一次训练进度；0表示仅输出epoch结果",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gpu", default=EARLY_GPU)
    parser.add_argument(
        "--precision",
        choices=("float32", "float16", "bfloat16"),
        default=DEFAULT_PRECISION,
    )
    parser.add_argument(
        "--resume",
        #default=str(DEFAULT_OUTPUT_DIR / "last_checkpoint.pt"),
        default=None,
        help="从last_checkpoint.pt或其他完整检查点恢复",
    )

    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="仅调试时限制训练样本数",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="仅调试时限制验证样本数",
    )
    return parser.parse_args()


def discover_group_names(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HDF5文件不存在: {path}")
    with h5py.File(path, "r") as h5_file:
        prefix = "data_label_"
        names = [
            name
            for name in h5_file.keys()
            if name.startswith(prefix) and name[len(prefix) :].isdigit()
        ]
        names.sort(key=lambda name: int(name[len(prefix) :]))
        if not names:
            raise ValueError(f"{path} 中没有 data_label_<id>")
        required = {"waveform", "geometry", "station_mask", "label"}
        missing = required - set(h5_file[names[0]].keys())
        if missing:
            raise KeyError(f"{path} 缺少字段: {sorted(missing)}")
    return names


def discover_data_shape(path: str | Path) -> tuple[int, int, int]:
    """读取并校验 HDF5 声明的台站数、时间长度和分量数。"""
    path = Path(path)
    with h5py.File(path, "r") as h5_file:
        names = [name for name in h5_file if name.startswith("data_label_")]
        if not names:
            raise ValueError(f"{path} 中没有 data_label_<id>")
        sample_shape = tuple(h5_file[names[0]]["waveform"].shape)
        station_count = int(h5_file.attrs.get("num_receivers", sample_shape[0]))
        window_length = int(h5_file.attrs.get("window_length", sample_shape[1]))
        expected = (station_count, window_length, 3)
        if sample_shape != expected:
            raise ValueError(
                f"{path} 首个 waveform 形状为 {sample_shape}，HDF5 属性声明 {expected}"
            )
    return expected


class H5BeachballDataset(Dataset):
    """每个DataLoader worker独立延迟打开HDF5，避免跨进程共享句柄。"""

    def __init__(
        self,
        path: str | Path,
        max_samples: int | None = None,
    ) -> None:
        self.path = str(Path(path))
        self.names = discover_group_names(self.path)
        self.waveform_shape = discover_data_shape(self.path)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples必须为正数")
            self.names = self.names[:max_samples]
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.names)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __getitem__(self, index: int):
        group = self._file()[self.names[index]]
        waveform = np.asarray(group["waveform"], dtype=np.float32)
        geometry = np.asarray(group["geometry"], dtype=np.float32)
        station_mask = np.asarray(group["station_mask"], dtype=np.float32)
        label = np.asarray(group["label"], dtype=np.float32)

        if waveform.shape != self.waveform_shape:
            raise ValueError(
                f"{self.names[index]}/waveform形状异常: {waveform.shape}，"
                f"预期 {self.waveform_shape}"
            )
        station_count = self.waveform_shape[0]
        if geometry.shape != (station_count, 4) or station_mask.shape != (station_count,):
            raise ValueError(f"{self.names[index]}的几何或mask形状异常")
        if label.shape != (128, 128):
            raise ValueError(
                f"{self.names[index]}/label形状异常: {label.shape}"
            )

        # torch.from_numpy不复制内存；DataLoader会负责拼成batch。
        return {
            "waveform": torch.from_numpy(waveform),
            "geometry": torch.from_numpy(geometry),
            "station_mask": torch.from_numpy(station_mask),
            "label": torch.from_numpy(label[None, ...]),
        }

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        self.close()


class ChannelLayerNorm1d(nn.Module):
    """对Conv1D的通道维做LayerNorm。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm1 = ChannelLayerNorm1d(input_channels)
        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = ChannelLayerNorm1d(output_channels)
        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + self.dropout(x)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class StationTCN(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Conv1d(3, 32, kernel_size=5, padding=2)
        blocks = []
        input_channels = 32
        for output_channels, dilation, kernel_size in (
            (32, 1, 5),
            (64, 2, 5),
            (96, 4, 3),
            (128, 8, 3),
        ):
            blocks.append(
                TCNResidualBlock(
                    input_channels,
                    output_channels,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )
            input_channels = output_channels
        self.blocks = nn.Sequential(*blocks)
        self.final_norm = ChannelLayerNorm1d(128)
        self.pool = TemporalAttentionPooling(128)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.stem(waveform)
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.pool(x)


class SetTransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout1(attended)
        return tokens + self.dropout2(self.ffn(self.norm2(tokens)))


class DecoderResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.skip = nn.Conv2d(input_channels, output_channels, 1)
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, 3, padding=1
        )
        self.norm1 = nn.GroupNorm(1, output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, 3, padding=1
        )
        self.norm2 = nn.GroupNorm(1, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        residual = self.skip(x)
        x = self.conv1(x)
        x = F.gelu(self.norm1(x))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class TCNSetTransformerBeachball(nn.Module):
    def __init__(
        self,
        station_count: int = 20,
        token_dim: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.station_count = station_count
        self.station_encoder = StationTCN(dropout)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )
        self.token_projection = nn.Linear(160, token_dim)
        self.event_token = nn.Parameter(
            torch.empty(1, 1, token_dim)
        )
        nn.init.xavier_uniform_(self.event_token)
        self.transformer_blocks = nn.ModuleList(
            [
                SetTransformerBlock(
                    model_dim=token_dim,
                    num_heads=4,
                    ff_dim=384,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        )
        self.event_norm = nn.LayerNorm(token_dim)
        self.event_dense = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.decoder_seed = nn.Sequential(
            nn.Linear(256, 8 * 8 * 128),
            nn.GELU(),
        )
        self.decoder_blocks = nn.Sequential(
            DecoderResidualBlock(128, 128),
            DecoderResidualBlock(128, 64),
            DecoderResidualBlock(64, 32),
            DecoderResidualBlock(32, 16),
        )
        self.output = nn.Conv2d(16, 1, 1)

    def forward(
        self,
        waveform: torch.Tensor,
        geometry: torch.Tensor,
        station_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, station_count, time_count, components = waveform.shape
        if station_count != self.station_count:
            raise ValueError(
                f"输入台站数为{station_count}，预期{self.station_count}"
            )

        # 一次编码B×S条波形，避免TimeDistributed的小算子调度。
        waveform = waveform.reshape(
            batch_size * station_count,
            time_count,
            components,
        ).transpose(1, 2)
        waveform_features = self.station_encoder(waveform)
        waveform_features = waveform_features.reshape(
            batch_size, station_count, -1
        )
        geometry_features = self.geometry_encoder(geometry)
        tokens = self.token_projection(
            torch.cat([waveform_features, geometry_features], dim=-1)
        )

        event_token = self.event_token.expand(batch_size, -1, -1)
        tokens = torch.cat([event_token, tokens], dim=1)
        valid = torch.cat(
            [
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=station_mask.device,
                ),
                station_mask > 0.5,
            ],
            dim=1,
        )
        key_padding_mask = ~valid
        for block in self.transformer_blocks:
            tokens = block(tokens, key_padding_mask)

        event_feature = self.event_dense(self.event_norm(tokens[:, 0]))
        x = self.decoder_seed(event_feature).reshape(
            batch_size, 128, 8, 8
        )
        x = self.decoder_blocks(x)
        # 与旧FCN一致，返回无激活、无范围约束的FP32图像预测，
        # 直接与[0, 1]标签计算逐像素MSE。
        return self.output(x).float()


def gaussian_kernel(
    size: int = 11,
    sigma: float = 1.5,
    device: torch.device | None = None,
) -> torch.Tensor:
    coordinates = torch.arange(
        size, dtype=torch.float32, device=device
    ) - (size - 1) / 2
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


def ssim_per_sample(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> torch.Tensor:
    """单通道SSIM，参数与TensorFlow默认11×11、sigma=1.5一致。"""
    kernel = gaussian_kernel(device=y_true.device)
    # 与tf.image.ssim一样只使用完整的11x11窗口，不做零填充。零填充会在
    # 白色背景边缘引入人为高对比，并加剧 E[x²]-E[x]² 的消减误差。
    mu_true = F.conv2d(y_true, kernel, padding=0)
    mu_pred = F.conv2d(y_pred, kernel, padding=0)
    mu_true_sq = mu_true.square()
    mu_pred_sq = mu_pred.square()
    mu_cross = mu_true * mu_pred
    sigma_true = F.conv2d(
        y_true.square(), kernel, padding=0
    ) - mu_true_sq
    sigma_pred = F.conv2d(
        y_pred.square(), kernel, padding=0
    ) - mu_pred_sq
    sigma_cross = F.conv2d(
        y_true * y_pred, kernel, padding=0
    ) - mu_cross
    # 理论方差非负，但卷积求和的浮点消减可能产生很小的负值。若直接使用，
    # SSIM分母可能为负，再被clamp成1e-8，最终产生数千的伪SSIM。
    sigma_true = sigma_true.clamp_min(0.0)
    sigma_pred = sigma_pred.clamp_min(0.0)
    covariance_limit = torch.sqrt(
        (sigma_true * sigma_pred).clamp_min(0.0) + 1.0e-12
    )
    sigma_cross = torch.maximum(
        torch.minimum(sigma_cross, covariance_limit),
        -covariance_limit,
    )
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (
        (mu_true_sq + mu_pred_sq + c1)
        * (sigma_true + sigma_pred + c2)
    )
    ssim_map = numerator / denominator.clamp_min(c1 * c2)
    # 数值舍入下可能略越界；显式限制保证1-SSIM始终是非负损失。
    return ssim_map.clamp(-1.0, 1.0).mean(dim=(1, 2, 3))


class MSEBeachballLoss(nn.Module):
    """与旧FCN的 ``loss='mse'`` 对齐的无权重逐像素MSE。"""

    def forward(
        self,
        prediction: torch.Tensor,
        y_true: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = prediction.float()
        y_true = y_true.float()
        loss = F.mse_loss(prediction, y_true, reduction="mean")
        ssim = ssim_per_sample(y_true, prediction).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "MSE loss出现NaN/Inf；请检查输入与混合精度"
            )
        return loss, {
            "mae": (y_true - prediction).abs().mean(),
            "ssim": ssim,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    args: argparse.Namespace,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed + (0 if shuffle else 1))
    loader_args = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if args.num_workers > 0:
        loader_args.update(
            {
                "persistent_workers": True,
                "prefetch_factor": args.prefetch_factor,
            }
        )
    return DataLoader(**loader_args)


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def autocast_context(
    device: torch.device,
    precision: str,
):
    enabled = device.type == "cuda" and precision != "float32"
    dtype = (
        torch.bfloat16
        if precision == "bfloat16"
        else torch.float16
    )
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MSEBeachballLoss,
    device: torch.device,
    precision: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    log_interval: int = 0,
    epoch_label: str = "",
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {
        "loss": 0.0,
        "mae": 0.0,
        "ssim": 0.0,
    }
    sample_count = 0
    prediction_probe = []
    start_time = time.perf_counter()

    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        batch_count = batch["label"].shape[0]
        if training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            with autocast_context(device, precision):
                prediction = model(
                    batch["waveform"],
                    batch["geometry"],
                    batch["station_mask"],
                )
                loss, metrics = criterion(
                    prediction, batch["label"]
                )

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

        sums["loss"] += float(loss.detach()) * batch_count
        for key in ("mae", "ssim"):
            sums[key] += float(metrics[key].detach()) * batch_count
        sample_count += batch_count

        if (
            training
            and log_interval > 0
            and (
                batch_index % log_interval == 0
                or batch_index == len(loader)
            )
        ):
            elapsed = time.perf_counter() - start_time
            print(
                f"{epoch_label} step {batch_index}/{len(loader)} - "
                f"loss: {float(loss.detach()):.5f} - "
                f"{sample_count / max(elapsed, 1.0e-8):.1f} samples/s",
                flush=True,
            )

        if not training and sum(len(value) for value in prediction_probe) < 32:
            prediction_probe.append(
                prediction[: 32 - sum(len(v) for v in prediction_probe)]
                .detach()
                .float()
                .cpu()
            )

    result = {
        key: value / sample_count for key, value in sums.items()
    }
    result["seconds"] = time.perf_counter() - start_time
    result["samples_per_second"] = (
        sample_count / max(result["seconds"], 1.0e-8)
    )
    if prediction_probe:
        probe = torch.cat(prediction_probe, dim=0)
        result["prediction_sample_std"] = float(
            probe.std(dim=0).mean()
        )
    return result


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_loss: float,
    epochs_without_improvement: int,
    args: argparse.Namespace,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "args": vars(args),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size必须为正数，num-workers不能为负数")
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if args.gpu.strip().lower() != "cpu" and device.type != "cuda":
        raise RuntimeError(
            "请求GPU训练，但PyTorch未发现GPU；请检查--gpu和驱动"
        )
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "Compute capability: "
            f"{torch.cuda.get_device_capability(0)}"
        )
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        output_dir / "config.json", "w", encoding="utf-8"
    ) as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)

    train_dataset = H5BeachballDataset(
        args.train_h5, args.max_train_samples
    )
    val_dataset = H5BeachballDataset(
        args.val_h5, args.max_val_samples
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(
        f"Batch size: {args.batch_size}; "
        f"train steps: {len(train_loader)}; "
        f"val steps: {len(val_loader)}"
    )

    model = TCNSetTransformerBeachball(
        dropout=args.dropout
    ).to(device)
    criterion = MSEBeachballLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        # RTX 5090当前环境中优先使用稳定的逐张量实现，避开
        # _multi_tensor_adam foreach路径在首步被中断时难以诊断的问题。
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.min_learning_rate,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda" and args.precision == "float16"
        ),
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(
        f"Parameters: {parameter_count:,} "
        f"({parameter_count * 4 / 1024**2:.2f} MiB in FP32)"
    )

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        print(
            f"Resumed from {args.resume}; "
            f"next epoch={start_epoch + 1}"
        )

    history_path = output_dir / "training_log.csv"
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_mae",
        "train_ssim",
        "train_seconds",
        "train_samples_per_second",
        "val_loss",
        "val_mae",
        "val_ssim",
        "val_prediction_sample_std",
        "val_seconds",
    ]
    write_header = not history_path.exists() or start_epoch == 0
    file_mode = "w" if write_header else "a"

    with open(
        history_path, file_mode, newline="", encoding="utf-8"
    ) as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                args.precision,
                optimizer=optimizer,
                scaler=scaler,
                log_interval=args.log_interval,
                epoch_label=f"Epoch {epoch + 1}/{args.epochs}",
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                args.precision,
            )
            scheduler.step(val_metrics["loss"])
            learning_rate = optimizer.param_groups[0]["lr"]

            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "train_ssim": train_metrics["ssim"],
                "train_seconds": train_metrics["seconds"],
                "train_samples_per_second": train_metrics[
                    "samples_per_second"
                ],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_ssim": val_metrics["ssim"],
                "val_prediction_sample_std": val_metrics.get(
                    "prediction_sample_std", float("nan")
                ),
                "val_seconds": val_metrics["seconds"],
            }
            writer.writerow(row)
            history_file.flush()

            save_checkpoint(
                output_dir / "last_checkpoint.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_loss,
                epochs_without_improvement,
                args,
            )
            if improved:
                save_checkpoint(
                    output_dir / "best_checkpoint.pt",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_loss,
                    epochs_without_improvement,
                    args,
                )

            print(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"{train_metrics['seconds']:.1f}s - "
                f"loss: {train_metrics['loss']:.5f} - "
                f"ssim: {train_metrics['ssim']:.5f} - "
                f"val_loss: {val_metrics['loss']:.5f} - "
                f"val_ssim: {val_metrics['ssim']:.5f} - "
                "val_prediction_sample_std: "
                f"{val_metrics.get('prediction_sample_std', float('nan')):.7f} - "
                f"lr: {learning_rate:.2e}"
            )
            if (
                val_metrics.get("prediction_sample_std", 1.0)
                < 1.0e-5
            ):
                print(
                    "WARNING: 不同验证输入的预测几乎相同，"
                    "模型可能塌缩为固定模板。"
                )
            if (
                epochs_without_improvement
                >= args.early_stopping_patience
            ):
                print(
                    "Early stopping: validation loss has not "
                    f"improved for {epochs_without_improvement} epochs."
                )
                break

    train_dataset.close()
    val_dataset.close()
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
