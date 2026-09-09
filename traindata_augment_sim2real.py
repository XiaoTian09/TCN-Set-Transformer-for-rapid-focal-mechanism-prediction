#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可组合的 Synthetic-to-Real 波形数据增广。

该脚本是独立版本，不修改 ``traindata_augment_utf8.py``。输入仍为脚本1生成的
完整理论波形 HDF5（data 的通道顺序为 Z/E/N）和事件元数据 CSV，输出为
``(20, 52, 5)`` 的 Z/E/N/方位角/离源角数据。

支持的增广名称：

* shift          : 事件公共平移 + 台站残差平移
* station_gain   : 每台站一个 lognormal 标量增益
* component_gain : 每台站、每分量独立的轻微增益
* real_noise     : Instance 三分量真实噪声，按随机目标 SNR 注入
* spectral       : 轻微频率缩放 + Hilbert 相位旋转

``recipes`` 可以让任意方法单独或组合使用，例如：

    base;shift;real_noise;shift+station_gain+component_gain+real_noise+spectral

噪声 HDF5 的结构应为 ``/data/{trace_name}``，数组形状 (3, 12000)，分量顺序
E/N/Z。噪声先从 100 Hz 降到 1 Hz，处理成120点，再反射填充为128点。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import norm
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NOISE_LIBRARY = SCRIPT_DIR / "Instance_noise_1hz_128_5GiB.h5"

VALID_AUGMENTATIONS = {
    "shift",
    "station_gain",
    "component_gain",
    "real_noise",
    "spectral",
}
NORMALIZATION_MODES = {"legacy", "station_joint", "global_joint"}
EPS = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成可组合 Sim2Real 增广训练集")
    parser.add_argument("--input-h5", default="Traindata_filter_fulllength_3ch.h5")
    parser.add_argument("--input-csv", default="Traindata_filter_fulllength_3ch_metadata.csv")
    parser.add_argument("--output-h5", default="Traindata_augmented_sim2real_5ch.h5")
    parser.add_argument(
        "--noise-library",
        default=str(DEFAULT_NOISE_LIBRARY),
        help="build_noise_library_1hz.py generated preprocessed noise library",
    )
    parser.add_argument("--num-station-selections", type=int, default=5)
    parser.add_argument("--num-receivers", type=int, default=20)
    parser.add_argument("--base-start", type=int, default=45)
    parser.add_argument("--window-len", type=int, default=52)
    parser.add_argument(
        "--recipes",
        default="base;shift;shift+station_gain+component_gain+real_noise+spectral",
        help="分号分隔的版本；每个版本用+组合增广，base表示无增广",
    )
    parser.add_argument(
        "--normalization",
        choices=sorted(NORMALIZATION_MODES),
        default="legacy",
        help="legacy=逐台站逐分量；station_joint=每台站三分量联合；global_joint=全样本联合",
    )
    parser.add_argument("--event-shift-max", type=int, default=2)
    parser.add_argument("--station-shift-std", type=float, default=1.0)
    parser.add_argument("--station-shift-clip", type=int, default=3)
    parser.add_argument(
        "--station-gain-sigma",
        type=float,
        default=0.26,
        help="lognormal的sigma；0.26时约95%%落在0.60--1.66",
    )
    parser.add_argument("--component-gain-min", type=float, default=0.9)
    parser.add_argument("--component-gain-max", type=float, default=1.1)
    parser.add_argument("--snr-db-min", type=float, default=5.0)
    parser.add_argument("--snr-db-max", type=float, default=25.0)
    parser.add_argument("--frequency-scale-min", type=float, default=0.98)
    parser.add_argument("--frequency-scale-max", type=float, default=1.02)
    parser.add_argument("--phase-max-deg", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mapminmax(array: np.ndarray) -> np.ndarray:
    lo = float(np.min(array))
    span = float(np.max(array) - lo)
    if span <= EPS:
        return np.zeros_like(array)
    return (array - lo) / span


def normalize_waveforms(data: np.ndarray, mode: str) -> np.ndarray:
    """对 (station, time, component) 的波形归一化。"""
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"未知归一化模式: {mode}")
    result = np.empty_like(data, dtype=np.float64)
    if mode == "legacy":
        for station_index in range(data.shape[0]):
            for component_index in range(data.shape[2]):
                result[station_index, :, component_index] = mapminmax(
                    data[station_index, :, component_index]
                )
    elif mode == "station_joint":
        for station_index in range(data.shape[0]):
            result[station_index] = mapminmax(data[station_index])
    else:
        result[:] = mapminmax(data)
    return result.astype(np.float32)


def center_waveforms(data: np.ndarray) -> np.ndarray:
    """去掉输入 [0,1] 归一化留下的每道直流偏置。"""
    return data.astype(np.float64) - np.mean(data, axis=1, keepdims=True)


def process_az_takeoff(
    azimuth: np.ndarray, takeoff: np.ndarray, output_length: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(1, output_length + 1)
    azimuth_encoded = np.zeros((len(azimuth), output_length), dtype=np.float32)
    takeoff_encoded = np.zeros((len(takeoff), output_length), dtype=np.float32)
    for index in range(len(azimuth)):
        azimuth_mu = output_length / 360.0 * float(azimuth[index])
        takeoff_mu = output_length / 180.0 * float(takeoff[index])
        azimuth_encoded[index] = mapminmax(norm.pdf(x, azimuth_mu, 4))
        takeoff_encoded[index] = mapminmax(norm.pdf(x, takeoff_mu, 4))
    return azimuth_encoded, takeoff_encoded


def parse_recipes(text: str) -> list[tuple[str, tuple[str, ...]]]:
    recipes: list[tuple[str, tuple[str, ...]]] = []
    for raw_recipe in text.split(";"):
        name = raw_recipe.strip().lower()
        if not name:
            continue
        methods: tuple[str, ...]
        if name == "base":
            methods = ()
        else:
            methods = tuple(part.strip() for part in name.split("+") if part.strip())
        unknown = set(methods) - VALID_AUGMENTATIONS
        if unknown:
            raise ValueError(f"增广版本 {name!r} 包含未知方法: {sorted(unknown)}")
        if len(methods) != len(set(methods)):
            raise ValueError(f"增广版本 {name!r} 中存在重复方法")
        recipes.append((name, methods))
    if not recipes:
        raise ValueError("recipes 至少需要一个版本")
    return recipes


def validate_ranges(args: argparse.Namespace) -> None:
    if args.event_shift_max < 0 or args.station_shift_std < 0 or args.station_shift_clip < 0:
        raise ValueError("平移参数必须非负")
    if args.component_gain_min <= 0 or args.component_gain_max < args.component_gain_min:
        raise ValueError("分量增益范围无效")
    if args.snr_db_max < args.snr_db_min:
        raise ValueError("SNR范围无效")
    if args.frequency_scale_min <= 0 or args.frequency_scale_max < args.frequency_scale_min:
        raise ValueError("频率缩放范围无效")


def make_rng(seed: int, event_id: int, selection_index: int, recipe_index: int) -> np.random.Generator:
    # SeedSequence 避免手工整数种子在大量事件时发生碰撞。
    return np.random.default_rng(
        np.random.SeedSequence([seed, event_id, selection_index, recipe_index])
    )


def frequency_scale_trace(trace: np.ndarray, scale: float) -> np.ndarray:
    """围绕记录中心做轻微时间缩放；scale>1 对应频率略升高。"""
    count = trace.size
    center = (count - 1) / 2.0
    output_x = np.arange(count, dtype=np.float64)
    input_x = center + (output_x - center) * scale
    return np.interp(input_x, output_x, trace, left=trace[0], right=trace[-1])


def apply_spectral_perturbation(
    data: np.ndarray,
    rng: np.random.Generator,
    frequency_scale_range: tuple[float, float],
    phase_max_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各台站使用共同作用于三分量的频率缩放和相位旋转。"""
    output = np.empty_like(data)
    frequency_scales = rng.uniform(*frequency_scale_range, size=data.shape[0])
    phase_degrees = rng.uniform(-phase_max_deg, phase_max_deg, size=data.shape[0])
    for station_index in range(data.shape[0]):
        phase_radians = math.radians(float(phase_degrees[station_index]))
        for component_index in range(data.shape[2]):
            stretched = frequency_scale_trace(
                data[station_index, :, component_index],
                float(frequency_scales[station_index]),
            )
            analytic = signal.hilbert(stretched)
            output[station_index, :, component_index] = np.real(
                analytic * np.exp(1j * phase_radians)
            )
    return output, frequency_scales, phase_degrees


def draw_shifts(
    station_count: int,
    rng: np.random.Generator,
    event_max: int,
    residual_std: float,
    residual_clip: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    event_shift = int(rng.integers(-event_max, event_max + 1)) if event_max else 0
    residual = np.rint(rng.normal(0.0, residual_std, station_count)).astype(int)
    residual = np.clip(residual, -residual_clip, residual_clip)
    return event_shift, residual, event_shift + residual


def slice_with_station_shifts(
    data: np.ndarray, base_start: int, window_length: int, shifts: np.ndarray
) -> np.ndarray:
    output = np.empty((data.shape[0], window_length, data.shape[2]), dtype=np.float64)
    for station_index, shift in enumerate(shifts):
        start = base_start + int(shift)
        end = start + window_length
        if start < 0 or end > data.shape[1]:
            raise ValueError(
                f"平移后窗口越界: station={station_index}, start={start}, "
                f"end={end}, full_length={data.shape[1]}"
            )
        output[station_index] = data[station_index, start:end]
    return output


def apply_station_gain(
    data: np.ndarray, rng: np.random.Generator, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    gains = rng.lognormal(mean=0.0, sigma=sigma, size=data.shape[0])
    return data * gains[:, None, None], gains


def apply_component_gain(
    data: np.ndarray,
    rng: np.random.Generator,
    gain_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    gains = rng.uniform(*gain_range, size=(data.shape[0], data.shape[2]))
    return data * gains[:, None, :], gains


class NoiseLibrary:
    """Fast random access to a preprocessed (N,128,3) Z/E/N noise library."""

    def __init__(self, library_path: str | Path) -> None:
        self.h5 = h5py.File(library_path, "r")
        if "noise" not in self.h5 or "trace_name" not in self.h5:
            raise KeyError(
                "Noise library must contain /noise and /trace_name; "
                "run build_noise_library_1hz.py first"
            )
        self.noise = self.h5["noise"]
        self.names = self.h5["trace_name"]
        if self.noise.ndim != 3 or self.noise.shape[1:] != (128, 3):
            raise ValueError(f"Invalid preprocessed noise shape: {self.noise.shape}")
        if len(self.names) != len(self.noise):
            raise ValueError("Noise and trace_name lengths do not match")

    def close(self) -> None:
        self.h5.close()

    def sample_many(
        self, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[str]]:
        replace = count > len(self.noise)
        indices = np.asarray(
            rng.choice(len(self.noise), size=count, replace=replace), dtype=np.int64
        )
        if replace:
            arrays = np.stack([self.noise[int(index)] for index in indices])
        else:
            # h5py fancy indices must be increasing. Read once, then restore random order.
            order = np.argsort(indices)
            sorted_values = self.noise[indices[order]]
            arrays = sorted_values[np.argsort(order)]
        raw_names = [self.names[int(index)] for index in indices]
        names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_names
        ]
        return np.asarray(arrays, dtype=np.float64), names


def inject_noise_at_snr(
    signal_window: np.ndarray,
    noise_window: np.ndarray,
    rng: np.random.Generator,
    snr_range_db: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按每台站三分量联合 RMS 定义的目标 SNR 注入噪声。"""
    target_snr = rng.uniform(*snr_range_db, size=signal_window.shape[0])
    scales = np.empty(signal_window.shape[0], dtype=np.float64)
    output = signal_window.copy()
    for station_index in range(signal_window.shape[0]):
        signal_rms = float(np.sqrt(np.mean(signal_window[station_index] ** 2)))
        noise_rms = float(np.sqrt(np.mean(noise_window[station_index] ** 2)))
        if signal_rms <= EPS or noise_rms <= EPS:
            scales[station_index] = 0.0
            continue
        scales[station_index] = signal_rms / (
            noise_rms * 10.0 ** (target_snr[station_index] / 20.0)
        )
        output[station_index] += scales[station_index] * noise_window[station_index]
    return output, target_snr, scales


def as_json(values: Iterable[float] | np.ndarray) -> str:
    return json.dumps(np.asarray(values).tolist(), ensure_ascii=False)


def augment_dataset(args: argparse.Namespace) -> pd.DataFrame:
    validate_ranges(args)
    recipes = parse_recipes(args.recipes)
    event_table = pd.read_csv(args.input_csv)
    required_columns = {"event_id", "depth", "strike", "dip", "rake"}
    missing = required_columns - set(event_table.columns)
    if missing:
        raise KeyError(f"理论元数据CSV缺少列: {sorted(missing)}")
    needs_noise = any("real_noise" in methods for _, methods in recipes)
    output_path = Path(args.output_h5)
    metadata_path = output_path.with_name(output_path.stem + "_metadata.csv")
    records: list[dict[str, object]] = []
    output_index = 0

    noise_context = (
        NoiseLibrary(args.noise_library)
        if needs_noise
        else None
    )

    try:
        with h5py.File(args.input_h5, "r") as source, h5py.File(output_path, "w") as output:
            for _, row in tqdm(event_table.iterrows(), total=len(event_table), desc="增广事件"):
                event_id = int(row["event_id"])
                group_name = f"data_label_{event_id}"
                if group_name not in source:
                    print(f"警告: 输入H5缺少 {group_name}，跳过")
                    continue
                group = source[group_name]
                data_full = np.asarray(group["data"][:], dtype=np.float64)
                if data_full.ndim != 3 or data_full.shape[2] != 3:
                    raise ValueError(f"{group_name}/data 形状异常: {data_full.shape}")
                full_length = data_full.shape[1]
                if full_length != 128:
                    raise ValueError(f"理论完整波形应为128点，实际为{full_length}")
                station_count = data_full.shape[0]
                if station_count < args.num_receivers:
                    print(f"警告: {group_name} 仅{station_count}个台站，跳过")
                    continue
                azimuth_full = group["az"][:]
                takeoff_full = group["takeoff"][:]
                label = group["label"][:]
                label_sdr = group["label_sdr"][:] if "label_sdr" in group else None

                for selection_index in range(args.num_station_selections):
                    selection_rng = make_rng(args.seed, event_id, selection_index, 0)
                    selected = np.sort(
                        selection_rng.choice(station_count, args.num_receivers, replace=False)
                    )
                    selected_data = center_waveforms(data_full[selected])
                    azimuth = azimuth_full[selected]
                    takeoff = takeoff_full[selected]
                    az_encoded, takeoff_encoded = process_az_takeoff(
                        azimuth, takeoff, args.window_len
                    )

                    for recipe_index, (recipe_name, methods) in enumerate(recipes, start=1):
                        rng = make_rng(args.seed, event_id, selection_index, recipe_index)
                        working = selected_data.copy()
                        frequency_scales = np.ones(args.num_receivers)
                        phase_degrees = np.zeros(args.num_receivers)
                        event_shift = 0
                        station_residuals = np.zeros(args.num_receivers, dtype=int)
                        total_shifts = np.zeros(args.num_receivers, dtype=int)
                        station_gains = np.ones(args.num_receivers)
                        component_gains = np.ones((args.num_receivers, 3))
                        snr_db = np.full(args.num_receivers, np.nan)
                        noise_scales = np.zeros(args.num_receivers)
                        noise_names: list[str] = []

                        if "spectral" in methods:
                            working, frequency_scales, phase_degrees = apply_spectral_perturbation(
                                working,
                                rng,
                                (args.frequency_scale_min, args.frequency_scale_max),
                                args.phase_max_deg,
                            )
                        if "shift" in methods:
                            event_shift, station_residuals, total_shifts = draw_shifts(
                                args.num_receivers,
                                rng,
                                args.event_shift_max,
                                args.station_shift_std,
                                args.station_shift_clip,
                            )

                        window = slice_with_station_shifts(
                            working, args.base_start, args.window_len, total_shifts
                        )

                        if "station_gain" in methods:
                            window, station_gains = apply_station_gain(
                                window, rng, args.station_gain_sigma
                            )
                        if "component_gain" in methods:
                            window, component_gains = apply_component_gain(
                                window,
                                rng,
                                (args.component_gain_min, args.component_gain_max),
                            )
                        if "real_noise" in methods:
                            if noise_context is None:
                                raise RuntimeError("real_noise已启用但噪声库未初始化")
                            noise_full, noise_names = noise_context.sample_many(
                                args.num_receivers, rng
                            )
                            # 随机循环移动噪声，使重复抽到同一记录时仍可使用不同的52秒片段。
                            noise_window = np.empty_like(window)
                            for station_index in range(args.num_receivers):
                                noise_start = int(
                                    rng.integers(0, full_length - args.window_len + 1)
                                )
                                noise_window[station_index] = noise_full[
                                    station_index,
                                    noise_start : noise_start + args.window_len,
                                    :,
                                ]
                            window, snr_db, noise_scales = inject_noise_at_snr(
                                window,
                                noise_window,
                                rng,
                                (args.snr_db_min, args.snr_db_max),
                            )

                        normalized = normalize_waveforms(window, args.normalization)
                        final_sample = np.stack(
                            [
                                normalized[:, :, 0],
                                normalized[:, :, 1],
                                normalized[:, :, 2],
                                az_encoded,
                                takeoff_encoded,
                            ],
                            axis=2,
                        ).astype(np.float32)
                        if not np.all(np.isfinite(final_sample)):
                            raise ValueError(
                                f"event={event_id}, recipe={recipe_name} 产生NaN/Inf"
                            )

                        output_group = output.create_group(f"data_label_{output_index}")
                        output_group.create_dataset("data", data=final_sample, compression="gzip")
                        output_group.create_dataset(
                            "label", data=np.asarray(label, dtype=np.float32), compression="gzip"
                        )
                        if label_sdr is not None:
                            output_group.create_dataset(
                                "label_sdr",
                                data=np.asarray(label_sdr, dtype=np.float32),
                                compression="gzip",
                            )
                        output_group.attrs["original_event_id"] = event_id
                        output_group.attrs["augmentation_recipe"] = recipe_name
                        output_group.attrs["normalization"] = args.normalization

                        record = {
                            "new_index": output_index,
                            "original_event_id": event_id,
                            "station_selection_idx": selection_index,
                            "augmentation_recipe": recipe_name,
                            "normalization": args.normalization,
                            "selected_station_indices": as_json(selected),
                            "event_shift_samples": event_shift,
                            "station_residual_shifts": as_json(station_residuals),
                            "total_shifts": as_json(total_shifts),
                            "station_gains": as_json(station_gains),
                            "component_gains_ZEN": as_json(component_gains),
                            "target_snr_db": as_json(snr_db),
                            "noise_scales": as_json(noise_scales),
                            "noise_trace_names": json.dumps(noise_names, ensure_ascii=False),
                            "frequency_scales": as_json(frequency_scales),
                            "phase_degrees": as_json(phase_degrees),
                            "depth": row["depth"],
                            "strike": row["strike"],
                            "dip": row["dip"],
                            "rake": row["rake"],
                        }
                        for optional_column in ("selection_type", "MAG", "entropy"):
                            record[optional_column] = row.get(optional_column, np.nan)
                        records.append(record)
                        output_index += 1
    finally:
        if noise_context is not None:
            noise_context.close()

    metadata = pd.DataFrame(records)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    print(f"完成：{output_index} 个样本")
    print(f"HDF5: {output_path}")
    print(f"元数据: {metadata_path}")
    return metadata


def main() -> None:
    args = parse_args()
    augment_dataset(args)


if __name__ == "__main__":
    main()
