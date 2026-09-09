#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 TCN + Set Transformer 生成可直接训练的 Sim2Real HDF5。

本文件不修改现有增广脚本，而是复用其中经过验证的增广函数。输出的每个
``data_label_<index>`` group 包含：

    waveform     (20, 52, 3)  float32，通道顺序 Z/E/N
    geometry     (20, 4)      float32，sin/cos 方位角与离源角
    station_mask (20,)        float32，当前均为1，为缺台扩展预留
    label        (128, 128)    float32，沙滩球标签

典型用法（训练集与验证集应使用事件级独立的 CSV/H5）：

    python traindata_augment_sim2real_settransformer.py \
        --input-h5 Traindata_filter_fulllength_3ch.h5 \
        --input-csv train_events.csv \
        --output-h5 train_tcn_settransformer.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import traindata_augment_sim2real as aug


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NOISE_LIBRARY = SCRIPT_DIR / "data" / "noise_library.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 TCN + Set Transformer 可直接读取的 Sim2Real HDF5"
    )
    parser.add_argument("--input-h5", default="data/Traindata_filter_fulllength_3ch.h5")
    parser.add_argument(
        "--input-csv", default="data/Traindata_filter_fulllength_3ch_metadata.csv"
    )
    parser.add_argument("--output-h5", default="Traindata_tcn_settransformer.h5")
    parser.add_argument("--noise-library", default=str(DEFAULT_NOISE_LIBRARY))
    parser.add_argument("--num-station-selections", type=int, default=5)
    parser.add_argument("--num-receivers", type=int, default=20)
    parser.add_argument("--base-start", type=int, default=45)
    parser.add_argument("--window-len", type=int, default=52)
    parser.add_argument(
        "--recipes",
        default="base;shift+station_gain+component_gain;real_noise",
    )
    parser.add_argument(
        "--normalization",
        choices=sorted(aug.NORMALIZATION_MODES),
        default="legacy",
        help="推荐 station_joint，以保留同一台站三分量的相对振幅",
    )
    parser.add_argument("--event-shift-max", type=int, default=2)
    parser.add_argument("--station-shift-std", type=float, default=1.0)
    parser.add_argument("--station-shift-clip", type=int, default=3)
    parser.add_argument("--station-gain-sigma", type=float, default=0.26)
    parser.add_argument("--component-gain-min", type=float, default=0.9)
    parser.add_argument("--component-gain-max", type=float, default=1.1)
    parser.add_argument("--snr-db-min", type=float, default=5.0)
    parser.add_argument("--snr-db-max", type=float, default=25.0)
    parser.add_argument("--frequency-scale-min", type=float, default=0.98)
    parser.add_argument("--frequency-scale-max", type=float, default=1.02)
    parser.add_argument("--phase-max-deg", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def geometry_sincos(azimuth_deg: np.ndarray, takeoff_deg: np.ndarray) -> np.ndarray:
    """生成 [sin(az), cos(az), sin(takeoff), cos(takeoff)]。"""
    azimuth = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64) % 360.0)
    takeoff = np.deg2rad(np.asarray(takeoff_deg, dtype=np.float64))
    geometry = np.stack(
        [
            np.sin(azimuth),
            np.cos(azimuth),
            np.sin(takeoff),
            np.cos(takeoff),
        ],
        axis=-1,
    )
    return geometry.astype(np.float32)


def validate_args(args: argparse.Namespace) -> None:
    aug.validate_ranges(args)
    if args.num_receivers <= 0:
        raise ValueError("num-receivers 必须为正数")
    if args.window_len <= 0:
        raise ValueError("window-len 必须为正数")
    max_shift = args.event_shift_max + args.station_shift_clip
    if args.base_start - max_shift < 0:
        raise ValueError("base-start 太小，负向平移后窗口会越界")
    if args.base_start + args.window_len + max_shift > 128:
        raise ValueError("base-start/window-len 太大，正向平移后窗口会越界")


def generate_dataset(args: argparse.Namespace) -> pd.DataFrame:
    validate_args(args)
    recipes = aug.parse_recipes(args.recipes)
    event_table = pd.read_csv(args.input_csv)
    required = {"event_id", "depth", "strike", "dip", "rake"}
    missing = required - set(event_table.columns)
    if missing:
        raise KeyError(f"输入CSV缺少字段: {sorted(missing)}")

    needs_noise = any("real_noise" in methods for _, methods in recipes)
    noise_library = aug.NoiseLibrary(args.noise_library) if needs_noise else None
    output_path = Path(args.output_h5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_name(output_path.stem + "_metadata.csv")

    records: list[dict[str, object]] = []
    output_index = 0
    try:
        with h5py.File(args.input_h5, "r") as source, h5py.File(output_path, "w") as output:
            output.attrs["schema"] = "tcn_settransformer_v1"
            output.attrs["component_order"] = "ZEN"
            output.attrs["geometry_order"] = "sin_az,cos_az,sin_takeoff,cos_takeoff"
            output.attrs["sampling_rate_hz"] = 1.0
            output.attrs["window_length"] = args.window_len
            output.attrs["num_receivers"] = args.num_receivers
            output.attrs["normalization"] = args.normalization

            iterator = event_table.iterrows()
            for _, row in tqdm(iterator, total=len(event_table), desc="生成模型输入"):
                event_id = int(row["event_id"])
                group_name = f"data_label_{event_id}"
                if group_name not in source:
                    print(f"警告: 输入H5缺少 {group_name}，跳过")
                    continue

                group = source[group_name]
                data_full = np.asarray(group["data"][:], dtype=np.float64)
                if data_full.ndim != 3 or data_full.shape[1:] != (128, 3):
                    raise ValueError(f"{group_name}/data 形状应为 (N,128,3)，实际为 {data_full.shape}")
                if data_full.shape[0] < args.num_receivers:
                    print(f"警告: {group_name} 台站不足，跳过")
                    continue

                azimuth_full = np.asarray(group["az"][:], dtype=np.float64)
                takeoff_full = np.asarray(group["takeoff"][:], dtype=np.float64)
                label = np.asarray(group["label"][:], dtype=np.float32)
                if label.shape != (128, 128):
                    raise ValueError(f"{group_name}/label 形状应为 (128,128)，实际为 {label.shape}")

                for selection_index in range(args.num_station_selections):
                    selection_rng = aug.make_rng(args.seed, event_id, selection_index, 0)
                    selected = np.sort(
                        selection_rng.choice(
                            data_full.shape[0], args.num_receivers, replace=False
                        )
                    )
                    selected_data = aug.center_waveforms(data_full[selected])
                    azimuth = azimuth_full[selected]
                    takeoff = takeoff_full[selected]
                    geometry = geometry_sincos(azimuth, takeoff)

                    for recipe_index, (recipe_name, methods) in enumerate(recipes, start=1):
                        rng = aug.make_rng(
                            args.seed, event_id, selection_index, recipe_index
                        )
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
                            working, frequency_scales, phase_degrees = (
                                aug.apply_spectral_perturbation(
                                    working,
                                    rng,
                                    (
                                        args.frequency_scale_min,
                                        args.frequency_scale_max,
                                    ),
                                    args.phase_max_deg,
                                )
                            )
                        if "shift" in methods:
                            event_shift, station_residuals, total_shifts = aug.draw_shifts(
                                args.num_receivers,
                                rng,
                                args.event_shift_max,
                                args.station_shift_std,
                                args.station_shift_clip,
                            )

                        window = aug.slice_with_station_shifts(
                            working, args.base_start, args.window_len, total_shifts
                        )
                        if "station_gain" in methods:
                            window, station_gains = aug.apply_station_gain(
                                window, rng, args.station_gain_sigma
                            )
                        if "component_gain" in methods:
                            window, component_gains = aug.apply_component_gain(
                                window,
                                rng,
                                (
                                    args.component_gain_min,
                                    args.component_gain_max,
                                ),
                            )
                        if "real_noise" in methods:
                            if noise_library is None:
                                raise RuntimeError("real_noise 已启用，但噪声库未初始化")
                            noise_full, noise_names = noise_library.sample_many(
                                args.num_receivers, rng
                            )
                            noise_window = np.empty_like(window)
                            for station_index in range(args.num_receivers):
                                noise_start = int(
                                    rng.integers(0, 128 - args.window_len + 1)
                                )
                                noise_window[station_index] = noise_full[
                                    station_index,
                                    noise_start : noise_start + args.window_len,
                                    :,
                                ]
                            window, snr_db, noise_scales = aug.inject_noise_at_snr(
                                window,
                                noise_window,
                                rng,
                                (args.snr_db_min, args.snr_db_max),
                            )

                        waveform = aug.normalize_waveforms(window, args.normalization)
                        if not np.all(np.isfinite(waveform)):
                            raise ValueError(
                                f"event={event_id}, recipe={recipe_name} 产生NaN/Inf"
                            )

                        output_group = output.create_group(
                            f"data_label_{output_index}"
                        )
                        output_group.create_dataset(
                            "waveform", data=waveform, compression="gzip"
                        )
                        output_group.create_dataset(
                            "geometry", data=geometry, compression="gzip"
                        )
                        output_group.create_dataset(
                            "station_mask",
                            data=np.ones(args.num_receivers, dtype=np.float32),
                            compression="gzip",
                        )
                        output_group.create_dataset(
                            "label", data=label, compression="gzip"
                        )
                        output_group.attrs["original_event_id"] = event_id
                        output_group.attrs["augmentation_recipe"] = recipe_name
                        output_group.attrs["station_selection_idx"] = selection_index

                        records.append(
                            {
                                "new_index": output_index,
                                "original_event_id": event_id,
                                "station_selection_idx": selection_index,
                                "augmentation_recipe": recipe_name,
                                "selected_station_indices": aug.as_json(selected),
                                "azimuth_deg": aug.as_json(azimuth),
                                "takeoff_deg": aug.as_json(takeoff),
                                "event_shift_samples": event_shift,
                                "station_residual_shifts": aug.as_json(
                                    station_residuals
                                ),
                                "total_shifts": aug.as_json(total_shifts),
                                "station_gains": aug.as_json(station_gains),
                                "component_gains_ZEN": aug.as_json(component_gains),
                                "target_snr_db": aug.as_json(snr_db),
                                "noise_scales": aug.as_json(noise_scales),
                                "noise_trace_names": json.dumps(
                                    noise_names, ensure_ascii=False
                                ),
                                "frequency_scales": aug.as_json(frequency_scales),
                                "phase_degrees": aug.as_json(phase_degrees),
                                "depth": row["depth"],
                                "strike": row["strike"],
                                "dip": row["dip"],
                                "rake": row["rake"],
                            }
                        )
                        output_index += 1
    finally:
        if noise_library is not None:
            noise_library.close()

    metadata = pd.DataFrame(records)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    print(f"完成：{output_index} 个样本")
    print(f"HDF5：{output_path}")
    print(f"元数据：{metadata_path}")
    return metadata


def main() -> None:
    generate_dataset(parse_args())


if __name__ == "__main__":
    main()
