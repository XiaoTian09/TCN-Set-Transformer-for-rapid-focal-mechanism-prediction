#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从预测沙滩球反演 SDR：圆内 ZNCC + 周期三级搜索 + Kagan 统计。

相较 beachball_sdr_parallel.py：
  1. 固定中心，仅在沙滩球圆内计算零均值归一化相关（ZNCC）；
  2. strike/rake 使用周期搜索，dip 使用闭区间 [0, 90]；
  3. 默认执行 15° -> 3° -> 1° 三级搜索，最后一级使用更高分辨率；
  4. 使用轻量 PIL 光栅器复现项目 beachball_generator 的绘图约定；
  5. 粗模板一次生成并通过矩阵乘法批量匹配，局部搜索可多进程；
  6. 读取 MAT 中 strike/dip/rake 真值，调用 kagan.py 统计 Kagan 角。

典型用法：
    python beachball_sdr_optimized.py \
      --input-mat instance_tcn_settransformer_pytorch_pred.mat \
      --output-dir beachball_sdr_results --workers 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
KAGAN_DIR = SCRIPT_DIR / "model_sdr_sincos"
if str(KAGAN_DIR) not in sys.path:
    sys.path.insert(0, str(KAGAN_DIR))
from kagan import get_kagan_angle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="优化的沙滩球到 SDR 反演")
    parser.add_argument(
        "--input-mat",
        default=str(
            SCRIPT_DIR / "outputs" / "predictions.mat"
        ),
    )
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "outputs" / "sdr"))
    parser.add_argument("--coarse-step", type=float, default=15.0)
    parser.add_argument("--refine-step", type=float, default=3.0)
    parser.add_argument("--final-step", type=float, default=1.0)
    parser.add_argument("--coarse-size", type=int, default=32)
    parser.add_argument("--final-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=3, help="粗搜索保留的候选中心数")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--template-batch-size", type=int, default=2048)
    parser.add_argument("--radius-fraction", type=float, default=0.345)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("coarse_step", "refine_step", "final_step"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须为正数")
    if args.refine_step > args.coarse_step or args.final_step > args.refine_step:
        raise ValueError("步长应满足 coarse-step >= refine-step >= final-step")
    if args.coarse_size < 16 or args.final_size < args.coarse_size:
        raise ValueError("尺寸应满足 final-size >= coarse-size >= 16")
    if args.top_k <= 0 or args.workers < 0 or args.image_batch_size <= 0:
        raise ValueError("top-k/image-batch-size 必须为正，workers 不能为负")
    if not 0.2 <= args.radius_fraction <= 0.5:
        raise ValueError("radius-fraction 应在 [0.2, 0.5] 内")
    if args.supersample <= 0:
        raise ValueError("supersample 必须为正数")


def _as_images(raw: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(raw)
    if values.ndim == 4:
        if values.shape[1] == 1:
            values = values[:, 0]
        elif values.shape[-1] == 1:
            values = values[..., 0]
        elif values.shape[-1] == 3:
            values = np.tensordot(values, np.asarray([0.299, 0.587, 0.114]), axes=([-1], [0]))
        else:
            raise ValueError(f"{name} 的四维形状不受支持: {values.shape}")
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError(f"{name} 应为 (N,H,W) 或带单通道，实际为 {values.shape}")
    values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} 包含 NaN/Inf")
    return values


def _flat_angle(data: dict, key: str, count: int) -> np.ndarray | None:
    if key not in data:
        return None
    value = np.asarray(data[key], dtype=np.float64).reshape(-1)
    if len(value) != count:
        raise ValueError(f"{key} 长度为 {len(value)}，但样本数为 {count}")
    return value


def load_input(path: str | Path, limit: int | None = None) -> dict[str, np.ndarray | None]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MAT 文件不存在: {path}")
    data = sio.loadmat(path)
    if "y_pred" not in data:
        raise KeyError("MAT 文件中没有 y_pred")
    predicted = _as_images(data["y_pred"], "y_pred")
    count = len(predicted)
    if count == 0:
        raise ValueError("y_pred contains no events")
    truth_images = (_as_images(data["y_test"], "y_test")
                    if "y_test" in data and np.asarray(data["y_test"]).ndim >= 3 else None)
    strike = _flat_angle(data, "strike", count)
    dip = _flat_angle(data, "dip", count)
    rake = _flat_angle(data, "rake", count)
    if strike is None and "y_test" in data:
        candidate = np.asarray(data["y_test"])
        if candidate.ndim == 2 and candidate.shape == (count, 3):
            strike, dip, rake = candidate[:, 0], candidate[:, 1], candidate[:, 2]
            truth_images = None
    target_sdr = None
    if strike is not None and dip is not None and rake is not None:
        target_sdr = np.stack([strike, dip, rake], axis=1).astype(np.float32)
        if not np.isfinite(target_sdr).all():
            raise ValueError("Reference SDR contains NaN/Inf; omit reference arrays for unlabeled input")
    if truth_images is not None and len(truth_images) != count:
        raise ValueError("y_test and y_pred sample counts differ")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须为正数")
        predicted = predicted[:limit]
        truth_images = truth_images[:limit] if truth_images is not None else None
        target_sdr = target_sdr[:limit] if target_sdr is not None else None
    sample_ids = np.asarray(data.get("sample_ids", np.arange(count))).reshape(-1)
    if len(sample_ids) != count:
        raise ValueError("sample_ids and y_pred sample counts differ")
    sample_ids = sample_ids[:len(predicted)]
    print(f"输入: {path}\ny_pred: {predicted.shape}; SDR 真值: {target_sdr is not None}")
    return {"y_pred": predicted, "y_test": truth_images,
            "target_sdr": target_sdr, "sample_ids": sample_ids}


def resize_images(images: np.ndarray, size: int) -> np.ndarray:
    if images.shape[1:] == (size, size):
        return images.astype(np.float32, copy=False)
    factors = (1.0, size / images.shape[1], size / images.shape[2])
    return zoom(images, factors, order=1, prefilter=False).astype(np.float32)


def circular_mask(size: int, radius_fraction: float) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    center = (size - 1.0) / 2.0
    radius = radius_fraction * size
    return (x - center) ** 2 + (y - center) ** 2 <= radius**2


def _strike_dip(n: np.ndarray, e: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    negative = u < 0
    n, e, u = n.copy(), e.copy(), u.copy()
    n[negative], e[negative], u[negative] = -n[negative], -e[negative], -u[negative]
    strike = (np.degrees(np.arctan2(e, n)) - 90.0) % 360.0
    dip = np.degrees(np.arctan2(np.hypot(n, e), u))
    return strike, dip


def auxiliary_plane(strike: float, dip: float, rake: float) -> tuple[float, float, float]:
    s = np.asarray([strike], dtype=np.float64)
    d = np.asarray([dip], dtype=np.float64)
    r = np.asarray([rake], dtype=np.float64)
    z, z2, z3 = np.radians(s + 90.0), np.radians(d), np.radians(r)
    sl1 = -np.cos(z3) * np.cos(z) - np.sin(z3) * np.sin(z) * np.cos(z2)
    sl2 = np.cos(z3) * np.sin(z) - np.sin(z3) * np.cos(z) * np.cos(z2)
    sl3 = np.sin(z3) * np.sin(z2)
    strike2, dip2 = _strike_dip(sl2, sl1, sl3)
    n1, n2 = np.sin(z) * np.sin(z2), np.cos(z) * np.sin(z2)
    h1, h2 = -sl2, sl1
    cosine = np.clip((h1 * n1 + h2 * n2) / np.maximum(np.hypot(h1, h2), 1e-12), -1, 1)
    rake2 = np.degrees(np.arccos(cosine)) * np.where(sl3 > 0, 1.0, -1.0)
    return float(strike2[0]), float(dip2[0]), float(rake2[0])


@lru_cache(maxsize=131072)
def render_beachball(strike: float, dip: float, rake: float, size: int,
                     supersample: int = 2) -> np.ndarray:
    """不创建 Matplotlib figure，快速复现 beachball_generator 的多边形图。"""
    strike = float(strike % 360.0)
    dip = float(np.clip(dip, 0.0, 90.0))
    rake = float((rake + 180.0) % 360.0 - 180.0)
    mechanism = rake < 0.0
    adjusted_rake = rake + 180.0 if mechanism else rake
    strike2, dip2, _ = auxiliary_plane(strike, dip, adjusted_rake)
    dip1 = min(dip, 89.9999)
    dip2 = min(dip2, 89.9999)
    phi = np.arange(0.0, np.pi + 0.01, 0.01)

    def plane_curve(s: float, d: float):
        minor = 90.0 - d
        radius = np.sqrt(minor**2 / (np.sin(phi)**2 + np.cos(phi)**2 * minor**2 / 90.0**2))
        angle = phi + np.radians(s)
        return radius * np.cos(angle), radius * np.sin(angle)

    x1, y1 = plane_curve(strike, dip1)
    x2, y2 = plane_curve(strike2, dip2)
    increment = 1.0
    if not mechanism:
        if strike - 180.0 > strike2:
            increment = -1.0
        th1 = np.arange(strike - 180.0, strike2 + increment, increment)
        th2 = np.arange(strike2 + 180.0, strike - increment, -increment)
    else:
        if strike2 - 180.0 > strike - 180.0:
            increment = -1.0
        th1 = np.arange(strike - 180.0, strike2 - 180.0 - increment, -increment)
        x2, y2 = x2[::-1], y2[::-1]
        th2 = np.arange(strike2, strike + increment, increment)
    bx1, by1 = 90.0 * np.cos(np.radians(th1)), 90.0 * np.sin(np.radians(th1))
    bx2, by2 = 90.0 * np.cos(np.radians(th2)), 90.0 * np.sin(np.radians(th2))
    physical_x = np.concatenate([x1, bx1, x2, bx2])
    physical_y = np.concatenate([y1, by1, y2, by2])

    scale = supersample
    canvas_size = size * scale
    # 原生成器坐标范围约为 [-15,15]，沙滩球半径为 10，即半径约占图宽 1/3。
    center = (canvas_size - 1) / 2.0
    radius = canvas_size / 3.0
    points = np.column_stack([
        center + physical_y / 90.0 * radius,
        center - physical_x / 90.0 * radius,
    ])
    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)
    draw.polygon([tuple(point) for point in points], fill=76)
    width = max(1, round(0.5 / 30.0 * canvas_size))
    draw.ellipse([center - radius, center - radius, center + radius, center + radius],
                 outline=0, width=width)
    if scale > 1:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def normalize_masked(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = np.asarray(images[:, mask], dtype=np.float32)
    pixels -= pixels.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(pixels, axis=1, keepdims=True)
    return np.divide(pixels, norms, out=np.zeros_like(pixels), where=norms > 1e-8)


def angle_grid(step: float) -> np.ndarray:
    strikes = np.arange(0.0, 360.0, step)
    dips = np.unique(np.append(np.arange(0.0, 90.0 + step * 0.25, step), 90.0))
    rakes = np.arange(-180.0, 180.0, step)
    return np.asarray(np.meshgrid(strikes, dips, rakes, indexing="ij")).reshape(3, -1).T


def template_features(angles: np.ndarray, size: int, mask: np.ndarray,
                      supersample: int, batch_size: int) -> np.ndarray:
    output = np.empty((len(angles), int(mask.sum())), dtype=np.float32)
    for start in range(0, len(angles), batch_size):
        end = min(start + batch_size, len(angles))
        images = np.stack([
            render_beachball(float(s), float(d), float(r), size, supersample)
            for s, d, r in angles[start:end]
        ])
        output[start:end] = normalize_masked(images, mask)
    return output


def coarse_search(image_features: np.ndarray, template_matrix: np.ndarray,
                  top_k: int, image_batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty((len(image_features), top_k), dtype=np.int64)
    scores = np.empty((len(image_features), top_k), dtype=np.float32)
    for start in tqdm(range(0, len(image_features), image_batch_size), desc="粗搜索"):
        end = min(start + image_batch_size, len(image_features))
        similarities = image_features[start:end] @ template_matrix.T
        selected = np.argpartition(similarities, -top_k, axis=1)[:, -top_k:]
        selected_scores = np.take_along_axis(similarities, selected, axis=1)
        order = np.argsort(selected_scores, axis=1)[:, ::-1]
        indices[start:end] = np.take_along_axis(selected, order, axis=1)
        scores[start:end] = np.take_along_axis(selected_scores, order, axis=1)
    return indices, scores


def periodic_candidates(centers: np.ndarray, radius: float, step: float) -> np.ndarray:
    offsets = np.arange(-radius, radius + step * 0.25, step)
    candidates = []
    for center in np.atleast_2d(centers):
        s = np.unique(np.round((center[0] + offsets) % 360.0, 7))
        d = np.unique(np.round(np.clip(center[1] + offsets, 0.0, 90.0), 7))
        r = np.unique(np.round((center[2] + offsets + 180.0) % 360.0 - 180.0, 7))
        candidates.append(np.asarray(np.meshgrid(s, d, r, indexing="ij")).reshape(3, -1).T)
    return np.unique(np.concatenate(candidates), axis=0)


def best_local(image: np.ndarray, centers: np.ndarray, radius: float, step: float,
               size: int, radius_fraction: float, supersample: int,
               template_batch_size: int) -> tuple[np.ndarray, float]:
    resized = resize_images(image[None], size)
    mask = circular_mask(size, radius_fraction)
    image_feature = normalize_masked(resized, mask)[0]
    candidates = periodic_candidates(centers, radius, step)
    best_score, best_angle = -np.inf, candidates[0]
    for start in range(0, len(candidates), template_batch_size):
        batch_angles = candidates[start:start + template_batch_size]
        features = template_features(batch_angles, size, mask, supersample, template_batch_size)
        scores = features @ image_feature
        index = int(np.argmax(scores))
        if scores[index] > best_score:
            best_score, best_angle = float(scores[index]), batch_angles[index]
    return best_angle.astype(np.float32), best_score


def local_worker(task):
    image, coarse_centers, args_dict = task
    refined, _ = best_local(
        image, coarse_centers, args_dict["coarse_step"], args_dict["refine_step"],
        args_dict["coarse_size"], args_dict["radius_fraction"], args_dict["supersample"],
        args_dict["template_batch_size"],
    )
    final, score = best_local(
        image, refined[None], args_dict["refine_step"], args_dict["final_step"],
        args_dict["final_size"], args_dict["radius_fraction"], args_dict["supersample"],
        args_dict["template_batch_size"],
    )
    return final, score


def kagan_values(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray([
        get_kagan_angle(float(p[0]), float(p[1]), float(p[2]),
                        float(t[0]), float(t[1]), float(t[2]))
        for p, t in zip(predicted, target)
    ], dtype=np.float32)


def summary_dict(kagan: np.ndarray | None, scores: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {
        "samples": int(len(scores)), "mean_zncc": float(np.mean(scores)),
        "median_zncc": float(np.median(scores)),
    }
    if kagan is not None:
        result.update({"mean_kagan_deg": float(np.mean(kagan)),
                       "median_kagan_deg": float(np.median(kagan)), "thresholds": {}})
        thresholds = result["thresholds"]
        assert isinstance(thresholds, dict)
        for threshold in (10, 20, 30, 40, 50):
            count = int(np.count_nonzero(kagan < threshold))
            thresholds[f"lt_{threshold}_deg"] = {
                "count": count, "percent": 100.0 * count / len(kagan)
            }
    return result


def save_results(output_dir: Path, sample_ids: np.ndarray, predicted: np.ndarray,
                 scores: np.ndarray, target: np.ndarray | None,
                 kagan: np.ndarray | None, summary: dict[str, object], args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mat = {"sample_ids": sample_ids, "strike": predicted[:, 0],
           "dip": predicted[:, 1], "rake": predicted[:, 2], "zncc": scores}
    if target is not None:
        mat.update({"strike_true": target[:, 0], "dip_true": target[:, 1],
                    "rake_true": target[:, 2], "kagan_angle_deg": kagan})
    sio.savemat(output_dir / "sdr_results.mat", mat, do_compression=True)
    with open(output_dir / "sdr_results.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        header = ["sample_id", "strike", "dip", "rake", "zncc"]
        if target is not None:
            header += ["strike_true", "dip_true", "rake_true", "kagan_angle_deg"]
        writer.writerow(header)
        for index in range(len(predicted)):
            row = [sample_ids[index], *predicted[index].tolist(), float(scores[index])]
            if target is not None:
                row += [*target[index].tolist(), float(kagan[index])]
            writer.writerow(row)
    summary["configuration"] = vars(args)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    loaded = load_input(args.input_mat, args.limit)
    images = loaded["y_pred"]
    assert isinstance(images, np.ndarray)
    coarse_images = resize_images(images, args.coarse_size)
    coarse_mask = circular_mask(args.coarse_size, args.radius_fraction)
    image_features = normalize_masked(coarse_images, coarse_mask)
    coarse_angles = angle_grid(args.coarse_step)
    print(f"生成 {len(coarse_angles)} 个粗搜索模板...")
    coarse_templates = template_features(
        coarse_angles, args.coarse_size, coarse_mask, args.supersample,
        args.template_batch_size,
    )
    coarse_indices, _ = coarse_search(
        image_features, coarse_templates, min(args.top_k, len(coarse_angles)),
        args.image_batch_size,
    )
    centers = coarse_angles[coarse_indices]
    worker_args = {
        key: getattr(args, key) for key in (
            "coarse_step", "refine_step", "final_step", "coarse_size", "final_size",
            "radius_fraction", "supersample", "template_batch_size",
        )
    }
    tasks = ((images[index], centers[index], worker_args) for index in range(len(images)))
    if args.workers == 0:
        local_results = [local_worker(task) for task in tqdm(tasks, total=len(images), desc="局部搜索")]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            local_results = list(tqdm(
                executor.map(local_worker, tasks, chunksize=1), total=len(images), desc="局部搜索"
            ))
    predicted = np.stack([item[0] for item in local_results]).astype(np.float32)
    scores = np.asarray([item[1] for item in local_results], dtype=np.float32)
    target = loaded["target_sdr"]
    kagan = kagan_values(predicted, target) if isinstance(target, np.ndarray) else None
    summary = summary_dict(kagan, scores)
    summary["elapsed_seconds"] = time.perf_counter() - started
    save_results(Path(args.output_dir), loaded["sample_ids"], predicted,
                 scores, target, kagan, summary, args)
    print("\n完成")
    print(f"平均/中位 ZNCC: {summary['mean_zncc']:.4f} / {summary['median_zncc']:.4f}")
    if kagan is not None:
        print(f"平均/中位 Kagan: {summary['mean_kagan_deg']:.3f}° / {summary['median_kagan_deg']:.3f}°")
        for threshold, item in summary["thresholds"].items():
            print(f"{threshold}: {item['count']} ({item['percent']:.2f}%)")
    print(f"结果目录: {args.output_dir}")


if __name__ == "__main__":
    main()
