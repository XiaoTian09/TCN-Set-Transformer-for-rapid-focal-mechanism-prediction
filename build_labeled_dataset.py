#!/usr/bin/env python3
"""Build leak-free INSTANCE labeled train/validation/test station combinations."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
import preprocessing as pre
import beachball_generator as beach

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=HERE / "configs" / "build_labeled.json")
    p.add_argument("--limit-events", type=int)
    return p.parse_args()


def parse_sdr(value) -> tuple[float, float, float]:
    if pd.isna(value):
        raise ValueError("机制为空")
    text = str(value).strip()
    match = re.search(
        r"strike\s*=\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"dip\s*=\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"rake\s*=\s*([-+]?\d+(?:\.\d+)?)",
        text,
    )
    if match:
        parts = match.groups()
    else:
        first_plane = text.replace("[", "").replace("]", "").split(";")[0]
        parts = [x.strip() for x in first_plane.replace("/", ",").split(",") if x.strip()]
    if len(parts) < 3:
        raise ValueError(f"无法解析机制: {value}")
    s, d, r = map(float, parts[:3])
    if not (0 <= d <= 90):
        raise ValueError(f"dip越界: {d}")
    return s % 360, d, ((r + 180) % 360) - 180


def station_key(row) -> str:
    return ".".join("" if pd.isna(getattr(row, c)) else str(getattr(row, c)).strip()
                    for c in ("station_network_code", "station_code", "station_location_code"))


def circular_gap(az):
    x = np.sort(np.asarray(az, float) % 360)
    return float(np.diff(np.r_[x, x[0] + 360]).max())


def choose_combinations(pool, count, cfg, rng):
    if len(pool) == 20:
        return [np.arange(20)]
    accepted = []
    names = pool.trace_name.astype(str).to_numpy()
    az = pd.to_numeric(pool.path_azimuth_deg).to_numpy() % 360
    for _ in range(int(cfg["max_attempts"])):
        idx = np.sort(rng.choice(len(pool), 20, replace=False))
        chosen = set(names[idx])
        if circular_gap(az[idx]) > float(cfg["max_azimuthal_gap_deg"]):
            continue
        if any(len(chosen & set(names[old])) > int(cfg["max_combination_overlap"]) for old in accepted):
            continue
        accepted.append(idx)
        if len(accepted) == count:
            break
    return accepted


def event_split(event_ids, fractions, seed):
    ids = np.array(sorted(map(str, event_ids)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = int(round(len(ids) * fractions[0]))
    n_val = int(round(len(ids) * fractions[1]))
    n_train = min(n_train, len(ids) - 2) if len(ids) >= 3 else len(ids)
    n_val = min(n_val, len(ids) - n_train - 1) if len(ids) - n_train >= 2 else 0
    return {sid: split for split, values in (
        ("train", ids[:n_train]), ("validation", ids[n_train:n_train+n_val]),
        ("test", ids[n_train+n_val:])) for sid in values}


def main():
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("metadata_csv", "waveforms_h5", "output_data_dir"):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str(value if value.is_absolute() else HERE / value)
    outdir = Path(cfg["output_data_dir"])
    if outdir.exists():
        raise FileExistsError(f"Output already exists: {outdir}; select a new output_data_dir")
    outdir.mkdir(parents=True)
    metadata = pd.read_csv(cfg["metadata_csv"], low_memory=False, dtype={"source_id": str, "trace_name": str, "station_network_code": str, "station_code": str, "station_location_code": str})
    required = {"source_id", "trace_name", "source_depth_km", "path_ep_distance_km",
                "path_azimuth_deg", "takeoff_angle_deg", "trace_P_arrival_sample",
                "source_mechanism_strike_dip_rake", "station_network_code",
                "station_code", "station_location_code"}
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"元数据缺少字段: {sorted(missing)}")
    metadata = metadata.drop_duplicates("trace_name").copy()
    metadata = metadata[
        metadata.source_mechanism_strike_dip_rake.notna()
        & (pd.to_numeric(metadata.source_depth_km, errors="coerce") < cfg["max_depth_km"])
        & (pd.to_numeric(metadata.path_ep_distance_km, errors="coerce") < cfg["max_epicentral_distance_km"])
    ]
    ids = sorted(metadata.source_id.astype(str).unique())
    if args.limit_events:
        ids = ids[:args.limit_events]
        metadata = metadata[metadata.source_id.astype(str).isin(ids)]
    split_map = event_split(ids, cfg["split_fractions"], int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    handles = {s: h5py.File(outdir / f"{s}.h5.partial", "w") for s in ("train", "validation", "test")}
    counts = {s: 0 for s in handles}
    manifest, qc = [], []
    try:
        with h5py.File(cfg["waveforms_h5"], "r") as raw:
            raw_data = raw["data"] if "data" in raw else raw
            for split, h in handles.items():
                h.attrs.update({"schema": "instance_labeled_transfer_v1", "split": split,
                                "seed": cfg["seed"], "event_level_split": True,
                                "component_order": "ZEN", "geometry_order": "sin_az,cos_az,sin_takeoff,cos_takeoff"})
            for sid, rows in tqdm(metadata.groupby(metadata.source_id.astype(str)), desc="构建有标签组合"):
                valid, waves, seen = [], {}, set()
                snr = pd.to_numeric(rows.get("trace_Z_snr_db", pd.Series(index=rows.index, dtype=float)), errors="coerce")
                for idx in rows.assign(_snr=snr).sort_values("_snr", ascending=False).index:
                    row = rows.loc[idx]
                    key = station_key(row)
                    if key in seen:
                        continue
                    try:
                        name = str(row.trace_name)
                        if name not in raw_data:
                            raise KeyError("原始波形不存在")
                        takeoff = float(row.takeoff_angle_deg)
                        if not np.isfinite(takeoff):
                            raise ValueError("takeoff无效")
                        waves[name] = pre.preprocess_trace(raw_data[name][()], float(row.trace_P_arrival_sample))
                        valid.append(row); seen.add(key)
                    except Exception as exc:
                        qc.append({"source_id": sid, "trace_name": str(row.trace_name), "status": "station_rejected", "reason": str(exc)})
                if len(valid) < 20:
                    qc.append({"source_id": sid, "status": "event_rejected", "reason": f"有效独立台站{len(valid)}<20"})
                    continue
                pool = pd.DataFrame(valid).reset_index(drop=True)
                combos = choose_combinations(pool, int(cfg["combinations_per_event"]), cfg, rng)
                if not combos:
                    qc.append({"source_id": sid, "status": "event_rejected", "reason": "无法生成满足方位约束的组合"})
                    continue
                strike, dip, rake = parse_sdr(rows.iloc[0].source_mechanism_strike_dip_rake)
                label = beach.generate_beachball(strike, dip, rake).astype(np.float32)
                split = split_map[sid]
                for repeat, idx in enumerate(combos):
                    selected = pool.iloc[idx].sort_values("path_azimuth_deg")
                    az = pd.to_numeric(selected.path_azimuth_deg).to_numpy(np.float32) % 360
                    takeoff = pd.to_numeric(selected.takeoff_angle_deg).to_numpy(np.float32)
                    normalized = pre.normalize_legacy(np.stack([waves[str(n)] for n in selected.trace_name]))
                    waveform = normalized[:, [2, 0, 1], :].transpose(0, 2, 1).astype(np.float32)
                    ar, tr = np.deg2rad(az), np.deg2rad(takeoff)
                    geometry = np.stack([np.sin(ar), np.cos(ar), np.sin(tr), np.cos(tr)], 1).astype(np.float32)
                    n = counts[split]; group = handles[split].create_group(f"data_label_{n}")
                    group.create_dataset("waveform", data=waveform, compression="gzip")
                    group.create_dataset("geometry", data=geometry, compression="gzip")
                    group.create_dataset("station_mask", data=np.ones(20, np.float32))
                    group.create_dataset("label", data=label, compression="gzip")
                    names = selected.trace_name.astype(str).tolist()
                    group.attrs.update({"event_id": sid, "repeat": repeat, "strike": strike,
                                        "dip": dip, "rake": rake, "pool_station_count": len(pool),
                                        "trace_names": json.dumps(names), "azimuthal_gap_deg": circular_gap(az)})
                    manifest.append({"split": split, "sample_index": n, "source_id": sid,
                                     "repeat": repeat, "pool_station_count": len(pool),
                                     "azimuthal_gap_deg": circular_gap(az), "strike": strike, "dip": dip, "rake": rake})
                    counts[split] += 1
        for h in handles.values(): h.close()
        for split in handles:
            (outdir / f"{split}.h5.partial").replace(outdir / f"{split}.h5")
    except Exception:
        for h in handles.values():
            try: h.close()
            except Exception: pass
        raise
    pd.DataFrame(manifest).to_csv(outdir / "manifest.csv", index=False)
    pd.DataFrame(qc).to_csv(outdir / "qc.csv", index=False)
    split_df = pd.DataFrame([{"source_id": k, "split": v} for k, v in split_map.items()])
    split_df.to_csv(outdir / "event_splits.csv", index=False)
    summary = {"seed": cfg["seed"], "candidate_events": len(ids), "samples": counts,
               "accepted_events": {s: int(pd.DataFrame(manifest).query("split == @s").source_id.nunique()) if manifest else 0 for s in counts}}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
