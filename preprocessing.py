"""Signal processing and station selection from the INSTANCE transfer pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal

ORIGINAL_FS = 100.0

TARGET_FS = 1.0

DOWNSAMPLE_FACTOR = 100

WINDOW_LEN = 52

PRE_P_SAMPLES = 5

F_LOW = 0.05

F_HIGH = 0.1

FILTER_ORDER = 4

TARGET_STATIONS = 20

CONSTANT_TOL = 1.0e-10

def waveform_quality_reason(waveform: np.ndarray) -> str | None:
    if waveform.ndim != 2 or waveform.shape[0] != 3:
        return f"形状应为 (3,N)，实际为 {waveform.shape}"
    for index, name in enumerate(("E", "N", "Z")):
        component = waveform[index]
        if not np.all(np.isfinite(component)):
            return f"{name}分量包含 NaN/Inf"
        if np.all(np.abs(component) <= CONSTANT_TOL):
            return f"{name}分量全零"
        if np.ptp(component) <= CONSTANT_TOL:
            return f"{name}分量恒定"
    return None

def preprocess_trace(waveform: np.ndarray, p_arrival_sample: float) -> np.ndarray:
    reason = waveform_quality_reason(waveform)
    if reason:
        raise ValueError(f"原始波形质控失败: {reason}")
    downsampled = signal.decimate(
        waveform, DOWNSAMPLE_FACTOR, ftype="fir", axis=1, zero_phase=True
    )
    detrended = signal.detrend(downsampled, axis=1, type="constant")
    detrended = signal.detrend(detrended, axis=1, type="linear")
    sos = signal.butter(
        FILTER_ORDER, [F_LOW, F_HIGH], btype="bandpass", fs=TARGET_FS, output="sos"
    )
    filtered = signal.sosfiltfilt(sos, detrended, axis=1)
    if not np.isfinite(p_arrival_sample):
        raise ValueError("P波到时无效")
    p_index = int(round(float(p_arrival_sample) / DOWNSAMPLE_FACTOR))
    start, end = p_index - PRE_P_SAMPLES, p_index - PRE_P_SAMPLES + WINDOW_LEN
    if start < 0 or end > filtered.shape[1]:
        raise ValueError(f"P波窗口越界: {start}:{end}/{filtered.shape[1]}")
    window = filtered[:, start:end]
    reason = waveform_quality_reason(window)
    if reason:
        raise ValueError(f"最终窗口质控失败: {reason}")
    return window.astype(np.float32)

def normalize_legacy(waves: np.ndarray) -> np.ndarray:
    output = np.empty_like(waves, dtype=np.float32)
    for station_index in range(waves.shape[0]):
        for component_index in range(waves.shape[1]):
            component = waves[station_index, component_index]
            minimum = float(component.min())
            span = float(component.max() - minimum)
            if span <= CONSTANT_TOL:
                raise ValueError("归一化前分量恒定")
            output[station_index, component_index] = (component - minimum) / span
    return output

def physical_station_key(row: pd.Series) -> str:
    values = []
    for column in ("station_network_code", "station_code", "station_location_code"):
        value = row[column]
        values.append("" if pd.isna(value) else str(value).strip())
    return ".".join(values)

def select_stations(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    work["path_azimuth_deg"] = pd.to_numeric(work["path_azimuth_deg"], errors="coerce") % 360.0
    work["_z_snr"] = pd.to_numeric(work["trace_Z_snr_db"], errors="coerce").fillna(-np.inf)
    work = work.sort_values("_z_snr", ascending=False).drop_duplicates("_station_key", keep="first")
    if len(work) < TARGET_STATIONS:
        raise ValueError(f"波形质控和物理台站去重后仅 {len(work)} 台")
    work = work.sort_values("path_azimuth_deg").reset_index(drop=True)
    sector_size = 360.0 / TARGET_STATIONS
    sector_ids = np.floor(work["path_azimuth_deg"] / sector_size).astype(int).clip(0, TARGET_STATIONS - 1)
    selected: list[int] = []
    for sector in range(TARGET_STATIONS):
        candidates = work.index[sector_ids == sector]
        if len(candidates):
            selected.append(int(work.loc[candidates, "_z_snr"].idxmax()))
    if len(selected) < TARGET_STATIONS:
        remaining = work.loc[~work.index.isin(selected)].sort_values(
            ["_z_snr", "path_azimuth_deg"], ascending=[False, True]
        )
        selected.extend(remaining.index[: TARGET_STATIONS - len(selected)].tolist())
    return work.loc[selected].sort_values("path_azimuth_deg").reset_index(drop=True)

def circular_gap(azimuth: np.ndarray) -> float:
    values = np.sort(np.asarray(azimuth, dtype=float) % 360.0)
    return float(np.diff(np.r_[values, values[0] + 360.0]).max())
