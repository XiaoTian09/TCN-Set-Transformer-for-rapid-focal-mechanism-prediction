
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本1：SAC 波形读取与预处理（重活，只需运行一次）

功能：
    - 读取 SAC 文件（.r/.t/.z），转换为 ENZ 坐标系
    - 带通滤波 + 归一化（对完整长度 128 点做，不做窗口截取）
    - 只保留 Z/E/N 三个数据通道（方位角/离源角编码留到脚本2再生成，
      因为窗口截取长度、抖动策略都在脚本2里决定，编码通道长度必须跟窗口长度一致）
    - 生成 label（beachball图像）、label_sdr（128,3 高斯分布标签）
    - 每个事件同时保存原始的 az（方位角）、takeoff（离源角）数值数组，
      供脚本2按选中的台站子集直接取用，不需要重新读SAC文件

输出：
    output_h5:  每个事件一个 group（data_label_{event_id}），包含：
                    data      (30, 128, 3)  float32   -- Z/E/N 完整长度波形
                    label     (128, 128) 或 beachball实际形状           -- 沙滩球图像标签
                    label_sdr (128, 3)   float32       -- strike/dip/rake 高斯分布标签
                    az        (30,)      float32       -- 各台站方位角(度)
                    takeoff   (30,)      float32       -- 各台站离源角(度)
    output_csv: 每行一个事件，包含 depth/strike/dip/rake/selection_type/MAG/entropy
                （继承自输入CSV）+ az_values/takeoff_values（逗号分隔字符串，人工可读+校验用）

注意：
    group 名直接用真实事件编号 event_id（即输入CSV的行号/SAC目录 data_{i} 的 i），
    脚本2读取时也直接按 event_id 拼接 group 名访问，不使用 f.keys() 遍历，
    避免 h5py 字典序与真实编号错位的问题。

@author: tianx (written by Claude)
"""

import os
import warnings

import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import norm
from scipy import signal
from tqdm import tqdm

from beachball_generator import generate_beachball

warnings.filterwarnings('ignore')


# ======================================================================
# SAC 文件读取与预处理
# ======================================================================

def extract_sac_metadata(stream):
    """从SAC头段提取关键信息"""
    trace = stream[0]
    sac_header = trace.stats.sac
    return {
        'dist': sac_header.get('dist', -999),
        'az': sac_header.get('az', -999),
        't1': sac_header.get('t1', -999),
        'b': sac_header.get('b', -999),
        'e': sac_header.get('e', -999),
        'delta': trace.stats.delta,
        'npts': trace.stats.npts,
    }


def process_sac_directory(directory):
    """处理单个目录中的SAC文件，返回该目录所有台站的 DataFrame"""
    import obspy
    files = [f for f in sorted(os.listdir(directory)) if f.endswith(('.r', '.t', '.z'))]
    if not files:
        print(f"警告: 目录 {directory} 中没有SAC文件")
        return pd.DataFrame()

    station_groups = defaultdict(list)
    for f in files:
        base = os.path.splitext(f)[0]
        station_groups[base].append(f)

    records = []
    for base, components in station_groups.items():
        try:
            station_data = {'station': base, 'data_dir': os.path.basename(directory)}
            waveform_data = {}

            for comp_file in components:
                comp = os.path.splitext(comp_file)[1][1:]
                filepath = os.path.join(directory, comp_file)

                stream = obspy.read(filepath)
                trace = stream[0]
                if not np.isclose(trace.stats.sampling_rate, 1.0):
                    raise ValueError(f"SAC input must be 1 Hz: {filepath}")
                if not np.isfinite(trace.data).all():
                    raise ValueError(f"SAC contains NaN/Inf: {filepath}")

                if comp == 'z':
                    station_data.update(extract_sac_metadata(stream))

                waveform_data[f'{comp}_data'] = trace.data[:]
                waveform_data[f'{comp}_fullpath'] = filepath

            station_data.update(waveform_data)
            records.append(station_data)

        except Exception as e:
            print(f"处理 {directory}/{base} 时出错: {str(e)}")

    return pd.DataFrame(records)


def rtz_to_enz1(R, T, Z, azimuth):
    """将RTZ分量转换回ENZ坐标系"""
    theta = np.radians(azimuth)[:, np.newaxis]
    N = R * np.cos(theta) - T * np.sin(theta)
    E = R * np.sin(theta) + T * np.cos(theta)
    return E, N, Z


def strike_dip(n, e, u):
    """由法向量分量求走向、倾角（来自 ObsPy）"""
    r2d = 180 / np.pi
    if u < 0:
        n, e, u = -n, -e, -u

    strike = np.arctan2(e, n) * r2d
    strike = strike - 90
    while strike >= 360:
        strike -= 360
    while strike < 0:
        strike += 360
    x = np.sqrt(n ** 2 + e ** 2)
    dip = np.arctan2(x, u) * r2d
    return strike, dip


def aux_plane(s1, d1, r1):
    """求辅助节面（来自 ObsPy）"""
    r2d = 180 / np.pi
    z = (s1 + 90) / r2d
    z2 = d1 / r2d
    z3 = r1 / r2d

    sl1 = -np.cos(z3) * np.cos(z) - np.sin(z3) * np.sin(z) * np.cos(z2)
    sl2 = np.cos(z3) * np.sin(z) - np.sin(z3) * np.cos(z) * np.cos(z2)
    sl3 = np.sin(z3) * np.sin(z2)
    strike, dip = strike_dip(sl2, sl1, sl3)

    n1 = np.sin(z) * np.sin(z2)
    n2 = np.cos(z) * np.sin(z2)
    h1 = -sl2
    h2 = sl1

    z = h1 * n1 + h2 * n2
    z = z / np.sqrt(h1 * h1 + h2 * h2)
    eps = 2.2204460492503131e-16
    if 1.0 < abs(z) < 1.0 + 100 * eps:
        z = np.copysign(1.0, z)
    z = np.arccos(z)

    rake = z * r2d if sl3 > 0 else -z * r2d
    return strike, dip, rake


def mapminmax(arr):
    """归一化到 [0, 1] 区间（含分母为0保护）"""
    lo, hi = np.min(arr), np.max(arr)
    if hi - lo == 0:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def process_label(srcs_strike, srcs_dip, srcs_rake):
    """生成 strike/dip/rake 的高斯分布标签 (128, 3)"""
    len_out = 128
    x = np.arange(1, len_out + 1)

    str2, dip2, rake2 = aux_plane(srcs_strike, srcs_dip, srcs_rake)
    plane1 = {'strike': srcs_strike, 'dip': srcs_dip, 'rake': srcs_rake}
    plane2 = {'strike': str2, 'dip': dip2, 'rake': rake2}
    selected_plane = plane1 if plane1['strike'] <= plane2['strike'] else plane2

    selected_strike = selected_plane['strike']
    selected_dip = selected_plane['dip']
    selected_rake = selected_plane['rake']

    mu_strike = len_out / 360 * selected_strike
    strike_normalized = mapminmax(norm.pdf(x, mu_strike, 4))

    mu_dip = len_out / 90 * selected_dip
    dip_normalized = mapminmax(norm.pdf(x, mu_dip, 4))

    mu_rake = len_out / 360 * (selected_rake + 180)
    rake_normalized = mapminmax(norm.pdf(x, mu_rake, 4))

    return np.column_stack((strike_normalized, dip_normalized, rake_normalized))


def bandpass_filter_array(data_2d, f_low, f_high, fs=1.0, order=4):
    """对 (N, T) 数组逐行做带通滤波"""
    sos = signal.butter(order, [f_low, f_high], btype='bandpass', fs=fs, output='sos')
    filtered = np.zeros_like(data_2d)
    for i in range(data_2d.shape[0]):
        filtered[i, :] = signal.sosfiltfilt(sos, data_2d[i, :])
    return filtered


# ======================================================================
# 主流程
# ======================================================================

def build_raw_dataset(
    csv_path,
    base_dir,
    takeoff_dir,
    output_h5,
    output_csv,
    full_length=128,
    bandpass_freq=(0.05, 0.1),
    sample_rate=1.0,
):
    """
    读取全部事件的 SAC 波形，生成完整长度(128点)的 Z/E/N 三通道数据，
    连同 label / label_sdr / az / takeoff 一起写入 H5，
    并输出对应的事件级元数据 CSV。
    """

    print("=" * 60)
    print("脚本1：SAC 波形读取与预处理（完整长度，未做窗口截取/增广）")
    print("=" * 60)

    print(f"\n[阶段1] 读取事件参数 CSV: {csv_path}")
    event_df = pd.read_csv(csv_path)
    print(f"   CSV 列名: {list(event_df.columns)}")

    required_cols = ['depth', 'strike', 'dip', 'rake']
    missing = [c for c in required_cols if c not in event_df.columns]
    if missing:
        raise ValueError(f"输入CSV缺少必需列: {missing}")

    has_uniformity_cols = all(c in event_df.columns for c in ['selection_type', 'MAG', 'entropy'])
    if not has_uniformity_cols:
        print("   警告: 输入CSV中未找到 selection_type/MAG/entropy 列，输出CSV中这三列将为空")

    depth = event_df['depth'].values
    strike = event_df['strike'].values
    dip = event_df['dip'].values
    rake = event_df['rake'].values
    num_folders = len(depth)
    print(f"   共 {num_folders} 个事件（行）")

    f_low, f_high = bandpass_freq

    csv_records = []
    n_written = 0

    with h5py.File(output_h5, 'w') as h5_out:
        for i in tqdm(range(num_folders), desc="处理事件"):
            dir_name = f"data_{i}"
            dir_path = os.path.join(base_dir, dir_name)

            if not os.path.exists(dir_path):
                print(f"警告: 目录 {dir_path} 不存在，跳过事件 {i}")
                continue

            df = process_sac_directory(dir_path)
            if df.empty:
                print(f"警告: 目录 {dir_path} 无有效数据，跳过事件 {i}")
                continue

            df = df.sort_values('dist')

            takeoff_file = os.path.join(takeoff_dir, f"depth_{depth[i]}_takeoff.txt")
            if not os.path.exists(takeoff_file):
                print(f"警告: takeoff文件 {takeoff_file} 不存在，跳过事件 {i}")
                continue

            junk = np.atleast_2d(np.loadtxt(takeoff_file))
            if junk.shape[0] != len(df) or junk.shape[1] < 3:
                raise ValueError(f"Takeoff table must have {len(df)} rows and >=3 columns: {takeoff_file}")
            if not np.isfinite(junk[:, 2]).all() or np.any((junk[:, 2] < 0) | (junk[:, 2] > 180)):
                raise ValueError(f"Invalid takeoff angles: {takeoff_file}")
            if not np.isfinite(df[['az', 'dist']].to_numpy()).all() or (df['dist'] < 0).any() or ((df['az'] < 0) | (df['az'] > 360)).any():
                raise ValueError(f"Missing/invalid SAC az/dist headers in {dir_path}")
            df['takeoff'] = junk[:, 2]

            sorted_df = df.sort_values(by='az')
            az, dist, takeoff = sorted_df[['az', 'dist', 'takeoff']].values.T

            z_data = np.array(sorted_df['z_data'].tolist())
            r_data = np.array(sorted_df['r_data'].tolist())
            t_data = np.array(sorted_df['t_data'].tolist())

            # 校验实际数据长度是否与期望的 full_length 一致（早发现早报错）
            actual_len = z_data.shape[1]
            if actual_len != full_length:
                print(f"警告: 事件 {i} 波形长度为 {actual_len}，与期望的 full_length={full_length} 不符，跳过")
                continue

            e_data, n_data, z_data = rtz_to_enz1(r_data, t_data, z_data, az)

            e_filtered = bandpass_filter_array(e_data, f_low, f_high, fs=sample_rate)
            n_filtered = bandpass_filter_array(n_data, f_low, f_high, fs=sample_rate)
            z_filtered = bandpass_filter_array(z_data, f_low, f_high, fs=sample_rate)

            for ii in range(len(az)):
                e_filtered[ii, :] = mapminmax(e_filtered[ii, :])
                n_filtered[ii, :] = mapminmax(n_filtered[ii, :])
                z_filtered[ii, :] = mapminmax(z_filtered[ii, :])

            traindata = np.stack([z_filtered, e_filtered, n_filtered], axis=2)  # (30, full_length, 3)

            if np.isnan(traindata).any():
                print(f"警告: 事件 {i} 波形数据包含 NaN，跳过")
                continue

            label_sdr = process_label(strike[i], dip[i], rake[i])
            label = generate_beachball(strike[i], dip[i], rake[i])

            if np.isnan(label).any() or np.isnan(label_sdr).any():
                print(f"警告: 事件 {i} 标签包含 NaN，跳过")
                continue

            # ---- 写入 H5（group 名直接用真实 event_id，脚本2按需直接访问） ----
            sample_group = h5_out.create_group(f'data_label_{i}')
            sample_group.create_dataset('data', data=traindata.astype('float32'))
            sample_group.create_dataset('label', data=label.astype('float32'))
            sample_group.create_dataset('label_sdr', data=label_sdr.astype('float32'))
            sample_group.create_dataset('az', data=np.asarray(az, dtype='float32'))
            sample_group.create_dataset('takeoff', data=np.asarray(takeoff, dtype='float32'))

            # ---- 收集CSV记录 ----
            record = {
                'event_id': i,
                'depth': depth[i],
                'strike': strike[i],
                'dip': dip[i],
                'rake': rake[i],
                'num_stations': len(az),
                'az_values': ','.join(f'{a:.2f}' for a in az),
                'takeoff_values': ','.join(f'{t:.2f}' for t in takeoff),
            }
            if has_uniformity_cols:
                record['selection_type'] = event_df.iloc[i]['selection_type']
                record['MAG'] = event_df.iloc[i]['MAG']
                record['entropy'] = event_df.iloc[i]['entropy']
            else:
                record['selection_type'] = np.nan
                record['MAG'] = np.nan
                record['entropy'] = np.nan

            csv_records.append(record)
            n_written += 1

    print(f"\n   成功写入事件数: {n_written} / {num_folders}")
    if n_written == 0:
        raise ValueError("没有任何事件成功写入，流程终止")

    csv_df = pd.DataFrame(csv_records)
    csv_df.to_csv(output_csv, index=False)

    print(f"? H5 文件已保存: {output_h5}")
    print(f"? CSV 文件已保存: {output_csv}")
    print(f"   （CSV 的 'event_id' 列即为 H5 中 data_label_{{event_id}} 组的真实编号）")

    print("\n=== 数据统计 ===")
    print(csv_df[['depth', 'strike', 'dip', 'rake']].describe())
    if has_uniformity_cols:
        print("\nselection_type 分布:")
        print(csv_df['selection_type'].value_counts())

    return csv_df


# ======================================================================
# 使用示例
# ======================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Convert 1 Hz synthetic RTZ SAC to full-length ZEN HDF5")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--waveform-dir", required=True)
    parser.add_argument("--takeoff-dir", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    for name in ("output_h5", "output_csv"):
        p = Path(getattr(args, name))
        if p.exists():
            raise FileExistsError(p)
        p.parent.mkdir(parents=True, exist_ok=True)
    build_raw_dataset(args.input_csv, args.waveform_dir, args.takeoff_dir,
                      args.output_h5, args.output_csv, full_length=128,
                      bandpass_freq=(0.05, 0.1), sample_rate=1.0)
