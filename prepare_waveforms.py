#!/usr/bin/env python3
"""Prepare native model input from 100 Hz ENZ waveforms and station metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import preprocessing as pre


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--waveforms', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-depth-km', type=float, default=60)
    parser.add_argument('--max-distance-km', type=float, default=200)
    args = parser.parse_args()
    required = {'source_id', 'trace_name', 'station_network_code', 'station_code',
                'station_location_code', 'source_depth_km', 'path_ep_distance_km',
                'path_azimuth_deg', 'takeoff_angle_deg', 'trace_P_arrival_sample'}
    ids = ['source_id', 'trace_name', 'station_network_code', 'station_code', 'station_location_code']
    frame = pd.read_csv(args.metadata, dtype={key: str for key in ids})
    if required - set(frame):
        raise ValueError(f'Missing metadata columns: {sorted(required - set(frame))}')
    if frame[ids[:2]].isna().any().any():
        raise ValueError('source_id and trace_name must be present')
    if args.max_depth_km <= 0 or args.max_distance_km <= 0:
        raise ValueError('Depth and distance limits must be positive')
    if 'trace_Z_snr_db' not in frame:
        frame['trace_Z_snr_db'] = np.nan
    frame = frame.drop_duplicates('trace_name')
    reports = [args.output, args.output.with_suffix('.events.csv'), args.output.with_suffix('.qc.csv')]
    for path in reports:
        if path.exists():
            raise FileExistsError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records, qc = [], []
    with h5py.File(args.waveforms, 'r') as raw, h5py.File(args.output, 'x') as out:
        traces = raw['data'] if 'data' in raw and isinstance(raw['data'], h5py.Group) else raw
        out.attrs.update(schema='tcn_settransformer_v1', component_order='ZEN',
                         geometry_order='sin_az,cos_az,sin_takeoff,cos_takeoff',
                         sampling_rate_hz=1., window_length=52, num_receivers=20,
                         normalization='legacy', bandpass_hz=[0.05, 0.1])
        for event_id, rows in frame.groupby('source_id', sort=True):
            valid, waves = [], {}
            for _, row in rows.iterrows():
                trace = str(row.trace_name)
                try:
                    depth, distance, az, takeoff, arrival = map(float, (
                        row.source_depth_km, row.path_ep_distance_km, row.path_azimuth_deg,
                        row.takeoff_angle_deg, row.trace_P_arrival_sample))
                    if not np.isfinite([depth, distance, az, takeoff, arrival]).all():
                        raise ValueError('Non-finite depth/distance/geometry/arrival')
                    if not (0 <= depth < args.max_depth_km and 0 <= distance < args.max_distance_km):
                        raise ValueError('Depth or distance outside configured range')
                    if not 0 <= takeoff <= 180:
                        raise ValueError('takeoff_angle_deg must be in [0,180]')
                    row = row.copy()
                    row['_station_key'] = pre.physical_station_key(row)
                    if not str(row.station_code).strip() or pd.isna(row.station_code):
                        raise ValueError('Missing station_code')
                    waves[trace] = pre.preprocess_trace(traces[trace][()], arrival)
                    valid.append(row)
                except (ValueError, KeyError, TypeError) as exc:
                    qc.append(dict(source_id=event_id, trace_name=trace, reason=str(exc)))
            try:
                if not valid:
                    raise ValueError('No usable traces')
                selected = pre.select_stations(pd.DataFrame(valid))
                window = pre.normalize_legacy(np.stack([waves[n] for n in selected.trace_name]))
            except ValueError as exc:
                qc.append(dict(source_id=event_id, trace_name='', reason=str(exc)))
                continue
            az = np.deg2rad(selected.path_azimuth_deg.to_numpy(float) % 360)
            takeoff = np.deg2rad(selected.takeoff_angle_deg.to_numpy(float))
            group = out.create_group(f'data_label_{len(records)}')
            group.create_dataset('waveform', data=window[:, [2, 0, 1], :].transpose(0, 2, 1), compression='gzip')
            group.create_dataset('geometry', data=np.stack([np.sin(az), np.cos(az), np.sin(takeoff), np.cos(takeoff)], 1).astype('float32'))
            group.create_dataset('station_mask', data=np.ones(20, dtype='float32'))
            group.attrs.update(event_id=event_id, trace_names=json.dumps(selected.trace_name.tolist()))
            records.append(dict(sample_id=len(records), source_id=event_id,
                                group_name=group.name.lstrip('/'),
                                azimuthal_gap_deg=pre.circular_gap(np.rad2deg(az))))
    pd.DataFrame(records, columns=['sample_id', 'source_id', 'group_name', 'azimuthal_gap_deg']).to_csv(reports[1], index=False)
    pd.DataFrame(qc, columns=['source_id', 'trace_name', 'reason']).to_csv(reports[2], index=False)
    if not records:
        raise ValueError(f'No events passed QC. See {reports[2]}')
    print(f'Prepared {len(records)} events: {args.output}')


if __name__ == '__main__':
    main()
