#!/usr/bin/env python3
"""Predict raw beachball images with the released MSE checkpoint; labels optional."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat
import torch

from train_tcn_settransformer_beachball_mse_pytorch import TCNSetTransformerBeachball

HERE = Path(__file__).resolve().parent


def read_sample(group):
    expected = {'waveform': (20, 52, 3), 'geometry': (20, 4), 'station_mask': (20,)}
    arrays = {}
    for key, shape in expected.items():
        array = np.asarray(group[key], dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'{group.name}/{key}: expected finite {shape}, got {array.shape}')
        arrays[key] = array
    mask = arrays['station_mask']
    if not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError(f'{group.name}: station_mask must be binary with at least one station')
    valid = mask.astype(bool)
    wave, geom = arrays['waveform'][valid], arrays['geometry'][valid]
    if wave.min() < -1e-5 or wave.max() > 1 + 1e-5:
        raise ValueError(f'{group.name}: expected waveforms normalized to [0,1]')
    if not np.allclose(geom[:, :2] ** 2 @ np.ones(2), 1, atol=1e-4) or not np.allclose(geom[:, 2:] ** 2 @ np.ones(2), 1, atol=1e-4):
        raise ValueError(f'{group.name}: geometry must contain sine/cosine pairs')
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-h5', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=HERE / 'checkpoints/best_checkpoint.pt')
    parser.add_argument('--output-mat', type=Path, default=HERE / 'outputs/predictions.mat')
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0, cuda:1, ...')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('batch-size and limit must be positive')
    mapping_path = args.output_mat.with_suffix('.samples.csv')
    if args.output_mat.exists() or mapping_path.exists():
        raise FileExistsError(f'Output already exists: {args.output_mat} or {mapping_path}')
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('output_mode', 'raw') != 'raw':
        raise ValueError('This entry point requires a raw-output MSE model')
    model = TCNSetTransformerBeachball(dropout=float(checkpoint.get('args', {}).get('dropout', 0.1)))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    predictions, mapping, targets = [], [], []
    with h5py.File(args.input_h5, 'r') as data, torch.inference_mode():
        for key, expected in [('component_order', 'ZEN'), ('normalization', 'legacy'),
                              ('geometry_order', 'sin_az,cos_az,sin_takeoff,cos_takeoff'),
                              ('sampling_rate_hz', 1.), ('window_length', 52), ('num_receivers', 20)]:
            if key in data.attrs and data.attrs[key] != expected:
                raise ValueError(f'{key} must be {expected}')
        names = sorted((n for n in data if n.startswith('data_label_') and n[11:].isdigit()), key=lambda n: int(n[11:]))
        names = names[:args.limit]
        if not names:
            raise ValueError('No data_label_<integer> groups in input HDF5')
        for start in range(0, len(names), args.batch_size):
            chunk = names[start:start + args.batch_size]
            values = [read_sample(data[name]) for name in chunk]
            batch = {key: torch.from_numpy(np.stack([v[key] for v in values])).to(device)
                     for key in ('waveform', 'geometry', 'station_mask')}
            images = model(**batch).cpu().numpy()[:, 0]
            if not np.isfinite(images).all():
                raise ValueError('Model produced non-finite predictions')
            predictions.append(images)
            for name in chunk:
                group = data[name]
                event_id = group.attrs.get('event_id', group.attrs.get('source_id', ''))
                if isinstance(event_id, bytes):
                    event_id = event_id.decode('utf-8')
                mapping.append(dict(sample_id=len(mapping), group_name=name, source_id=str(event_id)))
                targets.append([float(group.attrs.get(key, np.nan)) for key in ('strike', 'dip', 'rake')])
    output = {'y_pred': np.concatenate(predictions), 'sample_ids': np.arange(len(mapping))}
    target = np.asarray(targets)
    # Never invent reference mechanisms for unlabeled or partly labeled input.
    if np.isfinite(target).all():
        output.update({key: target[:, i] for i, key in enumerate(('strike', 'dip', 'rake'))})
    elif np.isfinite(target).any():
        print('Mixed/incomplete reference SDR: omitting reference arrays; no Kagan evaluation in this batch.')
    args.output_mat.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.output_mat, output, do_compression=True)
    with mapping_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sample_id', 'group_name', 'source_id'])
        writer.writeheader()
        writer.writerows(mapping)
    print(f'Predicted {len(mapping)} events: {args.output_mat}\nEvent mapping: {mapping_path}')


if __name__ == '__main__':
    main()
