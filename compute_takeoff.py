#!/usr/bin/env python3
"""Add P-wave takeoff angles to station metadata using a supplied TauP model."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--velocity-model', type=Path, required=True, help='User-supplied .tvel, .nd or compiled TauP .npz')
    args = parser.parse_args()
    from obspy.taup import TauPyModel
    from obspy.taup.taup_create import build_taup_model
    if args.output.exists():
        raise FileExistsError(args.output)
    # String IDs preserve leading zeros during CSV round trips.
    table = pd.read_csv(args.metadata, dtype={'source_id': str, 'trace_name': str,
                       'station_network_code': str, 'station_code': str, 'station_location_code': str})
    with tempfile.TemporaryDirectory(prefix='instance-taup-') as directory:
        model_path = args.velocity_model.resolve()
        if model_path.suffix.lower() != '.npz':
            build_taup_model(str(model_path), output_folder=directory)
            model_path = Path(directory) / (model_path.stem + '.npz')
        model = TauPyModel(model=str(model_path))
        values = []
        for row in table.itertuples():
            arrivals = model.get_travel_times(source_depth_in_km=float(row.source_depth_km),
                distance_in_degree=float(row.path_ep_distance_km) / 111.19, phase_list=['p', 'P'])
            values.append(float(arrivals[0].takeoff_angle) if arrivals else np.nan)
    table['takeoff_angle_deg'] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f'{args.output}: {len(table)} traces; missing rays: {int(pd.isna(values).sum())}')


if __name__ == '__main__':
    main()
