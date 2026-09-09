"""Optional ObsPy tests using generated SAC and ObsPy's bundled velocity model."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('obspy'), 'Install requirements-optional.txt for ObsPy tests')
class OptionalTest(unittest.TestCase):
    def run_script(self, script, *args):
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
            cwd=self.work, capture_output=True, text=True, timeout=120,
            env=dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(self.work / 'mpl')))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='instance-optional-test-')
        self.work = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_sac_preprocessing(self):
        from obspy import Trace
        wave = self.work / 'waveforms/data_0'
        wave.mkdir(parents=True)
        takeoff = self.work / 'takeoff'
        takeoff.mkdir()
        for station in range(20):
            for index, component in enumerate(('r', 't', 'z')):
                trace = Trace(np.sin(np.arange(128) * .45 + index + station*.1).astype('float32'))
                trace.stats.sampling_rate = 1.
                trace.stats.sac = dict(dist=10.+station, az=float(station*18))
                trace.write(str(wave / f'station{station:02d}.{component}'), format='SAC')
        np.savetxt(takeoff / 'depth_10.0_takeoff.txt', np.column_stack([np.arange(20), np.arange(20), np.full(20,45)]))
        pd.DataFrame([dict(depth=10., strike=120., dip=45., rake=-90.)]).to_csv(self.work / 'events.csv', index=False)
        self.run_script('prepare_synthetic_sac.py', '--input-csv', self.work / 'events.csv',
            '--waveform-dir', wave.parent, '--takeoff-dir', takeoff,
            '--output-h5', self.work / 'full.h5', '--output-csv', self.work / 'full.csv')
        with h5py.File(self.work / 'full.h5') as data:
            self.assertEqual(data['data_label_0/data'].shape, (20,128,3))
            self.assertTrue(np.isfinite(data['data_label_0/data'][:]).all())

    def test_takeoff_calculation(self):
        import obspy.taup
        model = Path(obspy.taup.__file__).parent / 'data/iasp91.tvel'
        pd.DataFrame([dict(source_id='001', source_depth_km=10., path_ep_distance_km=50.)]).to_csv(self.work / 'metadata.csv', index=False)
        self.run_script('compute_takeoff.py', '--metadata', self.work / 'metadata.csv',
                        '--velocity-model', model, '--output', self.work / 'takeoff.csv')
        result = pd.read_csv(self.work / 'takeoff.csv', dtype={'source_id': str})
        self.assertEqual(result.source_id.iloc[0], '001')
        self.assertTrue(0 <= result.takeoff_angle_deg.iloc[0] <= 180)


if __name__ == '__main__':
    unittest.main()
