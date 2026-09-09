"""CPU integration tests using generated signals only; no seismic dataset needed."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import beachball_sdr_optimized as sdr
import predict


class PipelineTest(unittest.TestCase):
    def run_script(self, script, *args):
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   MPLBACKEND='Agg', MPLCONFIGDIR=str(self.work / 'mpl'))
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                cwd=self.work, env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='instance-release-test-')
        self.work = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_checkpoint_checksum(self):
        expected = (ROOT / 'checkpoints/SHA256SUMS').read_text().split()[0]
        actual = hashlib.sha256((ROOT / 'checkpoints/best_checkpoint.pt').read_bytes()).hexdigest()
        self.assertEqual(expected, actual)

    def test_kagan_symmetry_and_mat_angle_labels(self):
        angles = (123., 45., -90.)
        self.assertLess(sdr.get_kagan_angle(*angles, *angles), 1e-5)
        auxiliary = sdr.auxiliary_plane(*angles)
        self.assertLess(sdr.get_kagan_angle(*angles, *auxiliary), 1e-5)
        path = self.work / 'labels.mat'
        savemat(path, {'y_pred': sdr.render_beachball(*angles, 128)[None],
                      'y_test': np.asarray([angles])})
        loaded = sdr.load_input(path)
        np.testing.assert_allclose(loaded['target_sdr'], [angles])
        self.assertIsNone(loaded['y_test'])
        self.run_script('beachball_sdr_optimized.py', '--input-mat', path,
                        '--output-dir', self.work / 'known_sdr', '--workers', 0)
        results = pd.read_csv(self.work / 'known_sdr/sdr_results.csv')
        self.assertLess(results.kagan_angle_deg.iloc[0], 5.)
        # Rasterizing at the final search size differs from resizing the 128px
        # fixture; Kagan recovery, rather than near-unity image ZNCC, is the test.
        self.assertTrue(np.isfinite(results.zncc.iloc[0]))
        self.assertTrue(-1 <= results.zncc.iloc[0] <= 1)

    def test_invalid_input_rejected(self):
        with h5py.File(self.work / 'invalid.h5', 'w') as out:
            group = out.create_group('data_label_0')
            group['waveform'] = np.zeros((20, 52, 3), dtype='float32')
            group['geometry'] = np.tile([0, 1, 0, 1], (20, 1)).astype('float32')
            group['station_mask'] = np.zeros(20, dtype='float32')
            with self.assertRaisesRegex(ValueError, 'at least one station'):
                predict.read_sample(group)

    def test_synthetic_augmentation(self):
        full = self.work / 'synthetic.h5'
        with h5py.File(full, 'w') as out:
            group = out.create_group('data_label_0')
            t = np.arange(128)
            waves = np.stack([np.sin(t * .45 + p) for p in [.1, .4, .8]], axis=-1)
            group['data'] = np.tile(waves[None], (20, 1, 1))
            group['az'] = np.arange(20) * 18.
            group['takeoff'] = np.full(20, 45.)
            group['label'] = sdr.render_beachball(120, 45, -90, 128)
        pd.DataFrame([dict(event_id=0, depth=10, strike=120, dip=45, rake=-90)]).to_csv(self.work / 'synthetic.csv', index=False)
        native = self.work / 'augmented.h5'
        self.run_script('traindata_augment_sim2real_settransformer.py', '--input-h5', full,
            '--input-csv', self.work / 'synthetic.csv', '--output-h5', native,
            '--recipes', 'base;shift+station_gain+component_gain', '--num-station-selections', 1)
        with h5py.File(native) as data:
            self.assertEqual(len(data), 2)
            self.assertEqual(data['data_label_0/waveform'].shape, (20, 52, 3))
            self.assertEqual(data.attrs['normalization'], 'legacy')

    def test_raw_to_prediction_and_training(self):
        # Three generated events allow disjoint train/validation/test splits.
        rng = np.random.default_rng(123)
        rows = []
        raw = self.work / 'raw.h5'
        with h5py.File(raw, 'w') as out:
            data = out.create_group('data')
            t = np.arange(12000) / 100
            for event in range(3):
                for station in range(20):
                    name = f'event{event}_station{station}'
                    wave = np.stack([np.sin(2*np.pi*.075*t + phase + station*.05)
                                     for phase in (.1, .8, 1.6)])
                    data[name] = (wave + rng.normal(0, .01, wave.shape)).astype('float32')
                    rows.append(dict(source_id=f'00{event}', trace_name=name,
                        station_network_code='XX', station_code=f'S{station:02d}', station_location_code='00',
                        source_depth_km=10, path_ep_distance_km=40,
                        path_azimuth_deg=station*18, takeoff_angle_deg=45,
                        trace_P_arrival_sample=5000, trace_Z_snr_db=20,
                        source_mechanism_strike_dip_rake='strike=120, dip=45, rake=-90'))
        metadata = self.work / 'metadata.csv'
        pd.DataFrame(rows).to_csv(metadata, index=False)
        native = self.work / 'input.h5'
        self.run_script('prepare_waveforms.py', '--metadata', metadata, '--waveforms', raw, '--output', native)
        with h5py.File(native) as data:
            self.assertEqual(len(data), 3)
            self.assertNotIn('label', data['data_label_0'])
            self.assertEqual(data['data_label_0'].attrs['event_id'], '000')
        mat = self.work / 'predictions.mat'
        self.run_script('predict.py', '--input-h5', native, '--output-mat', mat, '--device', 'cpu', '--batch-size', 2)
        output = loadmat(mat)
        self.assertEqual(output['y_pred'].shape, (3, 128, 128))
        self.assertNotIn('strike', output)
        self.assertTrue(np.isfinite(output['y_pred']).all())
        mapping = pd.read_csv(mat.with_suffix('.samples.csv'), dtype={'source_id': str})
        self.assertEqual(mapping.source_id.tolist(), ['000', '001', '002'])
        self.run_script('beachball_sdr_optimized.py', '--input-mat', mat, '--output-dir', self.work / 'sdr',
                        '--limit', 1, '--workers', 0, '--coarse-step', 30, '--refine-step', 10,
                        '--final-step', 5, '--top-k', 1)
        with (self.work / 'sdr/sdr_results.csv').open(encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            self.assertNotIn('kagan_angle_deg', reader.fieldnames)
            self.assertEqual(len(list(reader)), 1)
        cfg = json.loads((ROOT / 'configs/build_labeled.json').read_text())
        transfer = self.work / 'transfer'
        cfg.update(metadata_csv=str(metadata), waveforms_h5=str(raw), output_data_dir=str(transfer),
                   split_fractions=[1/3, 1/3, 1/3])
        config = self.work / 'build.json'
        config.write_text(json.dumps(cfg))
        self.run_script('build_labeled_dataset.py', '--config', config)
        sets = []
        for split in ('train', 'validation', 'test'):
            with h5py.File(transfer / f'{split}.h5') as data:
                self.assertEqual(len(data), 1)
                sets.append({g.attrs['event_id'] for g in data.values()})
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        # Exercise actual backward/optimizer/checkpoint code on generated inputs.
        trained = self.work / 'pretrain'
        self.run_script('train_tcn_settransformer_beachball_mse_pytorch.py',
            '--train-h5', transfer / 'train.h5', '--val-h5', transfer / 'validation.h5',
            '--output-dir', trained, '--epochs', 1, '--batch-size', 1,
            '--num-workers', 0, '--gpu', 'cpu', '--precision', 'float32', '--log-interval', 0)
        self.assertTrue((trained / 'best_checkpoint.pt').is_file())
        cfg = json.loads((ROOT / 'configs/finetune.json').read_text())
        cfg.update(train_h5=str(transfer / 'train.h5'), validation_h5=str(transfer / 'validation.h5'),
                   test_h5=str(transfer / 'test.h5'), pretrained_checkpoint=str(ROOT / 'checkpoints/best_checkpoint.pt'),
                   output_dir=str(self.work / 'finetune'), gpu='cpu', num_workers=0, epochs=1, batch_size=1)
        config.write_text(json.dumps(cfg))
        self.run_script('finetune_mse.py', '--config', config, '--cpu')
        self.assertTrue((self.work / 'finetune/best_checkpoint.pt').is_file())


if __name__ == '__main__':
    unittest.main()
