# Data formats and conventions

## Raw observed input

HDF5: `/data/<trace_name>` or `/<trace_name>`, float arrays `(3,N)` in **E,N,Z**
order, sampled at **100 Hz**. Typical records contain 12000 samples (120 seconds).
Filtering is performed on the full record; do not supply only the 52-second window.
P-arrival indices are zero-based indices in the original 100 Hz record. The output
window starts at `round(trace_P_arrival_sample / 100) - 5` and contains 52 samples.

CSV: one row per waveform/physical station record. Required for `prepare_waveforms.py`:

| Field | Meaning |
| --- | --- |
| `source_id` | Earthquake identifier, read as a string |
| `trace_name` | HDF5 dataset key |
| `station_network_code` | Network identifier |
| `station_code` | Station identifier |
| `station_location_code` | Location identifier; blank is allowed |
| `source_depth_km` | Source depth in km |
| `path_ep_distance_km` | Epicentral distance in km |
| `path_azimuth_deg` | Source-to-station azimuth, degrees clockwise from north; not back azimuth |
| `takeoff_angle_deg` | P-ray takeoff angle, degrees from downward vertical, in [0,180] |
| `trace_P_arrival_sample` | P-arrival sample index in the original 100 Hz trace |
| `trace_Z_snr_db` | Optional station-ranking SNR; missing values have lowest priority |

Stations are deduplicated by network + station + location. The inference preparer
selects the highest-SNR station in each of 20 azimuth sectors and fills remaining
slots by SNR. Stations are sorted by azimuth. Fewer than 20 usable stations cause
an event to be rejected. The native model supports masks, but this preprocessing
workflow does not pad sparse events or establish their prediction quality.

For `build_labeled_dataset.py`, add `source_mechanism_strike_dip_rake`, for example
`strike=120, dip=45, rake=-90`. Commas within this field must be quoted in CSV.
The labeled builder selects randomized station combinations using the original
maximum azimuthal gap / overlap constraints. It does not impose the inference
preparer's deterministic station selection. It writes one `event_splits.csv` so
all combinations of an event remain together. Use this file to audit splits.

## Native model HDF5

Groups: `data_label_0`, `data_label_1`, etc. Numeric suffixes define sample ordering.

| Dataset | Shape | Type / meaning |
| --- | --- | --- |
| `waveform` | `(20,52,3)` | float32; **Z,E,N**; 1 Hz; per-station/per-component [0,1] |
| `geometry` | `(20,4)` | float32; `[sin(az),cos(az),sin(takeoff),cos(takeoff)]` |
| `station_mask` | `(20,)` | float32; 1=valid, 0=masked; at least one valid station |
| `label` | `(128,128)` | float32 grayscale beachball in [0,1]; **required for training only** |

Angles are provided in degrees in metadata, converted to radians before sine/cosine.
Per-group attribute `event_id` preserves the source ID. Optional `strike`, `dip`,
`rake` attributes describe a **reference** mechanism; never write predictions there.

Recommended root attributes: `component_order="ZEN"`,
`geometry_order="sin_az,cos_az,sin_takeoff,cos_takeoff"`, `sampling_rate_hz=1.0`,
`window_length=52`, `num_receivers=20`, `normalization="legacy"`.
`predict.py` enforces the released input shape; the general training class can
read other temporal lengths, but those are not the released model's validated input.
Old five-channel Gaussian-geometry HDF5 is not accepted by `predict.py`; regenerate
native input from original angle metadata, rather than guessing channel meanings.

Generate labels using the supplied `beachball_generator.generate_beachball`.
The red-filled/white background rendering becomes grayscale on conversion; its
geometry, circle size and intensity convention matter to image-based training.

## Synthetic input and augmentation

`prepare_synthetic_sac.py` expects:

- Event CSV with `depth,strike,dip,rake`. Zero-based row `i` maps to `waveformDir/data_i/`.
- Matching `station.r`, `station.t`, `station.z` SAC files with **128 samples at 1 Hz**;
  this script does not resample SAC input. SAC headers `dist` and `az` must be valid.
- `takeoffDir/depth_<depth>_takeoff.txt` with one row per station sorted by increasing
  SAC `dist`; **third column** is takeoff angle in degrees. The depth string must
  match the CSV value as read by pandas, e.g. `depth_10.0_takeoff.txt` for 10.0.
  The original row-based takeoff association is preserved, so row alignment matters.
- Its RTZ → ENZ convention is `N=R*cos(az)-T*sin(az)`, `E=R*sin(az)+T*cos(az)`.
  Confirm this convention against the synthetic solver that generated your SACs.

It filters at 0.05–0.1 Hz and normalizes each station/component over the full
128 samples. Its full-length HDF5 contains `data_label_<event_id>` groups with:
`data (S,128,3)` in ZEN, `az (S,)`, `takeoff (S,)`, `label (128,128)` and optional
`label_sdr`. The companion CSV contains `event_id,depth,strike,dip,rake`.

The augmentation script selects 20 stations, centers full traces, crops
`base_start=45`, `window_len=52` (plus configured time shifts), then normalizes the
window using `legacy`. It exports native model fields and records augmentation
metadata. Full-length synthetic input should already have the filter/sampling
conventions above; this augmentation stage does not apply them again.

If enabling `real_noise`, supply `--noise-library <path>` containing:
`noise (N,128,3)` float arrays in ZEN, already sampled at 1 Hz and preprocessed
consistently, and `trace_name (N,)` string IDs. No noise library is included. The
older helper module's introductory raw-noise description is superseded by its
`NoiseLibrary` implementation, which consumes this **preprocessed** format.

## Prediction and SDR outputs

`predictions.mat`: `y_pred (N,128,128)` float32 raw network images, `sample_ids`,
and optional all-events reference `strike,dip,rake` arrays. MATLAB v5 format is
written by SciPy; v7.3/HDF5 MAT is not the input format for the SDR converter.
`predictions.samples.csv` maps sequential `sample_id` to `group_name,source_id`.

SDR converter writes `sdr_results.csv`, `sdr_results.mat`, `summary.json`.
`zncc` is the circular zero-mean normalized correlation between the predicted
image and its best searched template. It is not a direct waveform correlation,
Kagan angle, or calibrated accuracy probability. Actual Kagan angles require a
second reference mechanism. Different SDR triplets can represent equivalent
nodal planes; compare mechanisms with Kagan rather than componentwise differences.
