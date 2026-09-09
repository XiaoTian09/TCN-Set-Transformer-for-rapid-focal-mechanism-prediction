# INSTANCE focal-mechanism prediction

Predict a 128 × 128 focal-mechanism beachball from 20 three-component seismic
waveforms and station geometry, then estimate strike, dip and rake (SDR).
The network combines a shared temporal convolutional network (TCN), a Set
Transformer and an image decoder trained with unweighted pixel MSE.

This repository includes the **INSTANCE-transfer checkpoint requested for release**,
preprocessing, synthetic-data augmentation, training, transfer learning and inference.
Training/validation/test waveforms, event catalogs, noise libraries and generated
results are not distributed. The tests generate their own artificial signals.

```text
100 Hz ENZ waveforms + P arrivals + station geometry
    → preprocessing → 20 × 52 × 3 ZEN waveforms + 20 × 4 geometry
    → TCN–Set Transformer → raw 128 × 128 image
    → circular ZNCC template search → strike, dip, rake, ZNCC
```

## Files

| File | Purpose |
| --- | --- |
| `checkpoints/best_checkpoint.pt` | Released, already fine-tuned model; original file preserved |
| `prepare_waveforms.py`, `preprocessing.py` | Raw observed waveform processing and unlabeled input preparation |
| `compute_takeoff.py` | Optional P-ray takeoff calculation using a user-supplied velocity model |
| `prepare_synthetic_sac.py` | Synthetic 1 Hz RTZ SAC → filtered full-length ZEN HDF5 |
| `traindata_augment_sim2real_settransformer.py` | Full-length synthetic HDF5 → augmented native training input |
| `traindata_augment_sim2real.py` | Shared augmentation functions |
| `build_labeled_dataset.py` | Observed labeled data → disjoint event-level training/validation/test sets |
| `train_tcn_settransformer_beachball_mse_pytorch.py` | Model definition and MSE training from scratch |
| `finetune_mse.py` | INSTANCE transfer learning, freezing the station TCN by default |
| `predict.py` | Native HDF5 → predicted images and event-ID mapping |
| `beachball_sdr_optimized.py` | Predicted images → SDR; optional reference-based Kagan evaluation |
| `beachball_generator.py`, `model_sdr_sincos/` | Label rendering and Kagan-angle utilities |
| `configs/` | Editable relative-path configuration examples |
| `tests/test_pipeline.py` | CPU integration tests; generated signals only |

## Environment

The verified environment was Linux, **Python 3.10.20**, **PyTorch 2.12.0+cu132**;
core dependency versions are pinned in `requirements.txt`. These are the versions
observed in the existing working environment, not a claim that a fresh installation
or every other PyTorch/CUDA combination has been tested. CPU execution was used
for release validation. A compatible NVIDIA GPU is useful for full training.

```bash
conda create -n instance-fm python=3.10 pip -y
conda activate instance-fm
python -m pip install -r requirements.txt
# CPU-only installation:
python -m pip install 'torch>=2.4' --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For GPU execution, replace the CPU installation command with the command matching
your operating system and CUDA/driver from the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
The code uses `torch.amp.GradScaler`; PyTorch >= 2.4 is the intended API baseline,
but only the version above was validated. CPU inference uses float32. Training
supports float32, float16 and bfloat16; use float32 when BF16 is unsupported.
TensorFlow, GMT and a GUI are not required.

Optional, only for reading SAC files or calculating takeoff angles:

```bash
python -m pip install -r requirements-optional.txt
```

Run commands below **from the repository root**. JSON configuration paths are
resolved relative to the repository root; absolute user paths are also accepted.
CLI paths are relative to the current working directory.

## Predict using the released checkpoint

For native model input, the required per-event fields are `waveform`, `geometry`
and `station_mask`; labels are not required. See [data formats](docs/DATA_FORMAT.md).

```bash
python predict.py --input-h5 data/input.h5 \
  --checkpoint checkpoints/best_checkpoint.pt \
  --output-mat outputs/predictions.mat --device cpu --batch-size 16
# GPU alternative: --device cuda:0
python beachball_sdr_optimized.py --input-mat outputs/predictions.mat \
  --output-dir outputs/sdr --workers 4
```

`outputs/sdr/sdr_results.csv` contains `sample_id,strike,dip,rake,zncc`.
Join it to `outputs/predictions.samples.csv` on `sample_id` to recover the original
`source_id` and HDF5 group. `sample_id` is a row identifier, not an earthquake ID.
SDR angles are in degrees. The returned nodal plane is one of two equivalent
planes; its identification as the physical fault plane requires other evidence.

The model has a **raw, unbounded output head**. Do not apply sigmoid, thresholding,
PNG color conversion or per-image min–max normalization before SDR inversion.
Use the floating-point MAT output directly. The default circular-template search
uses 15° → 3° → 1° steps and a radius fraction of 0.345. `--workers 0` runs serially.

Kagan angles are computed only when **all events have reference** `strike`, `dip`
and `rake` attributes in the input HDF5. Mixed/incomplete references are omitted
with a message. Unlabeled events produce SDR and ZNCC, with no actual Kagan angle.
ZNCC measures image-template fit, not a calibrated probability of mechanism accuracy.
No `estimated_p_kagan_lt30` calibration model is distributed here.

## Prepare observed waveforms

Supply 100 Hz waveforms with components **E, N, Z** and metadata including P-arrival
sample indices and takeoff angles. At least 20 usable independent stations per
event are required by this preparation workflow.

```bash
python prepare_waveforms.py --metadata data/metadata_with_takeoff.csv \
  --waveforms data/waveforms.h5 --output data/input.h5
```

Processing follows the released model's transfer pipeline: FIR decimation to 1 Hz;
full-record demeaning/detrending and 0.05–0.1 Hz fourth-order zero-phase Butterworth
filtering; a 52-sample window `[P-5:P+47]`; station selection; independent per-station,
per-component min–max normalization to [0,1]; ENZ → ZEN reordering; sine/cosine
geometry encoding. The P index is rounded after division by 100. The script also
writes `.events.csv` and `.qc.csv` beside its output. It intentionally produces
unlabeled input even if reference mechanisms are present in metadata.

If takeoff angles are missing, provide your appropriate velocity model:

```bash
python compute_takeoff.py --metadata data/metadata.csv \
  --velocity-model data/velocity_model.tvel --output data/metadata_with_takeoff.csv
```

This selects the first available `p`/`P` arrival using distance in km / 111.19.
No velocity model is bundled. The model and arrival choice affect geometry;
use conventions consistent with your study. The original labeled preparation
used supplied takeoff angles; the release does not silently recompute them.

## Training and transfer learning

**Synthetic training input.** If starting from synthetic SAC, use the optional
ObsPy dependency and the directory/schema described in [data formats](docs/DATA_FORMAT.md):

```bash
python prepare_synthetic_sac.py --input-csv data/synthetic_events.csv \
  --waveform-dir data/waveformDir --takeoff-dir data/takeoffDir \
  --output-h5 data/synthetic_full.h5 --output-csv data/synthetic_full.csv
python traindata_augment_sim2real_settransformer.py \
  --input-h5 data/synthetic_full.h5 --input-csv data/synthetic_full.csv \
  --output-h5 data/train.h5 --normalization legacy \
  --recipes 'base;shift+station_gain+component_gain' --seed 42
```

Repeat for a separate validation event set. Split **events before augmentation**;
all station combinations and augmented copies of an event must stay in the same
partition. `real_noise` is optional and requires your own preprocessed noise HDF5;
it is intentionally excluded from the example command.

**Train the original MSE architecture from scratch:**

```bash
python train_tcn_settransformer_beachball_mse_pytorch.py \
  --train-h5 data/train.h5 --val-h5 data/validation.h5 \
  --output-dir outputs/pretrain --gpu 0 --precision bfloat16 \
  --batch-size 64 --epochs 50
```

For a CPU check use `--gpu cpu --precision float32 --num-workers 0 --batch-size 1`.
Resume a checkpoint produced by this training script with
`--resume outputs/pretrain/last_checkpoint.pt`. It contains optimizer/scheduler
state. The released transfer checkpoint does **not** contain those states and is
not a resumable checkpoint for this from-scratch training command.

**Build observed labeled data and fine-tune:**

Edit `configs/build_labeled.json` with your raw HDF5 and metadata paths, then:

```bash
python build_labeled_dataset.py --config configs/build_labeled.json
python finetune_mse.py --config configs/finetune.json
```

The builder preserves the original event-level split and station-combination
algorithm (default seed 20260807, fractions 0.70/0.15/0.15, up to five combinations).
Actual retained split counts depend on the supplied events and QC. An existing
output-data directory is refused. The fine-tuning configuration defaults to the
checkpoint you produce at `outputs/pretrain/best_checkpoint.pt`; that earlier
synthetic checkpoint is not bundled. To adapt the **released** model to your own
data, set `pretrained_checkpoint` to `checkpoints/best_checkpoint.pt`.

Fine-tuning defaults: batch 16, learning rate 2e-5, 30 maximum epochs, frozen TCN,
unweighted MSE, validation-based early stopping with patience 6. The test split
is evaluated for before/after comparison and does not select the checkpoint.
No dataset means the original published training run cannot be reproduced from
this repository alone. See the [model card](docs/MODEL_CARD.md) for provenance.


