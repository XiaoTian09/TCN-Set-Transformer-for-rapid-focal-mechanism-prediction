# Release validation

Date: 2026-09-08. All checks used generated signals or the specified model file.
No real earthquake waveform, training split or event catalog was added to the package.

## Core execution

The repository was copied to a fresh temporary directory outside the experiment
project. Tests invoked scripts by their copied paths while operating in separate
temporary working directories, so original project imports/default paths were
not available through the working directory.

Command: `python -m unittest discover -s tests -v`.

- Five core test cases passed, including the complete raw-waveform → native HDF5
  → released model → MAT → SDR CSV route.
- The same integration case built disjoint labeled event partitions, ran one
  epoch of from-scratch MSE training and one epoch of frozen-TCN fine-tuning, and
  verified both saved best checkpoints.
- Event IDs with leading zeros survived preprocessing and prediction mapping.
- Unlabeled inputs generated no invented reference SDR or Kagan-angle columns.
- A known rendered mechanism was recovered with Kagan error below 5° using the
  default 15°/3°/1° search. Its image ZNCC is not expected to equal 1 because the
  final-size template rasterization differs from resizing the 128-pixel image.
- Synthetic HDF5 augmentation ran with base and gain/shift recipes without a noise library.
- Invalid all-masked inputs were rejected; auxiliary-plane Kagan equivalence was checked.
- The `(N,3)` MAT reference-label regression was exercised.
- Two ObsPy tests were skipped in this core environment because ObsPy was absent;
  both were executed separately as described below.

Core environment: Linux, Python 3.10.20, PyTorch 2.12.0+cu132 (CPU execution),
NumPy 2.2.5, SciPy 1.15.3, h5py 3.14.0, pandas 2.3.3, Pillow 12.2.0,
tqdm 4.68.2, Matplotlib 3.10.9. CPU thread counts were set to 2 for testing.

## Optional ObsPy execution

Command: `python tests/test_optional.py -v` from the relocated directory.
Both optional tests passed:

- Generated 1 Hz RTZ SAC files → full-length filtered/normalized ZEN HDF5.
- User-supplied velocity-model compilation → finite P takeoff angles and preserved IDs.
  The fixture used ObsPy's bundled `iasp91.tvel`, not the study's training data.

Optional environment: Python 3.10.20, ObsPy 1.5.1, NumPy 2.2.6, SciPy 1.15.2,
h5py 3.14.0, pandas 2.3.3, Pillow 12.2.0, Matplotlib 3.10.9, tqdm 4.68.3.

## Source compatibility and packaging

- Released checkpoint SHA-256 exactly matched the original user-selected checkpoint.
- Strict state-dictionary loading succeeded into the selected MSE architecture.
- Original versus release network outputs were exactly equal on identical generated
  input with the same checkpoint in evaluation mode.
- Every extracted preprocessing function's AST matched the corresponding function
  in the original transfer preprocessing dependency.
- All Python files parsed successfully; nine CLI `--help` entry points succeeded
  from the relocated distribution, without optional ObsPy being installed.
- Code and executable JSON configuration contain no original machine-specific
  `/data/tianx`, `/home`, or `/software` import/data dependencies. The unchanged
  checkpoint retains historical paths as unused metadata.
- The archive includes only source, documentation, configuration, tests and the
  selected checkpoint. Training-data formats and Python caches are excluded.

## Limits of validation

No full training rerun, new scientific accuracy benchmark, GPU training run or
fresh dependency installation was performed. CPU software checks demonstrate
compatibility of the packaged workflow, not reproduction of the original experiment's
accuracy or proof of quality on arbitrary regions/station configurations.
