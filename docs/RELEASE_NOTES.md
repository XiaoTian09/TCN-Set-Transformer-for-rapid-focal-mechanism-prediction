# Release preparation

Prepared 2026-09-08 from the local `INSTANCE_test` project. The experiment tree was
left intact. Only selected code and the requested weight file were copied into
this independent repository directory.

## Preserved computation

- Original MSE model layers, forward computation, loss and training loop.
- Original circular-ZNCC SDR renderer and 15° → 3° → 1° search algorithms.
- Original moment-tensor conversion and Kagan-angle utility.
- Exact signal processing / legacy normalization functions used by the original
  transfer-dataset builder, extracted into `preprocessing.py`.
- Original event-level split and station-combination algorithm for labeled transfer data.
- Requested checkpoint, unchanged, with a SHA-256 checksum.

## Adaptations for distribution

- Replaced machine-specific source/data defaults with local imports and relative paths.
- Imported training code no longer overwrites CUDA visibility. Training CLI GPU
  selection still occurs before PyTorch import; fine-tuning keeps its configuration.
- Added native-label-optional prediction, input validation, and a CSV event-ID mapping.
- Added standalone observed waveform preparation using extracted processing functions.
- Added explicit TauP CLI with an externally supplied velocity model and temporary
  model compilation directory. It uses `p`/`P` rays, as in the original unlabeled pipeline.
- Converted original synthetic SAC script's hard-coded entry point to CLI arguments;
  added sampling-rate, finite-value and takeoff-table checks. Waveform filenames
  are traversed deterministically; takeoff association still follows sorted SAC distance.
- Kept the headless label generator from the observed-data pipeline.
- Preserved identifier strings, including leading zeros, when reading event metadata.
- Removed dynamic imports of source files outside the distribution from the labeled builder.
- Existing labeled output-data directories are refused; no recursive overwrite option.
- Fixed the SDR MAT loader's `(N,3)` `y_test` reference-SDR case, which previously
  attempted image validation before reaching the angle-label branch. Added checks
  for empty predictions, non-finite references and mismatched sample counts.
- Added bilingual instructions, schemas, provenance, and generated-signal tests.

`source_manifest.json` maps copied/extracted code to original project-relative paths
and records original source checksums. Distribution checksums are added separately
as `release_manifest.json`. The new entry points and tests have no corresponding
single source file. This package does not include datasets or historic result plots.

The synthetic augmentation module accepts a **preprocessed** `/noise` library.
The example uses no real noise because no noise data are distributed. Users may
provide their own library and enable the original `real_noise` augmentation option.

This is a local release bundle, not a GitHub deployment. No remote URL, author
identity, paper DOI or redistribution license was invented during preparation.
