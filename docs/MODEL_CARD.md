# Released model

## Identity and provenance

- File: `checkpoints/best_checkpoint.pt`.
- Original project-relative file: `instance_transfer_learning/mse_theory_transfer/outputs/best_checkpoint.pt`.
- Original checkpoint copied byte for byte, including its historical configuration metadata.
- File size: **13,834,833 bytes** (approximately 13.19 MiB).
- Trainable architecture parameters: **3,446,850**.
- Stored epoch: **11**, using the fine-tuning script's one-based epoch counter.
- Stored best validation loss: **0.013893239084422958** (unweighted pixel MSE).
- Model output: **raw**, without sigmoid or range constraints.
- SHA-256: see `checkpoints/SHA256SUMS`.

The checkpoint stores `model_state_dict`, epoch, validation loss, source-checkpoint
path, training arguments, output mode and loss name. Historical absolute paths
remain inside its metadata to preserve the original file. They are not used by
`predict.py` to locate inputs or load the model. No waveform or label arrays and
no optimizer/scheduler states are stored in this checkpoint.

This is the **post-transfer** model. The earlier theoretical-data checkpoint
is not part of the distribution. `train_tcn_settransformer_beachball_mse_pytorch.py`
reproduces the architecture and MSE training implementation; `finetune_mse.py`
provides the additional training stage used for the released weights.

## Architecture and input

20 stations × 52 temporal samples × 3 ZEN waveform components, at 1 Hz, with
4 geometric features and a station mask. The shared station TCN produces 128
features; the geometry MLP contributes 32; their concatenation is projected into
192-dimensional tokens. Two Set Transformer blocks use four attention heads,
and an event token feeds the image decoder. Output shape is `(batch,1,128,128)`.

Observed input processing uses 0.05–0.1 Hz filtering and `legacy` per-component
normalization. Preserve these conventions when using the supplied checkpoint.
The station mask implementation is not evidence that predictions with arbitrary
numbers of missing stations are validated; the distributed preparers select 20.

## Training context

The original transfer experiment used labeled INSTANCE events and an event-level
train/validation/test partition. Its accompanying experiment README reports
290/66/57 retained events. Those event lists and datasets are not distributed,
so those exact splits and final scientific performance are not independently
reproduced by the release tests.

Stored training configuration: seed 20260807, batch 16, maximum 30 epochs, learning
rate 2e-5, weight decay 1e-4, dropout 0.1, frozen station TCN, validation early-stop
patience 6, scheduler patience 2, minimum LR 1e-7, bfloat16 precision. The released
loss and output conventions match the selected MSE training implementation.

## Interpretation and scope

The network predicts an image. The supplied SDR solver fits a double-couple
beachball through circular image ZNCC; it outputs an equivalent nodal plane.
The solver's finite search grid and rasterization affect numerical SDR precision.
ZNCC is not a probability that the mechanism is correct. An actual Kagan-angle
comparison needs a reference mechanism; unlabeled events cannot supply one.

Applying the checkpoint to other frequency bands, component conventions, regions,
station distributions or normalization schemes requires separate evaluation.
The tests establish executable software compatibility, not generalization accuracy.

## Verification and redistribution

Strict checkpoint loading and CPU predictions were tested. The release model's
forward output was exactly equal to that of the original requested training class
for the same checkpoint and generated inputs. Model SHA-256 matched the original.
See `VALIDATION.md` for the complete checks and environment.

No code/model license has been assigned by this packaging operation. The repository
owner controls the intended license and publication metadata.
