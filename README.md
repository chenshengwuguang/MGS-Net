# MGS-Net

**Morphology-Guided Segmentation Network for Weld Defect Segmentation**

[中文说明](README.zh-CN.md) · [Method](docs/method.md) · [Data format](docs/data-format.md)

MGS-Net combines three morphology-specific encoder branches, spatial input enhancement,
layer-wise gated fusion, a U-shaped decoder and annotation-derived auxiliary supervision.
This repository includes model, loss, training, evaluation, image prediction and generic
data-preparation code. **Data and pretrained weights are not distributed.**

## Installation

Use Python 3.10 or newer and a compatible PyTorch installation (2.6 or newer).
An existing Conda environment with PyTorch is suitable. To install a CPU build in
a new environment, use the command from [PyTorch](https://pytorch.org/get-started/locally/), for example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

The other runtime dependencies are NumPy, Pillow and PyYAML. Run the commands below
from the repository root. Dependencies are declared in `pyproject.toml`.

## Synthetic CPU walkthrough

The generator draws geometric shapes from scratch. Its examples and predictions
check the software pipeline and do not measure performance on weld images.
The demonstration uses **4 base channels** for speed; the full model configuration
uses **32**, with the same branch structure and loss definitions.

```bash
python scripts/generate_synthetic_data.py
python scripts/train.py --config configs/synthetic_cpu.yaml --smoke-test
python scripts/evaluate.py --config configs/synthetic_cpu.yaml --checkpoint runs/synthetic/checkpoints/best.pth --device cpu
python scripts/predict.py --config configs/synthetic_cpu.yaml --checkpoint runs/synthetic/checkpoints/best.pth --input generated/synthetic/images/example_000.png --output-dir runs/predictions --device cpu
```

`--smoke-test` executes one training batch and one validation batch and saves locally
generated parameters. Omit it to run the two demonstration epochs. Existing training
or generated-data directories are protected against accidental overwriting; select
a new output directory when rerunning a command.

## Train with your own data

Prepare paired images and LabelMe annotations as described in [Data format](docs/data-format.md).
Set `data.root` in `configs/mgs_net.yaml` to your data directory. Keep the documented
class order. Split files are relative to this data root, as are paths inside those files.

```bash
python scripts/train.py --config configs/mgs_net.yaml
python scripts/evaluate.py --config configs/mgs_net.yaml --checkpoint runs/mgs_net/checkpoints/best.pth --split test --device cpu --batch-size 1
python scripts/predict.py --config configs/mgs_net.yaml --checkpoint runs/mgs_net/checkpoints/best.pth --input data/user/images --output-dir runs/user_predictions --device cpu
```

The full training configuration uses Adam, cosine learning-rate scheduling and
validation mean Dice to save the best parameters and stop training after a configured
number of epochs without improvement. Set `train.device_ids: [0]` for one GPU;
the provided `[0, 1]` setting requires two visible GPUs. CPU execution can be selected
with `--device cpu --batch-size 1`. Original input pixels are converted to grayscale
and divided by 255. Images are not resized; both dimensions must be multiples of 16.
The usual input size is **128 high × 256 wide**.

Prediction accepts a single image or a flat directory and needs no annotations. It
requires a checkpoint trained by you and writes a class-ID PNG, a color PNG and an
overlay for each input. Class IDs are background=0, incomplete penetration=1,
lack of fusion=2, porosity=3 and slag inclusion=4. Evaluation accumulates pixels
across the split before computing per-class IoU and Dice; means exclude background
and classes without ground-truth pixels. See [Method](docs/method.md).

## Python interface

```python
import torch
from mgs_net import MGSNet

model = MGSNet(base_channels=32)
output = model(torch.rand(1, 1, 128, 256))
scores = output["logits"]  # N,5,H,W; unnormalized class scores
mask = scores.argmax(dim=1)
```

The three auxiliary outputs have two channels each (background and branch foreground).
They are independent binary predictions, not a mutually exclusive shape classifier.
Compatible state dictionaries retain their module parameter names and load strictly.

## Tests

```bash
python -m pytest -q
python scripts/check_release.py
```

Tests use only generated geometry. They cover annotation geometry, invalid inputs,
loss gradients, spatial gates, parameter loading, metrics, data preparation and the
CPU command-line pipeline. The automated workflow runs these checks without external
data or pretrained parameters. See [Validation](docs/validation.md).

## License

MIT. Keep the copyright and license notice when using or redistributing the code.
Dependencies retain their own licenses.

