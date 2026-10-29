# Validation

All public checks use generated geometry and temporary locally trained parameters.
They do not require an external dataset or pretrained parameters.

Run `python -m pytest -q` for:

- three-branch output sizes, five fusion levels and normalized spatial gates;
- finite complete-objective gradients for foreground and background images;
- strict parameter save/load round trips and rejection of missing checkpoints;
- circle and polygon rasterization, derived shape rules and annotation errors;
- hand-computed pixel overlap metrics and absent-class handling;
- deterministic disjoint generic splits and training-only augmentation;
- a CPU CLI walkthrough in a separate working directory, including image-only prediction.

`python scripts/check_release.py` checks the explicit source-file manifest and any
tracked Git files. It rejects binary payloads, extra tracked files and machine-specific
paths. A published commit must contain exactly the source manifest. The automated
workflow runs these checks on a CPU runner and does not upload generated outputs.

The local reference validation uses Python 3.13.5, PyTorch 2.7.1+cu118, NumPy 2.1.2,
Pillow 11.0.0 and PyYAML 6.0.3, with CPU execution.
The full-width model's outputs, total loss, input gradient and parameter gradients
were compared to the reference implementation using identical parameters and synthetic
inputs. This is a software-equivalence check, not a benchmark result. CUDA and
multi-GPU training are configurable; the automated workflow verifies CPU execution.

