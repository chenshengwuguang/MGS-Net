from pathlib import Path

import pytest
import torch

from mgs_net import MGSNet
from mgs_net.losses import build_loss
from mgs_net.metrics import SegmentationMeter
from mgs_net.runtime import load_config, load_trained_model

ROOT = Path(__file__).resolve().parents[1]


def test_outputs_and_spatial_gates():
    model = MGSNet(base_channels=4).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 1, 128, 256))
    assert output["logits"].shape == (1, 5, 128, 256)
    for key in ("small_logits", "large_logits", "elongated_logits"):
        assert output[key].shape == (1, 2, 128, 256)
    assert len(output["fusion_weights_by_level"]) == 5
    for gates in output["fusion_weights_by_level"]:
        torch.testing.assert_close(gates.sum(1), torch.ones_like(gates[:, 0]))
    assert output["position_attention"].min() >= 0
    assert output["position_attention"].max() <= 1
    assert output["elongated_position_attention"].count_nonzero() == 0


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 1, 31, 64), torch.float32),
        ((1, 3, 32, 64), torch.float32),
        ((1, 1, 32, 64), torch.uint8),
    ],
)
def test_invalid_input_is_actionable(shape, dtype):
    with pytest.raises(ValueError):
        MGSNet(base_channels=4)(torch.zeros(shape, dtype=dtype))


@pytest.mark.parametrize("background", [False, True])
def test_complete_loss_has_finite_gradients(background):
    torch.manual_seed(7)
    model = MGSNet(base_channels=4)
    config = load_config(ROOT / "configs/mgs_net.yaml")
    images = torch.rand(1, 1, 32, 64)
    targets = torch.zeros(1, 32, 64, dtype=torch.long)
    shape_targets = targets.clone()
    if not background:
        for class_id in range(1, 5):
            targets[:, 4:28, (class_id - 1) * 16 : class_id * 16] = class_id
        shape_targets = (targets % 3 + 1) * (targets > 0)
    loss = build_loss(config["loss"])(model(images), targets, shape_targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert model.classifier.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("wrapped", [False, True])
def test_strict_checkpoint_round_trip(tmp_path, wrapped):
    config = load_config(ROOT / "configs/synthetic_cpu.yaml")
    original = MGSNet(base_channels=4).eval()
    state = original.state_dict()
    checkpoint = tmp_path / "own.pth"
    torch.save({"model_state": state} if wrapped else state, checkpoint)
    restored = load_trained_model(config, checkpoint, "cpu")
    for key in state:
        torch.testing.assert_close(
            state[key], restored.state_dict()[key], rtol=0, atol=0
        )


def test_missing_checkpoint_never_runs_random_predictions():
    with pytest.raises(FileNotFoundError, match="Train with"):
        load_trained_model(
            load_config(ROOT / "configs/synthetic_cpu.yaml"), None, "cpu"
        )


def test_pixel_accumulation_and_absent_class_policy():
    meter = SegmentationMeter(5)
    target = torch.tensor([[[0, 1], [1, 0]]])
    prediction = torch.tensor([[[1, 1], [0, 0]]])
    logits = torch.nn.functional.one_hot(prediction, 5).permute(0, 3, 1, 2).float()
    meter.update(logits, target)
    values = meter.compute()
    assert values["mean_iou"] == pytest.approx(1 / 3)
    assert values["mean_dice"] == pytest.approx(0.5)
    assert values["iou_per_class"][2:] == [0, 0, 0]
