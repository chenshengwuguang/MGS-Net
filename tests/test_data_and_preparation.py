import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mgs_net.data import LabelMeDataset
from mgs_net.preparation import affine_image, transform_annotation

ROOT = Path(__file__).resolve().parents[1]


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_synthetic_masks_include_background_circle_polygon_and_three_shapes(synthetic):
    dataset = LabelMeDataset(synthetic / "splits/train.txt", root=synthetic)
    sample = dataset[0]
    assert set(sample["mask"].unique().tolist()) == {0, 1, 2, 3, 4}
    assert set(sample["shape_mask"].unique().tolist()) == {0, 1, 2, 3}
    assert (sample["shape_mask"][sample["mask"] == 3] == 1).all()
    assert (sample["shape_mask"][sample["mask"] == 4] == 2).all()
    assert (sample["shape_mask"][sample["mask"] == 1] == 3).all()
    assert dataset[1]["mask"].count_nonzero() == 0
    assert dataset[1]["shape_mask"].count_nonzero() == 0


@pytest.mark.parametrize(
    "problem", ["unknown_label", "bad_circle", "wrong_size", "unsupported_shape"]
)
def test_bad_annotations_fail(synthetic, problem):
    dataset = LabelMeDataset(synthetic / "splits/train.txt", root=synthetic)
    path = dataset.samples[0]["json"]
    data = json.loads(path.read_text())
    if problem == "unknown_label":
        data["shapes"][0]["label"] = "unmapped"
    elif problem == "bad_circle":
        data["shapes"][2]["points"].pop()
    elif problem == "wrong_size":
        data["imageWidth"] = 32
    else:
        data["shapes"][0]["shape_type"] = "rectangle"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        dataset[0]


def test_missing_pair_and_empty_split(tmp_path):
    split = tmp_path / "split.txt"
    split.write_text("missing.png\tmissing.json\n")
    with pytest.raises(FileNotFoundError):
        LabelMeDataset(split, root=tmp_path)
    split.write_text("")
    with pytest.raises(ValueError, match="empty"):
        LabelMeDataset(split, root=tmp_path)


def test_geometric_translation_matches_image_pixels():
    data = dict(
        imageWidth=64,
        imageHeight=32,
        shapes=[
            dict(
                label="porosity",
                shape_type="polygon",
                points=[[10, 10], [18, 10], [18, 18], [10, 18]],
            )
        ],
    )
    original = Image.new("L", (64, 32))
    ImageDraw.Draw(original).polygon(
        [tuple(p) for p in data["shapes"][0]["points"]], fill=255
    )
    transformed = transform_annotation(data, 0, 3, -2)
    expected = Image.new("L", original.size)
    ImageDraw.Draw(expected).polygon(
        [tuple(p) for p in transformed["shapes"][0]["points"]], fill=255
    )
    np.testing.assert_array_equal(
        np.asarray(affine_image(original, 0, 3, -2)), np.asarray(expected)
    )
    assert transformed["imageData"] is None
    assert data["shapes"][0]["points"][0] == [10, 10]


def test_split_reproducibility_and_training_only_augmentation(synthetic):
    for name in ("split_a", "split_b"):
        run_script("split_dataset.py", "--root", synthetic, "--output-dir", name)
    buckets = []
    for split in ("train", "val", "test"):
        first = (synthetic / "split_a" / f"{split}.txt").read_text()
        assert first == (synthetic / "split_b" / f"{split}.txt").read_text()
        buckets.append(set(first.splitlines()))
    assert all(buckets)
    assert sum(map(len, buckets)) == len(set.union(*buckets)) == 10
    before = {
        name: (synthetic / "splits" / f"{name}.txt").read_bytes()
        for name in ("train", "val", "test")
    }
    run_script("augment_training.py", "--root", synthetic, "--target-size", 8)
    augmented = LabelMeDataset(synthetic / "splits/train_augmented.txt", root=synthetic)
    assert len(augmented) == 8
    for index in range(len(augmented)):
        augmented[index]
    assert all(
        before[name] == (synthetic / "splits" / f"{name}.txt").read_bytes()
        for name in before
    )
