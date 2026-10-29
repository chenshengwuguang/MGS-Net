"""Load external grayscale images and derive semantic and morphology masks."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ._mask_geometry import LabelMeDataset as MaskGeometry
from .data_validation import CLASS_NAMES, validate_image_size


def read_pairs(split_file, root):
    root = Path(root).resolve()
    split_file = Path(split_file)
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. Supply your own data or generate the synthetic example."
        )
    pairs = []
    for number, raw in enumerate(
        split_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        parts = raw.strip().split("\t")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Split line {number}: expected image path, TAB, annotation path and optional TAB-separated labels."
            )
        image, annotation = (root / part for part in parts[:2])
        if not image.is_file() or not annotation.is_file():
            raise FileNotFoundError(
                f"Split line {number}: paired image or annotation does not exist."
            )
        pairs.append({"image": image, "json": annotation})
    if not pairs:
        raise ValueError("The split is empty; at least one paired sample is required.")
    return pairs


class LabelMeDataset(MaskGeometry, Dataset):
    """Five semantic classes and four annotation-derived morphology IDs.

    Paths inside a split are relative to root. No annotation contains an image
    needed by the loader: pixels are always read from the paired external file.
    """

    def __init__(
        self,
        split_file,
        root=".",
        class_names=CLASS_NAMES,
        image_mode="L",
        shape_line_aspect_ratio=3.0,
        shape_small_area_threshold=128.0,
    ):
        if tuple(class_names) != CLASS_NAMES or image_mode != "L":
            raise ValueError(
                "Use grayscale mode L and the four class names in their documented order."
            )
        self.root = Path(root).resolve()
        self.samples = read_pairs(split_file, self.root)
        self.class_names = list(class_names)
        self.label_to_id = {name: i + 1 for i, name in enumerate(class_names)}
        self.image_mode = image_mode
        self.shape_line_aspect_ratio = shape_line_aspect_ratio
        self.shape_small_area_threshold = shape_small_area_threshold

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample["image"]) as source:
            image = source.convert("L")
        validate_image_size(image.size)
        array = np.asarray(image, dtype=np.float32)[None] / 255.0
        mask, shape_mask = self._load_masks(sample["json"], image.size)
        return {
            "image": torch.from_numpy(array),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "shape_mask": torch.from_numpy(shape_mask.astype(np.int64)),
            "image_path": str(sample["image"]),
            "json_path": str(sample["json"]),
        }


def make_dataset(config, split):
    data = config["data"]
    root = Path(data["root"]).resolve()
    options = {
        key: data[key]
        for key in (
            "class_names",
            "image_mode",
            "shape_line_aspect_ratio",
            "shape_small_area_threshold",
        )
        if key in data
    }
    return LabelMeDataset(root / data[f"{split}_split"], root=root, **options)
