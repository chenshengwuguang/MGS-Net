"""Generic label-set splitting and offline balanced augmentation for user data."""

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .data import read_pairs
from .data_validation import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    validate_annotation,
    validate_image_size,
)


def read_annotation(path, size):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_annotation(data, size, {name: i + 1 for i, name in enumerate(CLASS_NAMES)})
    return data


def split_main():
    parser = argparse.ArgumentParser(
        description="Create deterministic, approximately label-set-stratified splits."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ratios = (args.train_ratio, args.val_ratio, 1 - args.train_ratio - args.val_ratio)
    if min(ratios) <= 0:
        parser.error("train, validation and remaining test ratios must all be positive")
    root = Path(args.root).resolve()
    images_dir, labels_dir, output = (
        root / args.images_dir,
        root / args.labels_dir,
        root / args.output_dir,
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Split output directory is not empty; choose a new --output-dir."
        )
    images = defaultdict(list)
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.stem].append(path)
    groups = defaultdict(list)
    for path in sorted(labels_dir.glob("*.json")):
        candidates = images.get(path.stem, [])
        if len(candidates) != 1:
            raise ValueError(
                "Each annotation needs exactly one image with the same stem."
            )
        with Image.open(candidates[0]) as image:
            validate_image_size(image.size)
            data = read_annotation(path, image.size)
        labels = tuple(sorted({s["label"].strip() for s in data["shapes"]}))
        row = f"{candidates[0].relative_to(root).as_posix()}\t{path.relative_to(root).as_posix()}"
        groups[labels].append(row)
    count = sum(map(len, groups.values()))
    if count < 3:
        raise ValueError(
            "At least three annotated images are needed for three nonempty splits."
        )
    # Reserve one sample per split, then distribute the remaining capacity.
    capacities = [1 + int((count - 3) * ratio) for ratio in ratios]
    for index in sorted(
        range(3), key=lambda i: (count - 3) * ratios[i] % 1, reverse=True
    )[: count - sum(capacities)]:
        capacities[index] += 1
    remaining, splits = capacities[:], [[], [], []]
    rng = random.Random(args.seed)
    for rows in sorted(groups.values(), key=len):
        rng.shuffle(rows)
        allocated = [0, 0, 0]
        for row in rows:
            candidates = [i for i in range(3) if remaining[i]]
            index = max(
                candidates,
                key=lambda i: (ratios[i] * len(rows) - allocated[i], remaining[i]),
            )
            splits[index].append(row)
            remaining[index] -= 1
            allocated[index] += 1
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(("train", "val", "test"), splits):
        (output / f"{name}.txt").write_text(
            "\n".join(sorted(rows)) + "\n", encoding="utf-8"
        )
    print(f"Wrote disjoint splits to {output}. Inspect class coverage before training.")


def transform_annotation(data, angle, dx, dy):
    """Rotate then translate x/y points; Pillow clips shapes at the image boundary."""
    data = copy.deepcopy(data)
    cx, cy = data["imageWidth"] / 2, data["imageHeight"] / 2
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    for shape in data["shapes"]:
        shape["points"] = [
            [
                cosine * (x - cx) - sine * (y - cy) + cx + dx,
                sine * (x - cx) + cosine * (y - cy) + cy + dy,
            ]
            for x, y in shape["points"]
        ]
    data["imageData"] = None
    return data


def affine_image(image, angle, dx, dy):
    cx, cy = image.width / 2, image.height / 2
    radians = math.radians(angle)
    c, s = math.cos(radians), math.sin(radians)
    inverse = (
        c,
        s,
        cx - c * (cx + dx) - s * (cy + dy),
        -s,
        c,
        cy + s * (cx + dx) - c * (cy + dy),
    )
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        inverse,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )


def photometric_image(image, rng):
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.08))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.92, 1.08))
    pixels = (np.asarray(image, dtype=np.float32) / 255.0) ** rng.uniform(0.92, 1.08)
    pixels = np.clip(pixels * 255, 0, 255)
    if rng.random() < 0.35:
        sigma = abs(rng.normalvariate(0.0, rng.uniform(1.0, 3.0)))
        pixels += np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(
            0, sigma, pixels.shape
        )
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    choice = rng.random()
    if choice < 0.18:
        return image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.6)))
    if choice < 0.36:
        return image.filter(
            ImageFilter.UnsharpMask(
                radius=1, percent=rng.randint(110, 150), threshold=3
            )
        )
    return image


def pixel_counts(data):
    mask = Image.new("L", (data["imageWidth"], data["imageHeight"]), 0)
    draw = ImageDraw.Draw(mask)
    for shape in data["shapes"]:
        points = [tuple(point) for point in shape["points"]]
        value = CLASS_NAMES.index(shape["label"].strip()) + 1
        if shape.get("shape_type", "polygon") == "circle":
            (x, y), (ex, ey) = points
            radius = math.hypot(ex - x, ey - y)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)
        else:
            draw.polygon(points, fill=value)
    mask = np.asarray(mask)
    return {label: int((mask == i + 1).sum()) for i, label in enumerate(CLASS_NAMES)}


def augment_main():
    parser = argparse.ArgumentParser(
        description="Expand only the training split with balanced offline augmentation."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--train-split", default="splits/train.txt")
    parser.add_argument("--output-dir", default="augmented")
    parser.add_argument("--output-split", default="splits/train_augmented.txt")
    parser.add_argument("--target-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    samples = read_pairs(root / args.train_split, root)
    output, split = (root / args.output_dir).resolve(), (
        root / args.output_split
    ).resolve()
    if not output.is_relative_to(root) or not split.is_relative_to(root):
        raise ValueError("Augmentation output paths must stay inside --root.")
    if args.target_size <= len(samples):
        raise ValueError("--target-size must exceed the current training size.")
    if split.exists() or (output.exists() and any(output.iterdir())):
        raise FileExistsError("Augmentation output exists; choose new output paths.")
    metadata, totals = [], Counter()
    for sample in samples:
        with Image.open(sample["image"]) as image:
            data = read_annotation(sample["json"], image.size)
        counts = pixel_counts(data)
        totals.update(counts)
        metadata.append((sample, data, counts))
    available = [label for label in CLASS_NAMES if totals[label] > 0]
    rng = random.Random(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{s['image'].relative_to(root).as_posix()}\t{s['json'].relative_to(root).as_posix()}"
        for s in samples
    ]
    for index in range(args.target_size - len(samples)):
        label = min(available, key=lambda value: totals[value]) if available else None
        choices = [entry for entry in metadata if label is None or entry[2][label] > 0]
        weights = [max(entry[2][label], 1) if label else 1 for entry in choices]
        sample, data, _ = rng.choices(choices, weights=weights, k=1)[0]
        angle, dx, dy = rng.uniform(-5, 5), rng.uniform(-8, 8), rng.uniform(-8, 8)
        with Image.open(sample["image"]) as source:
            image = photometric_image(
                affine_image(source.convert("L"), angle, dx, dy), rng
            )
        data = transform_annotation(data, angle, dx, dy)
        stem = f"augmented_{index:06d}"
        image_path, annotation_path = output / f"{stem}.png", output / f"{stem}.json"
        image.save(image_path)
        data["imagePath"] = image_path.name
        annotation_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        rows.append(
            f"{image_path.relative_to(root).as_posix()}\t{annotation_path.relative_to(root).as_posix()}"
        )
        totals.update(pixel_counts(data))
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        f"Augmented training split written to {split}; validation and test files are unchanged."
    )
