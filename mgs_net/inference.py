"""Label-free image prediction and annotated-split evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data_validation import CLASS_NAMES, IMAGE_EXTENSIONS, validate_image_size
from .metrics import SegmentationMeter
from .runtime import load_config, load_trained_model, select_device
from .training import make_loader

COLORS = np.array(
    [(0, 0, 0), (230, 57, 70), (42, 157, 143), (69, 123, 157), (244, 162, 97)],
    dtype=np.uint8,
)


@torch.no_grad()
def evaluate(config, checkpoint, split="test", device=None, batch_size=None):
    device = select_device(device or config["train"].get("device"))
    model = load_trained_model(config, checkpoint, device)
    loader = make_loader(config, split, batch_size)
    meter = SegmentationMeter(5, ignore_index=0)
    for batch in loader:
        meter.update(
            model(batch["image"].to(device))["logits"], batch["mask"].to(device)
        )
    metrics = meter.compute()
    names = ("background",) + CLASS_NAMES
    metrics["iou_by_name"] = dict(zip(names, metrics["iou_per_class"]))
    metrics["dice_by_name"] = dict(zip(names, metrics["dice_per_class"]))
    return metrics


@torch.no_grad()
def predict_image(model, image_path, device):
    with Image.open(image_path) as source:
        original = source.convert("RGB")
        grayscale = source.convert("L")
    validate_image_size(grayscale.size)
    image = np.asarray(grayscale, dtype=np.float32)[None, None] / 255.0
    logits = model(torch.from_numpy(image).to(device))["logits"]
    mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return original, mask


def predict_main():
    parser = argparse.ArgumentParser(
        description="Predict external images; no annotation is needed."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        help="A checkpoint trained by you; pre-trained parameters are not distributed.",
    )
    parser.add_argument(
        "--input", required=True, help="One image or a flat image directory."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    device = select_device(args.device)
    model = load_trained_model(config, args.checkpoint, device)
    source, output = Path(args.input), Path(args.output_dir)
    if source.is_dir():
        images = sorted(
            p
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    elif source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        images = [source]
    else:
        raise FileNotFoundError("--input must be a supported image or image directory.")
    if not images:
        raise ValueError("The input directory contains no supported images.")
    destinations = [
        output / (p.name + suffix)
        for p in images
        for suffix in (".ids.png", ".color.png", ".overlay.png")
    ]
    if any(p.exists() for p in destinations):
        raise FileExistsError(
            "Prediction outputs already exist; choose a new output directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    for path in images:
        original, mask = predict_image(model, path, device)
        colored = Image.fromarray(COLORS[mask])
        Image.fromarray(mask).save(output / (path.name + ".ids.png"))
        colored.save(output / (path.name + ".color.png"))
        Image.blend(original, colored, 0.45).save(output / (path.name + ".overlay.png"))
    (output / "classes.json").write_text(
        json.dumps(
            {
                str(i): {"name": name, "rgb": COLORS[i].tolist()}
                for i, name in enumerate(("background",) + CLASS_NAMES)
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Predicted {len(images)} images; masks and overlays saved to {output}")


def evaluate_main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on your annotated split."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    config = load_config(args.config)
    path = (
        Path(args.output)
        if args.output
        else Path(config["experiment"]["output_dir"])
        / "metrics"
        / f"{args.split}_metrics.json"
    )
    if path.exists():
        raise FileExistsError("Evaluation output exists; choose a new --output file.")
    metrics = evaluate(
        config, args.checkpoint, args.split, args.device, args.batch_size
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metrics, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    print(payload)
