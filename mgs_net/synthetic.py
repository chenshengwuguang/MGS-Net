"""Generate geometrical examples from scratch, independent of any real images."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data_validation import CLASS_NAMES


def create_synthetic_dataset(output, seed=42):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Synthetic output directory is not empty; choose a new --output-dir."
        )
    for name in ("images", "labels", "splits"):
        (output / name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    index = 0
    for split, count in (("train", 6), ("val", 2), ("test", 2)):
        rows = []
        for _ in range(count):
            stem = f"example_{index:03d}"
            pixels = np.clip(rng.normal(110, 8, (128, 256)), 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels)
            draw = ImageDraw.Draw(image)
            shapes = []
            if index % 3 != 1:
                shift = int(rng.integers(-3, 4))
                specifications = [
                    (
                        CLASS_NAMES[0],
                        "polygon",
                        [[20, 20], [195, 20], [195, 27], [20, 27]],
                    ),
                    (
                        CLASS_NAMES[1],
                        "polygon",
                        [[30, 47], [210, 47], [210, 54], [30, 54]],
                    ),
                    (CLASS_NAMES[2], "circle", [[45, 88], [49, 88]]),
                    (
                        CLASS_NAMES[3],
                        "polygon",
                        [[105, 75], [127, 75], [127, 97], [105, 97]],
                    ),
                ]
                for label, kind, points in specifications:
                    points = [[x + shift, y] for x, y in points]
                    shapes.append(
                        dict(label=label, points=points, shape_type=kind, flags={})
                    )
                    if kind == "circle":
                        x, y = points[0]
                        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=45)
                    else:
                        draw.polygon([tuple(point) for point in points], fill=45)
            image.save(output / "images" / f"{stem}.png")
            annotation = dict(
                synthetic=True,
                imagePath=f"../images/{stem}.png",
                imageData=None,
                imageWidth=256,
                imageHeight=128,
                shapes=shapes,
            )
            (output / "labels" / f"{stem}.json").write_text(
                json.dumps(annotation, indent=2) + "\n", encoding="utf-8"
            )
            rows.append(f"images/{stem}.png\tlabels/{stem}.json")
            index += 1
        (output / "splits" / f"{split}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic geometry for a pipeline demonstration."
    )
    parser.add_argument("--output-dir", default="generated/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        f"Synthetic examples written to {create_synthetic_dataset(args.output_dir, args.seed)}"
    )
