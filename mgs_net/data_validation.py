"""Validate the supported external-image LabelMe annotation format."""

import math

CLASS_NAMES = (
    "incomplete_penetration",
    "lack_of_fusion",
    "porosity",
    "slag_inclusion",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def validate_image_size(size):
    if min(size) < 16 or any(value % 16 for value in size):
        raise ValueError(
            "Image width and height must be multiples of 16 and at least 16; no automatic resizing is applied."
        )


def validate_annotation(data, image_size, label_to_id):
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
        raise ValueError(
            "An annotation must be an object with a shapes list (empty for background)."
        )
    for field, size in zip(("imageWidth", "imageHeight"), image_size):
        if data.get(field) != size:
            raise ValueError(
                f"Annotation {field} must match the paired image ({size})."
            )
    for shape in data["shapes"]:
        if (
            not isinstance(shape, dict)
            or shape.get("label", "").strip() not in label_to_id
        ):
            raise ValueError("Every shape must use a supported class label.")
        kind = shape.get("shape_type", "polygon") or "polygon"
        points = shape.get("points", [])
        if kind not in {"circle", "polygon"}:
            raise ValueError(f"Unsupported shape_type: {kind}; use circle or polygon.")
        if (kind == "circle" and len(points) != 2) or (
            kind == "polygon" and len(points) < 3
        ):
            raise ValueError(
                "Circles need two points (center and edge); polygons need at least three."
            )
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in points):
            raise ValueError("Each annotation point must contain x and y.")
        if any(
            not isinstance(v, (int, float)) or not math.isfinite(v)
            for p in points
            for v in p
        ):
            raise ValueError("Annotation coordinates must be finite numbers.")
        if kind == "circle" and points[0] == points[1]:
            raise ValueError("A circle must have a positive radius.")
