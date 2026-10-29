"""Pillow rasterization and annotation-derived shape geometry."""

import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .data_validation import validate_annotation


class LabelMeDataset:

    def _load_mask(self, json_path, image_size):
        mask, _ = self._load_masks(json_path, image_size)
        return mask

    def _load_masks(self, json_path, image_size):
        width, height = image_size
        mask = Image.new("L", (width, height), 0)
        shape_mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        shape_draw = ImageDraw.Draw(shape_mask)
        with Path(json_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        validate_annotation(data, image_size, self.label_to_id)
        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            class_id = self.label_to_id.get(label)
            points = shape.get("points", [])
            shape_type = shape.get("shape_type", "polygon") or "polygon"
            if class_id is None:
                continue
            xy = [(float(x), float(y)) for x, y in points]
            if shape_type == "circle" and len(xy) >= 2:
                bbox = self._circle_bbox(xy)
                draw.ellipse(bbox, fill=class_id)
                shape_draw.ellipse(bbox, fill=self._shape_id(label, xy, shape_type))
            elif len(xy) >= 3:
                draw.polygon(xy, fill=class_id)
                shape_draw.polygon(xy, fill=self._shape_id(label, xy, shape_type))
        return (
            np.asarray(mask, dtype=np.uint8),
            np.asarray(shape_mask, dtype=np.uint8),
        )

    def _shape_id(self, label, points, shape_type="polygon"):
        if label == "incomplete_penetration":
            return 3
        if label == "lack_of_fusion":
            return 3
        if label in ("porosity", "slag_inclusion"):
            aspect = 1.0 if shape_type == "circle" else self._bbox_aspect(points)
            if label == "slag_inclusion" and aspect >= self.shape_line_aspect_ratio:
                return 3
            area = (
                self._circle_area(points)
                if shape_type == "circle"
                else self._polygon_area(points)
            )
            return 1 if area < self.shape_small_area_threshold else 2
        return 0

    def _circle_bbox(self, points):
        center_x, center_y = points[0]
        edge_x, edge_y = points[1]
        radius = math.hypot(edge_x - center_x, edge_y - center_y)
        return (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )

    def _circle_area(self, points):
        center_x, center_y = points[0]
        edge_x, edge_y = points[1]
        radius = math.hypot(edge_x - center_x, edge_y - center_y)
        return math.pi * radius * radius

    def _bbox_aspect(self, points):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return max(width, height) / max(min(width, height), 1.0)

    def _polygon_area(self, points):
        area = 0.0
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            area += point[0] * next_point[1] - next_point[0] * point[1]
        return abs(area) * 0.5
