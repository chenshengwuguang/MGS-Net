# Bring your own data

Data and pretrained weights are not distributed. The examples below are invented
file names; use your own paired images and annotations.

```text
data/user/
  images/sample_a.png
  labels/sample_a.json
  splits/train.txt
  splits/val.txt
  splits/test.txt
```

Set `data.root: data/user`. Relative roots are resolved from the current working
directory. Configured split paths and paths in each split are relative to that root.
Each nonempty line contains an image path, a literal TAB and its JSON path:

```text
images/sample_a.png<TAB>labels/sample_a.json
```

Replace `<TAB>` with a tab character. An optional third tab-separated field of
comma-separated labels is accepted for compatibility; mask labels come from JSON.
Train, validation and test must each be nonempty, and splits should be disjoint.

## Images and annotations

Pixels are read from the external image file, converted to grayscale and divided by
255. Supported file types include PNG, JPEG, BMP and TIFF. Use 8-bit images matching
the documented grayscale convention. Inputs are not resized; width and height must
be multiples of 16 and at least 16. A typical input is 256 pixels wide and 128 high.
Images in the same training batch must have matching dimensions.

An annotation is a LabelMe-compatible JSON object:

```json
{
  "imagePath": "../images/sample_a.png",
  "imageData": null,
  "imageWidth": 256,
  "imageHeight": 128,
  "shapes": [
    {
      "label": "porosity",
      "shape_type": "circle",
      "points": [[40, 40], [44, 40]],
      "flags": {}
    },
    {
      "label": "slag_inclusion",
      "shape_type": "polygon",
      "points": [[90, 60], [110, 60], [110, 80], [90, 80]],
      "flags": {}
    }
  ]
}
```

A circle has a center point and one edge point. A polygon has at least three x/y
points. Rectangles, lines and unmapped labels must be converted explicitly before
loading; the loader raises errors rather than silently treating them as background.
An image without defects has `"shapes": []`. Image dimensions in JSON must match
the paired file. Embedded `imageData` is unnecessary and is not used by the loader.

The class order is fixed:

| ID | Label | 中文 |
|---|---|---|
| 0 | background | 背景 |
| 1 | incomplete_penetration | 未焊透 |
| 2 | lack_of_fusion | 未熔合 |
| 3 | porosity | 气孔 |
| 4 | slag_inclusion | 夹渣 |

Shape supervision is derived automatically; see [Method](method.md).

## Generic preparation helpers

With matching file stems in separate `images` and `labels` directories, the generic
splitter can create deterministic, approximately label-set-stratified splits:

```bash
python scripts/split_dataset.py --root data/user
```

The splitter requires at least three samples, reserves one per split and apportions
the rest by the requested ratios. Exact class coverage is not guaranteed. Inspect
the generated splits. It does not identify related source images: if several crops
come from one source, prepare group-separated splits yourself before augmentation.

Only augment the training split. For example, to expand a small training split to
a user-selected size:

```bash
python scripts/augment_training.py --root data/user --target-size 100
```

Choose a target greater than your current training size and update `data.train_split`
to `splits/train_augmented.txt`. The helper samples a currently least-represented
available class by foreground pixel totals, applies rotation within ±5 degrees,
translation within ±8 pixels, brightness/contrast/gamma factors within 0.92–1.08,
and occasional light noise, blur or sharpening. It supports background-only samples
and datasets missing a class. Geometry moves both circle points without clamping
them separately, preserving radius; rasterization clips shapes at the image boundary.
Validation and test split files are not changed. This generic tool does not reconstruct
any original training data or original split.

