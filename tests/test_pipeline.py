import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_cli_pipeline_in_separate_directory(tmp_path):
    def command(script, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )

    command("generate_synthetic_data.py")
    config = tmp_path / "demo.yaml"
    config.write_text(
        (ROOT / "configs/synthetic_cpu.yaml").read_text(), encoding="utf-8"
    )
    command("train.py", "--config", config, "--smoke-test")
    checkpoint = tmp_path / "runs/synthetic/checkpoints/best.pth"
    assert checkpoint.is_file()
    command(
        "evaluate.py", "--config", config, "--checkpoint", checkpoint, "--device", "cpu"
    )
    assert (tmp_path / "runs/synthetic/metrics/test_metrics.json").is_file()
    # Delete an annotation before predicting: the prediction CLI must not need it.
    (tmp_path / "generated/synthetic/labels/example_000.json").unlink()
    command(
        "predict.py",
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--input",
        "generated/synthetic/images/example_000.png",
        "--output-dir",
        "runs/predictions",
        "--device",
        "cpu",
    )
    with Image.open(tmp_path / "runs/predictions/example_000.png.ids.png") as image:
        mask = np.asarray(image)
        assert mask.shape == (128, 256)
        assert mask.min() >= 0 and mask.max() <= 4
    assert (tmp_path / "runs/predictions/example_000.png.color.png").is_file()
    assert (tmp_path / "runs/predictions/example_000.png.overlay.png").is_file()
    classes = json.loads((tmp_path / "runs/predictions/classes.json").read_text())
    assert classes["0"]["name"] == "background"
