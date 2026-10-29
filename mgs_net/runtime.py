"""Configuration, device selection and strict parameter loading."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .data_validation import CLASS_NAMES
from .model import MGSNet


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or any(
        key not in config for key in ("experiment", "model", "data", "loss", "train")
    ):
        raise ValueError(
            "Configuration needs experiment, model, data, loss and train sections."
        )
    if tuple(config["data"].get("class_names", ())) != CLASS_NAMES:
        raise ValueError("The data.class_names order must match docs/data-format.md.")
    model = config["model"]
    if (
        model.get("name") != "mgs_net"
        or model.get("in_channels") != 1
        or model.get("num_classes") != 5
        or model.get("norm") != "group"
    ):
        raise ValueError(
            "Use model name mgs_net, one input channel, five classes and group normalization."
        )
    if config["loss"].get("num_classes") != 5:
        raise ValueError("loss.num_classes must be five.")
    if (
        int(config["train"].get("batch_size", 0)) < 1
        or int(config["train"].get("epochs", 0)) < 1
    ):
        raise ValueError("Training batch_size and epochs must be positive.")
    return config


def build_model(config):
    options = dict(config)
    if options.pop("name", "mgs_net") != "mgs_net":
        raise ValueError("Only the MGS-Net model is included.")
    return MGSNet(**options)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested=None, threads=2):
    torch.set_num_threads(threads)
    requested = requested or "auto"
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu or cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
    return torch.device(
        "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"
    )


def load_trained_model(config, checkpoint, device):
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError(
            "A trained checkpoint is required. Train with scripts/train.py and pass --checkpoint <your-best.pth>."
        )
    # Never fall back to unrestricted pickle loading.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved.get("model_state", saved)
    model = build_model(config["model"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
