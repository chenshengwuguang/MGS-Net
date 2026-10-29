"""Training with Adam, cosine scheduling and validation-Dice selection."""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import make_dataset
from .losses import build_loss, get_main_logits
from .metrics import SegmentationMeter
from .runtime import build_model, load_config, select_device, set_seed


def make_loader(config, split, batch_size=None):
    settings = config["train"]
    workers = 0 if os.name == "nt" else settings.get("num_workers", 0)
    options = dict(
        batch_size=batch_size or settings["batch_size"],
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=settings.get("pin_memory", False),
    )
    if workers:
        options.update(
            prefetch_factor=settings.get("prefetch_factor", 1),
            persistent_workers=settings.get("persistent_workers", True),
            multiprocessing_context=settings.get("multiprocessing_context", "spawn"),
        )
    return DataLoader(make_dataset(config, split), **options)


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    meter = SegmentationMeter(raw_model.classifier.out_channels, ignore_index=0)
    total_loss, seen = 0.0, 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        with torch.set_grad_enabled(training):
            output = model(images)
            loss = criterion(output, masks, batch["shape_mask"].to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Training produced a non-finite loss.")
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
        meter.update(get_main_logits(output).detach(), masks)
    result = meter.compute()
    result["loss"] = total_loss / max(seen, 1)
    return result


def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        {
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def train(config, smoke_test=False):
    settings = config["train"]
    set_seed(settings.get("seed", 42))
    device = select_device(settings.get("device"))
    model = build_model(config["model"]).to(device)
    device_ids = settings.get("device_ids", [])
    if device.type == "cuda" and len(device_ids) > 1:
        if max(device_ids) >= torch.cuda.device_count():
            raise ValueError(
                "Configured GPUs are unavailable. Set train.device_ids to [0] for one GPU."
            )
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    criterion = build_loss(config["loss"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["lr"],
        weight_decay=settings.get("weight_decay", 0.0),
    )
    schedule = settings["scheduler"]
    if schedule["name"] != "cosine":
        raise ValueError("The included training recipe uses the cosine scheduler.")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=schedule["t_max"], eta_min=schedule["eta_min"]
    )
    train_loader, val_loader = make_loader(config, "train"), make_loader(config, "val")
    output = Path(config["experiment"]["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Training output directory is not empty; choose a new --output-dir."
        )
    for name in ("checkpoints", "metrics", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    with (output / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    stop = settings["early_stopping"]
    best, best_loss, waiting = -1.0, float("inf"), 0
    fields = ("epoch", "lr", "train_loss", "val_loss", "val_mean_iou", "val_mean_dice")
    with (output / "metrics/val_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, (1 if smoke_test else settings["epochs"]) + 1):
            limit = 1 if smoke_test else None
            learning_rate = optimizer.param_groups[0]["lr"]
            training = run_epoch(
                model, train_loader, criterion, device, optimizer, limit
            )
            validation = run_epoch(
                model, val_loader, criterion, device, max_batches=limit
            )
            writer.writerow(
                dict(
                    epoch=epoch,
                    lr=learning_rate,
                    train_loss=training["loss"],
                    val_loss=validation["loss"],
                    val_mean_iou=validation["mean_iou"],
                    val_mean_dice=validation["mean_dice"],
                )
            )
            stream.flush()
            save_checkpoint(
                output / "checkpoints/last.pth",
                model,
                optimizer,
                epoch,
                validation,
                config,
            )
            best_loss = min(best_loss, validation["loss"])
            if validation["mean_dice"] > best + stop.get("min_delta", 0.0):
                best, waiting = validation["mean_dice"], 0
                save_checkpoint(
                    output / "checkpoints/best.pth",
                    model,
                    optimizer,
                    epoch,
                    validation,
                    config,
                )
            else:
                waiting += 1
            message = f"epoch={epoch} train_loss={training['loss']:.6f} val_loss={validation['loss']:.6f} val_dice={validation['mean_dice']:.6f}"
            print(message)
            with (output / "logs/train.log").open("a", encoding="utf-8") as log:
                log.write(message + "\n")
            scheduler.step()
            if waiting >= stop["patience"]:
                break
    (output / "metrics/best_val.json").write_text(
        json.dumps({"best_mean_dice": best, "best_val_loss": best_loss}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output / "checkpoints/best.pth"


def main():
    parser = argparse.ArgumentParser(description="Train MGS-Net using your own data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="One train batch and one validation batch; outputs are for a pipeline check.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    if args.device:
        config["train"]["device"] = args.device
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        config["train"]["batch_size"] = args.batch_size
    print(f"Saved checkpoint: {train(config, args.smoke_test)}")
