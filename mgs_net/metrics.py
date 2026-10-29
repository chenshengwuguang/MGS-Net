"""Accumulate pixels over a split; report per-class IoU and Dice.

Means exclude background and classes with no ground-truth pixels in the split.
Background pixels still contribute false positives to foreground class statistics.
"""

import torch


class SegmentationMeter:

    def __init__(self, num_classes, ignore_index=0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.intersections = torch.zeros(num_classes, dtype=torch.float64)
        self.unions = torch.zeros(num_classes, dtype=torch.float64)
        self.pred_areas = torch.zeros(num_classes, dtype=torch.float64)
        self.target_areas = torch.zeros(num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, targets):
        preds = torch.argmax(logits, dim=1)
        valid = (targets >= 0) & (targets < self.num_classes)
        encoded = targets[valid].to(torch.int64) * self.num_classes + preds[valid].to(
            torch.int64
        )
        confusion = torch.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        confusion = confusion.to(dtype=torch.float64, device="cpu")
        intersections = confusion.diag()
        target_areas = confusion.sum(dim=1)
        pred_areas = confusion.sum(dim=0)
        self.intersections += intersections
        self.unions += target_areas + pred_areas - intersections
        self.pred_areas += pred_areas
        self.target_areas += target_areas

    def compute(self):
        eps = 1e-07
        iou = self.intersections / (self.unions + eps)
        dice = 2 * self.intersections / (self.pred_areas + self.target_areas + eps)
        valid_classes = [
            i
            for i in range(self.num_classes)
            if i != self.ignore_index and self.target_areas[i] > 0
        ]
        if valid_classes:
            mean_iou = iou[valid_classes].mean().item()
            mean_dice = dice[valid_classes].mean().item()
        else:
            mean_iou = 0.0
            mean_dice = 0.0
        return {
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "iou_per_class": iou.tolist(),
            "dice_per_class": dice.tolist(),
        }
