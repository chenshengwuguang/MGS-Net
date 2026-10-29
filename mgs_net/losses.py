"""Classification, overlap, boundary, auxiliary and morphology training losses.

Direction references are uniformly sampled foreground pixels in row-major order,
not object centers. The soft direction penalty does not guarantee star convexity.
The horizontal penalty promotes continuity and discourages large vertical changes.
See docs/method.md for the reductions, weights and supervision rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_main_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs


class DiceLoss(nn.Module):

    def __init__(self, num_classes, include_background=False, smooth=1e-06):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = get_main_logits(logits)
        probs = torch.softmax(logits, dim=1)
        one_hot = (
            F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        )
        if not self.include_background:
            probs = probs[:, 1:]
            one_hot = one_hot[:, 1:]
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        cardinality = torch.sum(probs + one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        logits = get_main_logits(logits)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - target_probs).pow(self.gamma)
        loss = -focal_factor * target_log_probs
        if self.alpha is not None:
            alpha_factor = self.alpha.gather(0, targets.reshape(-1)).reshape_as(targets)
            loss = loss * alpha_factor
        return loss.mean()


class BoundaryBandLoss(nn.Module):

    def __init__(self, num_classes, radius=2, include_background=False, smooth=1e-06):
        super().__init__()
        self.num_classes = num_classes
        self.radius = radius
        self.include_background = include_background
        self.smooth = smooth

    def _dilate(self, x, radius):
        kernel_size = 2 * radius + 1
        return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=radius)

    def _erode(self, x, radius):
        return -self._dilate(-x, radius)

    def _boundary(self, x):
        dilated = self._dilate(x, 1)
        eroded = self._erode(x, 1)
        return torch.clamp(dilated - eroded, min=0.0, max=1.0)

    def forward(self, logits, targets):
        logits = get_main_logits(logits)
        probs = torch.softmax(logits, dim=1)
        target = (
            F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        )
        if not self.include_background:
            probs = probs[:, 1:]
            target = target[:, 1:]
        pred_boundary = self._boundary(probs)
        target_boundary = self._boundary(target)
        pred_band = self._dilate(pred_boundary, self.radius)
        target_band = self._dilate(target_boundary, self.radius)
        dims = (0, 2, 3)
        precision = torch.sum(pred_boundary * target_band, dim=dims) / (
            torch.sum(pred_boundary, dim=dims) + self.smooth
        )
        recall = torch.sum(target_boundary * pred_band, dim=dims) / (
            torch.sum(target_boundary, dim=dims) + self.smooth
        )
        boundary_score = (2.0 * precision * recall + self.smooth) / (
            precision + recall + self.smooth
        )
        valid = torch.sum(target_boundary, dim=dims) > 0
        if valid.any():
            return 1.0 - boundary_score[valid].mean()
        return logits.new_tensor(0.0)


class ConvexStarShapePriorLoss(nn.Module):

    def __init__(
        self,
        num_classes,
        class_weights=None,
        max_centers=16,
        temperature=0.08,
        downsample=2,
        min_pixels=8,
        smooth=1e-06,
    ):
        super().__init__()
        self.num_classes = num_classes
        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None
        self.max_centers = max_centers
        self.temperature = temperature
        self.downsample = downsample
        self.min_pixels = min_pixels
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = get_main_logits(logits)
        probs = torch.softmax(logits, dim=1)
        if self.downsample > 1:
            probs = F.avg_pool2d(
                probs, kernel_size=self.downsample, stride=self.downsample
            )
            targets = (
                F.interpolate(
                    targets.unsqueeze(1).float(), size=probs.shape[-2:], mode="nearest"
                )
                .squeeze(1)
                .long()
            )
        height, width = probs.shape[-2:]
        coords = self._make_coords(height, width, probs.device, probs.dtype)
        losses = []
        for batch_index in range(probs.shape[0]):
            for class_index in range(1, self.num_classes):
                class_mask = targets[batch_index] == class_index
                if int(class_mask.sum().item()) < self.min_pixels:
                    continue
                class_coords = coords[class_mask.reshape(-1)]
                centers = self._sample_centers(class_coords)
                direction = self._direction_field(coords, centers, height, width)
                grad_y, grad_x = self._probability_gradient(
                    probs[batch_index, class_index]
                )
                gradient = torch.stack([grad_y, grad_x], dim=-1)
                valid = class_mask & (
                    torch.linalg.vector_norm(direction, dim=-1) > self.smooth
                )
                if not valid.any():
                    continue
                cosine = F.cosine_similarity(
                    gradient[valid], direction[valid], dim=-1, eps=self.smooth
                )
                loss = F.relu(-cosine).mean()
                if self.class_weights is not None:
                    loss = loss * self.class_weights[class_index]
                losses.append(loss)
        if not losses:
            return logits.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _make_coords(self, height, width, device, dtype):
        y_coords = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return torch.stack([yy, xx], dim=-1).reshape(-1, 2)

    def _sample_centers(self, class_coords):
        count = class_coords.shape[0]
        if count <= self.max_centers:
            return class_coords
        indices = torch.linspace(
            0, count - 1, self.max_centers, device=class_coords.device
        ).long()
        return class_coords[indices]

    def _direction_field(self, coords, centers, height, width):
        distances = torch.cdist(coords, centers).pow(2)
        weights = torch.softmax(-distances / max(self.temperature, self.smooth), dim=1)
        smooth_centers = weights @ centers
        return (smooth_centers - coords).reshape(height, width, 2)

    def _probability_gradient(self, probability):
        grad_y = torch.zeros_like(probability)
        grad_x = torch.zeros_like(probability)
        grad_y[1:-1] = 0.5 * (probability[2:] - probability[:-2])
        grad_x[:, 1:-1] = 0.5 * (probability[:, 2:] - probability[:, :-2])
        grad_y[0] = probability[1] - probability[0]
        grad_y[-1] = probability[-1] - probability[-2]
        grad_x[:, 0] = probability[:, 1] - probability[:, 0]
        grad_x[:, -1] = probability[:, -1] - probability[:, -2]
        return (grad_y, grad_x)


class HorizontalElongationPriorLoss(nn.Module):

    def __init__(self, class_weights=None, downsample=2, min_pixels=8, smooth=1e-06):
        super().__init__()
        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None
        self.downsample = downsample
        self.min_pixels = min_pixels
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = get_main_logits(logits)
        probs = torch.softmax(logits, dim=1)
        if self.downsample > 1:
            probs = F.avg_pool2d(
                probs, kernel_size=self.downsample, stride=self.downsample
            )
            targets = (
                F.interpolate(
                    targets.unsqueeze(1).float(), size=probs.shape[-2:], mode="nearest"
                )
                .squeeze(1)
                .long()
            )
        losses = []
        for class_index, weight in self._iter_class_weights(probs.device, probs.dtype):
            for batch_index in range(probs.shape[0]):
                class_mask = targets[batch_index] == class_index
                if int(class_mask.sum().item()) < self.min_pixels:
                    continue
                probability = probs[batch_index, class_index]
                horizontal_diff = torch.abs(probability[:, 1:] - probability[:, :-1])
                horizontal_mask = class_mask[:, 1:] & class_mask[:, :-1]
                vertical_diff = torch.abs(probability[1:, :] - probability[:-1, :])
                vertical_mask = class_mask[1:, :] & class_mask[:-1, :]
                class_losses = []
                if horizontal_mask.any():
                    class_losses.append(horizontal_diff[horizontal_mask].mean())
                if vertical_mask.any():
                    class_losses.append(
                        0.5 * F.relu(vertical_diff[vertical_mask] - 0.05).mean()
                    )
                if class_losses:
                    losses.append(weight * torch.stack(class_losses).sum())
        if not losses:
            return logits.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _iter_class_weights(self, device, dtype):
        if self.class_weights is None:
            return []
        return [
            (index, self.class_weights[index].to(device=device, dtype=dtype))
            for index in range(1, len(self.class_weights))
            if float(self.class_weights[index].item()) > 0
        ]


class FocalDiceBoundaryLoss(nn.Module):

    def __init__(
        self,
        num_classes,
        focal_weight=0.5,
        dice_weight=0.3,
        boundary_weight=0.2,
        gamma=2.0,
        alpha=None,
        include_background_in_dice=False,
        include_background_in_boundary=False,
        boundary_radius=2,
        aux_shape_weight=0.0,
        star_shape_weight=0.0,
        star_shape_class_weights=None,
        star_shape_max_centers=16,
        star_shape_temperature=0.08,
        star_shape_downsample=2,
        horizontal_shape_weight=0.0,
        horizontal_shape_class_weights=None,
        horizontal_shape_downsample=2,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice = DiceLoss(
            num_classes=num_classes, include_background=include_background_in_dice
        )
        self.boundary = BoundaryBandLoss(
            num_classes=num_classes,
            radius=boundary_radius,
            include_background=include_background_in_boundary,
        )
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.aux_shape_weight = aux_shape_weight
        self.star_shape_weight = star_shape_weight
        self.star_shape = ConvexStarShapePriorLoss(
            num_classes=num_classes,
            class_weights=star_shape_class_weights,
            max_centers=star_shape_max_centers,
            temperature=star_shape_temperature,
            downsample=star_shape_downsample,
        )
        self.horizontal_shape_weight = horizontal_shape_weight
        self.horizontal_shape = HorizontalElongationPriorLoss(
            class_weights=horizontal_shape_class_weights,
            downsample=horizontal_shape_downsample,
        )

    def forward(self, logits, targets, shape_targets=None):
        main_logits = get_main_logits(logits)
        loss = (
            self.focal_weight * self.focal(main_logits, targets)
            + self.dice_weight * self.dice(main_logits, targets)
            + self.boundary_weight * self.boundary(main_logits, targets)
        )
        if (
            self.aux_shape_weight > 0
            and isinstance(logits, dict)
            and (shape_targets is not None)
        ):
            loss = loss + self.aux_shape_weight * self._aux_shape_loss(
                logits, shape_targets
            )
        if self.star_shape_weight > 0:
            loss = loss + self.star_shape_weight * self.star_shape(main_logits, targets)
        if self.horizontal_shape_weight > 0:
            loss = loss + self.horizontal_shape_weight * self.horizontal_shape(
                main_logits, targets
            )
        return loss

    def _aux_shape_loss(self, outputs, shape_targets):
        branch_targets = {
            "small_logits": (shape_targets == 1).long(),
            "large_logits": (shape_targets == 2).long(),
            "elongated_logits": (shape_targets == 3).long(),
        }
        losses = []
        for key, target in branch_targets.items():
            if key in outputs:
                losses.append(
                    F.cross_entropy(outputs[key], target)
                    + self._binary_dice_loss(outputs[key], target)
                )
        if not losses:
            return torch.zeros((), device=shape_targets.device, dtype=torch.float32)
        return torch.stack(losses).mean()

    def _binary_dice_loss(self, logits, target, smooth=1e-06):
        positive_prob = torch.softmax(logits, dim=1)[:, 1]
        target = target.float()
        dims = (0, 1, 2)
        intersection = torch.sum(positive_prob * target, dim=dims)
        cardinality = torch.sum(positive_prob + target, dim=dims)
        return 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)


def build_loss(config):
    from inspect import signature

    settings = dict(config)
    if settings.pop("name", "focal_dice_boundary") != "focal_dice_boundary":
        raise ValueError("MGS-Net supports the focal_dice_boundary training objective.")
    unknown = set(settings) - set(signature(FocalDiceBoundaryLoss).parameters)
    if unknown:
        raise ValueError(f"Unsupported loss options: {sorted(unknown)}")
    return FocalDiceBoundaryLoss(**settings)
