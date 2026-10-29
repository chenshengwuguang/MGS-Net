"""Convolution and normalization layers shared by the MGS-Net decoder."""

import torch.nn as nn


def choose_group_count(num_channels, max_groups=8):
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def build_norm_layer(norm, num_channels):
    if norm in (None, "none"):
        return None
    if norm == "group":
        return nn.GroupNorm(choose_group_count(num_channels), num_channels)
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    raise ValueError(f"Unsupported normalization: {norm}")


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, norm=None):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        norm1 = build_norm_layer(norm, out_channels)
        if norm1 is not None:
            layers.append(norm1)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        norm2 = build_norm_layer(norm, out_channels)
        if norm2 is not None:
            layers.append(norm2)
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
