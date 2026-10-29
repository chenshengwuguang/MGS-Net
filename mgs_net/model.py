"""Morphology-Guided Segmentation Network: three branches and spatial fusion.

Parameter module names are stable for strict loading of compatible state dictionaries.
The point and area branches receive enhanced inputs; the line branch receives the image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DoubleConv, build_norm_layer


class ConvNormAct(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, norm=None
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.norm = build_norm_layer(norm, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x)


class PositionContextAttention(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.attention(x)


class PixelPreservingDownsample(nn.Module):

    def __init__(self, in_channels, out_channels, norm=None):
        super().__init__()
        self.block = nn.Sequential(
            nn.PixelUnshuffle(2),
            ConvNormAct(
                in_channels * 4, out_channels, kernel_size=1, padding=0, norm=norm
            ),
        )

    def forward(self, x):
        return self.block(x)


class SmallDefectBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm=None):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, padding=1, norm=norm),
            ConvNormAct(
                out_channels, out_channels, kernel_size=3, padding=1, norm=norm
            ),
        )

    def forward(self, x):
        return self.block(x)


class LargeDefectBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm=None):
        super().__init__()
        self.local = ConvNormAct(
            in_channels, out_channels, kernel_size=5, padding=2, norm=norm
        )
        self.context = ConvNormAct(
            in_channels, out_channels, kernel_size=3, padding=2, dilation=2, norm=norm
        )
        self.project = ConvNormAct(
            out_channels * 2, out_channels, kernel_size=1, padding=0, norm=norm
        )

    def forward(self, x):
        return self.project(torch.cat([self.local(x), self.context(x)], dim=1))


class ElongatedDefectBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm=None):
        super().__init__()
        self.paths = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(
                        in_channels,
                        out_channels,
                        kernel_size=(1, 9),
                        padding=(0, 4),
                        norm=norm,
                    ),
                    ConvNormAct(
                        out_channels,
                        out_channels,
                        kernel_size=(1, 5),
                        padding=(0, 4),
                        dilation=(1, 2),
                        norm=norm,
                    ),
                ),
                nn.Sequential(
                    ConvNormAct(
                        in_channels,
                        out_channels,
                        kernel_size=(9, 1),
                        padding=(4, 0),
                        norm=norm,
                    ),
                    ConvNormAct(
                        out_channels,
                        out_channels,
                        kernel_size=(5, 1),
                        padding=(4, 0),
                        dilation=(2, 1),
                        norm=norm,
                    ),
                ),
            ]
        )
        self.project = ConvNormAct(
            out_channels * len(self.paths),
            out_channels,
            kernel_size=1,
            padding=0,
            norm=norm,
        )

    def forward(self, x):
        return self.project(torch.cat([path(x) for path in self.paths], dim=1))


class GatedBranchFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv2d(channels * 3, 3, kernel_size=1)
        self.refine = ConvNormAct(
            channels, channels, kernel_size=3, padding=1, norm="group"
        )

    def forward(self, small, large, elongated):
        reference_size = small.shape[-2:]
        if large.shape[-2:] != reference_size:
            large = F.interpolate(
                large, size=reference_size, mode="bilinear", align_corners=False
            )
        if elongated.shape[-2:] != reference_size:
            elongated = F.interpolate(
                elongated, size=reference_size, mode="bilinear", align_corners=False
            )
        gates = torch.softmax(
            self.gate(torch.cat([small, large, elongated], dim=1)), dim=1
        )
        fused = (
            gates[:, 0:1] * small + gates[:, 1:2] * large + gates[:, 2:3] * elongated
        )
        return (self.refine(fused), gates)


class MGSNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        num_classes=5,
        base_channels=32,
        norm="group",
        aux_classes=2,
    ):
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]
        self.position_attention = PositionContextAttention(in_channels)
        self.small_blocks = nn.ModuleList(
            [
                SmallDefectBlock(in_channels, channels[0], norm=norm),
                SmallDefectBlock(channels[1], channels[1], norm=norm),
                SmallDefectBlock(channels[2], channels[2], norm=norm),
                SmallDefectBlock(channels[3], channels[3], norm=norm),
                SmallDefectBlock(channels[4], channels[4], norm=norm),
            ]
        )
        self.large_blocks = nn.ModuleList(
            [
                LargeDefectBlock(in_channels, channels[0], norm=norm),
                LargeDefectBlock(channels[1], channels[1], norm=norm),
                LargeDefectBlock(channels[2], channels[2], norm=norm),
                LargeDefectBlock(channels[3], channels[3], norm=norm),
                LargeDefectBlock(channels[4], channels[4], norm=norm),
            ]
        )
        self.elongated_blocks = nn.ModuleList(
            [
                ElongatedDefectBlock(in_channels, channels[0], norm=norm),
                ElongatedDefectBlock(channels[1], channels[1], norm=norm),
                ElongatedDefectBlock(channels[2], channels[2], norm=norm),
                ElongatedDefectBlock(channels[3], channels[3], norm=norm),
                ElongatedDefectBlock(channels[4], channels[4], norm=norm),
            ]
        )
        self.fusions = nn.ModuleList([GatedBranchFusion(channels[i]) for i in range(5)])
        self.small_down = nn.ModuleList(
            [
                PixelPreservingDownsample(channels[i], channels[i + 1], norm=norm)
                for i in range(4)
            ]
        )
        self.large_down = nn.ModuleList(
            [
                PixelPreservingDownsample(channels[i], channels[i + 1], norm=norm)
                for i in range(4)
            ]
        )
        self.elongated_down = nn.ModuleList(
            [
                PixelPreservingDownsample(channels[i], channels[i + 1], norm=norm)
                for i in range(4)
            ]
        )
        self.small_aux = nn.Conv2d(channels[4], aux_classes, kernel_size=1)
        self.large_aux = nn.Conv2d(channels[4], aux_classes, kernel_size=1)
        self.elongated_aux = nn.Conv2d(channels[4], aux_classes, kernel_size=1)
        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(channels[4], channels[3], norm=norm)
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(channels[3], channels[2], norm=norm)
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(channels[2], channels[1], norm=norm)
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(channels[1], channels[0], norm=norm)
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        if (
            x.ndim != 4
            or not x.is_floating_point()
            or x.shape[1] != self.small_blocks[0].block[0].conv.in_channels
        ):
            raise ValueError("Expected a floating-point image tensor shaped N,C,H,W.")
        if (
            x.shape[0] < 1
            or min(x.shape[-2:]) < 16
            or any((size % 16 for size in x.shape[-2:]))
        ):
            raise ValueError(
                "Image height and width must be positive multiples of 16 (at least 16)."
            )
        input_size = x.shape[-2:]
        position_map = self.position_attention(x)
        small = x * (1.0 + position_map)
        large = x * (1.0 + position_map)
        elongated = x
        elongated_position_map = torch.zeros_like(x)
        fused_features = []
        gate_maps = []
        for level in range(5):
            small = self.small_blocks[level](small)
            large = self.large_blocks[level](large)
            elongated = self.elongated_blocks[level](elongated)
            fused, gates = self.fusions[level](small, large, elongated)
            fused_features.append(fused)
            gate_maps.append(gates)
            if level < 4:
                small = self.small_down[level](small + fused)
                large = self.large_down[level](large + fused)
                elongated = self.elongated_down[level](elongated + fused)
        enc1, enc2, enc3, enc4, bottleneck = fused_features
        x = self.dec4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), enc3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), enc2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), enc1], dim=1))
        return {
            "logits": self.classifier(x),
            "small_logits": F.interpolate(
                self.small_aux(small),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "large_logits": F.interpolate(
                self.large_aux(large),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "elongated_logits": F.interpolate(
                self.elongated_aux(elongated),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "point_logits": F.interpolate(
                self.small_aux(small),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "line_logits": F.interpolate(
                self.elongated_aux(elongated),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "area_logits": F.interpolate(
                self.large_aux(large),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
            "fusion_weights": gate_maps[-1],
            "fusion_weights_by_level": gate_maps,
            "position_attention": position_map,
            "elongated_position_attention": elongated_position_map,
        }
