# Ultralytics YOLO, AGPL-3.0 license
"""Lightweight depth-completion heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("SPNetDepthHead",)


class SPNorm2d(nn.Module):
    """Scale-preserving normalization: normalize locally, then restore feature scale."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.var(dim=(2, 3), keepdim=True, unbiased=False).add(self.eps).sqrt()
        return self.norm((x - mean) / std) * std + mean


class SPDepthBlock(nn.Module):
    """Small decoder block used by the SPNet-style depth head."""

    def __init__(self, channels, activation="silu"):
        super().__init__()
        act = nn.ReLU(inplace=True) if activation == "relu" else nn.SiLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            SPNorm2d(channels),
            act,
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            SPNorm2d(channels),
            nn.ReLU(inplace=True) if activation == "relu" else nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SparseStatsPrompt(nn.Module):
    """Convert sparse-depth statistics into a small FiLM prompt."""

    def __init__(self, channels):
        super().__init__()
        width = max(channels // 2, 8)
        self.mlp = nn.Sequential(nn.Linear(6, width), nn.SiLU(inplace=True), nn.Linear(width, channels * 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, sparse, sparse_mask=None):
        if sparse is None:
            return x
        mask = sparse > 0 if sparse_mask is None else sparse_mask > 0.5
        b = sparse.shape[0]
        flat = sparse.flatten(1)
        mask_flat = mask.flatten(1)
        count = mask_flat.sum(1).clamp_min(1).to(sparse.dtype)
        coverage = mask_flat.to(sparse.dtype).mean(1)
        known_sum = (flat * mask_flat.to(sparse.dtype)).sum(1)
        mean = known_sum / count
        var = (((flat - mean[:, None]) ** 2) * mask_flat.to(sparse.dtype)).sum(1) / count
        std = var.clamp_min(0).sqrt()
        h = sparse.shape[-2]
        upper_mask = mask[..., : h // 2, :].flatten(1)
        lower_mask = mask[..., h // 2 :, :].flatten(1)
        upper = sparse[..., : h // 2, :].flatten(1)
        lower = sparse[..., h // 2 :, :].flatten(1)
        upper_mean = (upper * upper_mask.to(sparse.dtype)).sum(1) / upper_mask.sum(1).clamp_min(1).to(sparse.dtype)
        lower_mean = (lower * lower_mask.to(sparse.dtype)).sum(1) / lower_mask.sum(1).clamp_min(1).to(sparse.dtype)
        stats = torch.stack((coverage, mean, std, upper_mean, lower_mean, lower_mean - upper_mean), dim=1)
        gamma, beta = self.mlp(stats).view(b, 2, x.shape[1], 1, 1).unbind(1)
        return x * (1 + gamma) + beta


class SPNetDepthHead(nn.Module):
    """
    Minimal SPNet-style depth decoder for ConvNeXt multi-scale features.

    The head projects C2/C3/C4/C5 features to a small shared width, upsamples them to C2 resolution,
    fuses them, then upsamples once more to the input image size.
    """

    def __init__(
        self,
        channels,
        hidden=64,
        out_channels=1,
        variant="spnet",
        sparse_encoder=False,
        feature_adapters=False,
        scale_prompt=False,
        activation="silu",
        output_activation="identity",
    ):
        super().__init__()
        if not channels:
            raise ValueError("SPNetDepthHead requires at least one input channel.")
        self.channels = tuple(channels)
        self.variant = str(variant or "spnet")
        self.output_activation = str(output_activation or "identity").lower()
        variant_tokens = {x.strip().lower() for x in self.variant.replace("+", "_").split("_") if x.strip()}
        sparse_encoder = bool(sparse_encoder or {"sparse", "encoder"} <= variant_tokens)
        feature_adapters = bool(feature_adapters or "adapters" in variant_tokens or "adapter" in variant_tokens)
        scale_prompt = bool(scale_prompt or "scale" in variant_tokens or "prompt" in variant_tokens)
        activation = "relu" if activation == "relu" or "relu" in variant_tokens else "silu"
        self.proj = nn.ModuleList(nn.Conv2d(c, hidden, 1, bias=False) for c in self.channels)
        self.adapters = (
            nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
                    SPNorm2d(hidden),
                    nn.ReLU(inplace=True) if activation == "relu" else nn.SiLU(inplace=True),
                    nn.Conv2d(hidden, hidden, 1, bias=False),
                )
                for _ in self.channels
            )
            if feature_adapters
            else None
        )
        sparse_layers = [
            nn.Conv2d(2, hidden, 3, padding=1, bias=False),
            SPNorm2d(hidden),
            nn.ReLU(inplace=True) if activation == "relu" else nn.SiLU(inplace=True),
        ]
        if sparse_encoder:
            sparse_layers.extend(
                [
                    nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
                    SPNorm2d(hidden),
                    nn.ReLU(inplace=True) if activation == "relu" else nn.SiLU(inplace=True),
                ]
            )
        self.sparse_proj = nn.Sequential(*sparse_layers)
        self.scale_prompt = SparseStatsPrompt(hidden) if scale_prompt else None
        self.fuse = nn.Sequential(SPDepthBlock(hidden, activation), SPDepthBlock(hidden, activation))
        self.image_proj = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1, bias=False),
            SPNorm2d(hidden),
            nn.ReLU(inplace=True) if activation == "relu" else nn.SiLU(inplace=True),
        )
        self.refine = SPDepthBlock(hidden, activation)
        self.out = nn.Conv2d(hidden, out_channels, 3, padding=1)

    def forward(
        self,
        features,
        out_shape=None,
        sparse_depth=None,
        image=None,
        preserve_sparse=True,
        sparse_mask=None,
    ):
        if len(features) != len(self.proj):
            raise ValueError(f"Expected {len(self.proj)} depth features, got {len(features)}.")

        target_hw = features[0].shape[-2:]
        fused = None
        for i, (feature, proj) in enumerate(zip(features, self.proj)):
            x = proj(feature)
            if self.adapters is not None:
                x = x + self.adapters[i](x)
            if x.shape[-2:] != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            fused = x if fused is None else fused + x

        divisor = len(features)
        sparse_for_prompt = None
        sparse_mask_for_prompt = None
        if sparse_depth is not None and hasattr(self, "sparse_proj"):
            if sparse_depth.ndim == 3:
                sparse_depth = sparse_depth.unsqueeze(1)
            sparse = sparse_depth.to(device=fused.device, dtype=fused.dtype)
            if sparse.shape[-2:] != target_hw:
                sparse = F.interpolate(sparse, size=target_hw, mode="nearest")
            sparse_for_prompt = sparse
            if sparse_mask is None:
                sparse_mask_for_prompt = (sparse > 0).to(sparse.dtype)
            else:
                sparse_mask_for_prompt = sparse_mask.to(device=fused.device, dtype=fused.dtype)
                if sparse_mask_for_prompt.ndim == 3:
                    sparse_mask_for_prompt = sparse_mask_for_prompt.unsqueeze(1)
                if sparse_mask_for_prompt.shape[-2:] != target_hw:
                    sparse_mask_for_prompt = F.interpolate(sparse_mask_for_prompt, size=target_hw, mode="nearest")
                sparse_mask_for_prompt = (sparse_mask_for_prompt > 0.5).to(sparse.dtype)
            # Depth completion should condition on the known sparse samples, not only compare to them in the loss.
            fused = fused + self.sparse_proj(torch.cat((sparse, sparse_mask_for_prompt), dim=1))
            divisor += 1

        x = fused / divisor
        if self.scale_prompt is not None:
            x = self.scale_prompt(x, sparse_for_prompt, sparse_mask_for_prompt)
        x = self.fuse(x)
        if out_shape is not None and x.shape[-2:] != tuple(out_shape):
            x = F.interpolate(x, size=out_shape, mode="bilinear", align_corners=False)

        if image is not None and hasattr(self, "image_proj"):
            image = image.to(device=x.device, dtype=x.dtype)
            image = image[:, :3]
            if image.shape[-2:] != x.shape[-2:]:
                image = F.interpolate(image, size=x.shape[-2:], mode="bilinear", align_corners=False)
            h, w = x.shape[-2:]
            y_coord = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(
                image.shape[0], 1, h, w
            )
            # A small full-resolution RGB guide keeps the decoder from losing fine image structure at C2 stride.
            x = x + self.image_proj(torch.cat((image, y_coord), dim=1))

        if hasattr(self, "refine"):
            x = self.refine(x)
        x = self.out(x)
        if self.output_activation in {"sigmoid", "normalized"}:
            x = x.sigmoid()
        elif self.output_activation in {"softplus", "positive"}:
            x = F.softplus(x)
        elif self.output_activation in {"relu"}:
            x = F.relu(x)
        elif self.output_activation in {"identity", "linear", "none"}:
            pass
        else:
            raise ValueError(f"Unsupported depth output activation: {self.output_activation}")

        if preserve_sparse and sparse_depth is not None:
            sparse = sparse_depth.to(device=x.device, dtype=x.dtype)
            if sparse.ndim == 3:
                sparse = sparse.unsqueeze(1)
            if sparse.shape[-2:] != x.shape[-2:]:
                sparse = F.interpolate(sparse, size=x.shape[-2:], mode="nearest")
            if sparse_mask is None:
                known = sparse > 0
            else:
                known = sparse_mask.to(device=x.device, dtype=x.dtype)
                if known.ndim == 3:
                    known = known.unsqueeze(1)
                if known.shape[-2:] != x.shape[-2:]:
                    known = F.interpolate(known, size=x.shape[-2:], mode="nearest")
                known = known > 0.5
            x = torch.where(known, sparse.clamp_min(0), x)
        return x
