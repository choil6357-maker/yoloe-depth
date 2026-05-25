# Ultralytics YOLO ??, AGPL-3.0 license
"""Backbone adapters."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("TimmBackbone", "FeatureSelect")


class TimmBackbone(nn.Module):
    """Wrap a timm features_only backbone for YOLO-style multi-scale feature extraction."""

    _KNOWN_CHANNELS = {
        "convnextv2_small": (96, 192, 384, 768),
    }

    def __init__(
        self,
        model_name="convnextv2_small",
        pretrained=False,
        out_indices=(1, 2, 3),
        in_chans=3,
        zero_init_extra=True,
    ):
        """Create a timm backbone with configurable pretrained weights and output stages."""
        super().__init__()
        if isinstance(out_indices, int):
            out_indices = (out_indices,)
        self.model_name = model_name
        self.pretrained = pretrained
        self.out_indices = tuple(out_indices)
        self.in_chans = int(in_chans)

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "TimmBackbone requires the 'timm' package. Install timm or use a non-timm YOLOE backbone."
            ) from e

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
            in_chans=self.in_chans,
        )
        if zero_init_extra and self.in_chans > 3:
            self._zero_init_extra_stem_channels()
        self.out_channels = tuple(self.model.feature_info.channels())

    def _zero_init_extra_stem_channels(self):
        """Zero any channels beyond RGB in the first convolutional stem."""
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) and module.weight.ndim == 4 and module.weight.shape[1] == self.in_chans:
                with torch.no_grad():
                    module.weight[:, 3:, :, :].zero_()
                return

    @classmethod
    def channels(cls, model_name="convnextv2_small", out_indices=(1, 2, 3)):
        """Return known output channels without importing timm during YAML parsing."""
        if isinstance(out_indices, int):
            out_indices = (out_indices,)
        if model_name not in cls._KNOWN_CHANNELS:
            raise ValueError(
                f"Unknown timm backbone channels for '{model_name}'. Add them to TimmBackbone._KNOWN_CHANNELS."
            )
        channels = cls._KNOWN_CHANNELS[model_name]
        return [channels[i] for i in out_indices]

    def forward(self, x):
        """Return selected feature maps in increasing stride order."""
        if x.shape[1] < self.in_chans:
            pad = x.new_zeros((x.shape[0], self.in_chans - x.shape[1], *x.shape[-2:]))
            x = torch.cat((x, pad), dim=1)
        elif x.shape[1] > self.in_chans:
            x = x[:, : self.in_chans]
        return list(self.model(x))


class FeatureSelect(nn.Module):
    """Select one tensor from a multi-scale backbone output list."""

    def __init__(self, index=0):
        """Select feature map at index."""
        super().__init__()
        self.index = index

    def forward(self, x):
        """Return a single feature tensor."""
        return x[self.index]
