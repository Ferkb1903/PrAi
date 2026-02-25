from torch import nn

from src.config import defaults
from src.models.unet3d import UNet3D


def build_model() -> nn.Module:
    return UNet3D(
        in_channels=defaults.IN_CHANNELS,
        out_channels=defaults.OUT_CHANNELS,
        base_channels=defaults.BASE_CHANNELS,
        depth=defaults.DEPTH,
    )
