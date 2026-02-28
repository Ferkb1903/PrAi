from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x)
        return x * scale


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = False, se_reduction: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.se = SEBlock3D(out_ch, se_reduction) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.se is not None:
            out = self.se(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = False, se_reduction: int = 8) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_ch, out_ch, use_se=use_se, se_reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_se: bool = False, se_reduction: int = 8) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch, use_se=use_se, se_reduction=se_reduction)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        if dz != 0 or dy != 0 or dx != 0:
            x = nn.functional.pad(
                x,
                [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResidualUNet3D(nn.Module):
    """Residual 3D U-Net para corrección de dosis.

    Entrada esperada: canales [D_low, SPR, E0, BeamMask(optional)].
    Salida:
      - residual=True: delta de dosis
      - residual=False: dosis predicha = D_low + delta
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 24,
        residual: bool = True,
        use_se: bool = False,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.residual = residual

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = ConvBlock3D(in_channels, c1, use_se=use_se, se_reduction=se_reduction)
        self.enc2 = DownBlock(c1, c2, use_se=use_se, se_reduction=se_reduction)
        self.enc3 = DownBlock(c2, c3, use_se=use_se, se_reduction=se_reduction)
        self.enc4 = DownBlock(c3, c4, use_se=use_se, se_reduction=se_reduction)

        self.bottleneck = DownBlock(c4, c4, use_se=use_se, se_reduction=se_reduction)

        self.up4 = UpBlock(c4, c4, c3, use_se=use_se, se_reduction=se_reduction)
        self.up3 = UpBlock(c3, c3, c2, use_se=use_se, se_reduction=se_reduction)
        self.up2 = UpBlock(c2, c2, c1, use_se=use_se, se_reduction=se_reduction)
        self.up1 = UpBlock(c1, c1, c1, use_se=use_se, se_reduction=se_reduction)

        self.out_head = nn.Conv3d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_low = x[:, 0:1, ...]

        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b = self.bottleneck(s4)
        u4 = self.up4(b, s4)
        u3 = self.up3(u4, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)

        delta = self.out_head(u1)
        if self.residual:
            return delta
        return d_low + delta
