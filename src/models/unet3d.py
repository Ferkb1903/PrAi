from typing import List

import torch
from torch import nn


class ConvBlock3D(nn.Module):
    """Bloque convolucional base usado en encoder y decoder.

    Estructura: Conv3D -> INorm -> LeakyReLU -> Conv3D -> INorm -> LeakyReLU.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """U-Net 3D mínima para corrección de dosis volumétrica.

    Entrada esperada: (B, C_in, D, H, W)
    Salida: (B, C_out, D, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 16,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth debe ser >= 2")

        enc_channels: List[int] = [base_channels * (2**i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        # Encoder: extrae contexto multi-escala y reduce resolución con max-pooling.
        prev = in_channels
        for ch in enc_channels:
            self.encoders.append(ConvBlock3D(prev, ch))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev = ch

        # Bottleneck: mayor capacidad en la resolución más baja.
        self.bottleneck = ConvBlock3D(enc_channels[-1], enc_channels[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Decoder: upsampling + skip connections para recuperar detalle espacial.
        dec_in = enc_channels[-1] * 2
        for ch in reversed(enc_channels):
            self.upconvs.append(nn.ConvTranspose3d(dec_in, ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock3D(ch * 2, ch))
            dec_in = ch

        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            # Ajuste defensivo de shape para volúmenes no divisibles exactamente por 2^depth.
            if x.shape[-3:] != skip.shape[-3:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.head(x)
