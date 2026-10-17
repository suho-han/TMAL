"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597
"""

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Double convolution block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, dim=64, device=None):
        """
        Initialize the U-Net model.
        :param in_channels: Number of input channels (e.g., 3 for RGB images).
        :param out_channels: Number of output channels (e.g., 1 for binary segmentation).
        :param dim: Base number of channels for the first layer, doubling at each level.
        :param loss: Loss function to be used for training (default is DiceCELoss).
        :param device: Device to run the model on (e.g., 'cuda' or 'cpu').
        """
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        if device is not None:
            device = torch.device(device)
        else:
            device = torch.device("cpu")
        self.device = device

        # Encoder (contracting path)
        self.encoders = nn.ModuleList()
        # 3, 64, 128, 256, 512, 1024
        channels = [in_channels] + [dim * (2**i) for i in range(5)]
        for idx in range(5):
            self.encoders.append(DoubleConv(channels[idx], channels[idx + 1]))

        # Max pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder (expanding path)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_channels = channels[::-1][:-1]
        for idx in range(4):
            self.upconvs.append(
                nn.ConvTranspose2d(
                    decoder_channels[idx],
                    decoder_channels[idx + 1],
                    kernel_size=2,
                    stride=2,
                )
            )
            in_ch = decoder_channels[idx + 1] * 2
            out_ch = decoder_channels[idx + 1]
            self.decoders.append(DoubleConv(in_ch, out_ch))

        # Final output layer
        self.final_conv = nn.Conv2d(channels[1], out_channels, kernel_size=1)

    def feed_data(self, batch):
        input, target = batch["image"].to(
            self.device), batch["label"].to(self.device)
        idx = batch.get("idx", None)
        return input, target, idx

    def forward(self, x):
        """
        Forward pass through the U-Net model.
        :param image: Input image tensor.
        :return: Output of the model.
        """
        skip_connections = []
        # Encoder
        for idx, down in enumerate(self.encoders):
            x = down(x)
            if idx != len(self.encoders) - 1:  # bottleneck 출력 제외
                skip_connections.append(x)
                x = self.pool(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            x = torch.cat([skip_connections[idx], x], dim=1)
            x = self.decoders[idx](x)

        return self.final_conv(x)


from model.registry import register  # noqa: E402


class _UnetDual(Unet):
    def forward(self, x):
        o = super().forward(x)
        return o, o


@register('unet')
def _build_unet(args):
    in_ch = int(getattr(args, 'num_channels', 3) or 3)
    if getattr(args, 'simple_loss', False):
        out_ch = args.num_class
    else:
        from connect_loss import resolve_connectivity_layout
        layout = resolve_connectivity_layout(args.conn_num, getattr(args, 'conn_layout', None))
        out_ch = args.num_class * layout['channel_count']
    return _UnetDual(in_channels=in_ch, out_channels=out_ch, dim=64)
