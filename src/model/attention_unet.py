"""
Attention U-Net: Learning Where to Look for the Pancreas
https://arxiv.org/abs/1804.03999

Reference implementation: https://github.com/sfczekalski/attention_unet
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
    """Additive attention gate for skip connections."""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        F_g: channels in gating signal (from decoder)
        F_l: channels in skip connection (from encoder)
        n_coefficients: intermediate attention channels
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return skip_connection * psi


class AttentionUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, dim=64):
        super(AttentionUNet, self).__init__()
        # channels: dim, dim*2, dim*4, dim*8, dim*16
        c = [dim * (2 ** i) for i in range(5)]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, c[0])
        self.enc2 = ConvBlock(c[0], c[1])
        self.enc3 = ConvBlock(c[1], c[2])
        self.enc4 = ConvBlock(c[2], c[3])
        self.enc5 = ConvBlock(c[3], c[4])  # bottleneck

        # Decoder with attention gates
        self.up5 = UpConv(c[4], c[3])
        self.att5 = AttentionBlock(F_g=c[3], F_l=c[3], n_coefficients=c[2])
        self.dec5 = ConvBlock(c[4], c[3])

        self.up4 = UpConv(c[3], c[2])
        self.att4 = AttentionBlock(F_g=c[2], F_l=c[2], n_coefficients=c[1])
        self.dec4 = ConvBlock(c[3], c[2])

        self.up3 = UpConv(c[2], c[1])
        self.att3 = AttentionBlock(F_g=c[1], F_l=c[1], n_coefficients=c[0])
        self.dec3 = ConvBlock(c[2], c[1])

        self.up2 = UpConv(c[1], c[0])
        self.att2 = AttentionBlock(F_g=c[0], F_l=c[0], n_coefficients=c[0] // 2)
        self.dec2 = ConvBlock(c[1], c[0])

        self.final_conv = nn.Conv2d(c[0], out_channels, kernel_size=1)

    def feed_data(self, batch):
        input, target = batch["image"].to(self.device), batch["label"].to(self.device)
        idx = batch.get("idx", None)
        return input, target, idx

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d5 = self.up5(e5)
        d5 = self.dec5(torch.cat([self.att5(gate=d5, skip_connection=e4), d5], dim=1))

        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat([self.att4(gate=d4, skip_connection=e3), d4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([self.att3(gate=d3, skip_connection=e2), d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self.att2(gate=d2, skip_connection=e1), d2], dim=1))

        return self.final_conv(d2)


from model.registry import register  # noqa: E402


class _AttentionUNetDual(AttentionUNet):
    def forward(self, x):
        o = super().forward(x)
        return o, o


@register('attention_unet')
def _build_attention_unet(args):
    in_ch = int(getattr(args, 'num_channels', 3) or 3)
    if getattr(args, 'simple_loss', False):
        out_ch = args.num_class
    else:
        from connect_loss import resolve_connectivity_layout
        layout = resolve_connectivity_layout(args.conn_num, getattr(args, 'conn_layout', None))
        out_ch = args.num_class * layout['channel_count']
    return _AttentionUNetDual(in_channels=in_ch, out_channels=out_ch, dim=64)
