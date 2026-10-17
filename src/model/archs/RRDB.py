import torch
import torch.nn as nn


class _DenseBlock(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.c1 = nn.Conv2d(nf,       gc, 3, 1, 1)
        self.c2 = nn.Conv2d(nf+gc,    gc, 3, 1, 1)
        self.c3 = nn.Conv2d(nf+2*gc,  gc, 3, 1, 1)
        self.c4 = nn.Conv2d(nf+3*gc,  gc, 3, 1, 1)
        self.c5 = nn.Conv2d(nf+4*gc,  nf, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], 1)))
        return self.c5(torch.cat([x, x1, x2, x3, x4], 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.b1 = _DenseBlock(nf, gc)
        self.b2 = _DenseBlock(nf, gc)
        self.b3 = _DenseBlock(nf, gc)

    def forward(self, x):
        return self.b3(self.b2(self.b1(x))) * 0.2 + x


def make_layer(block_f, n):
    return nn.Sequential(*[block_f() for _ in range(n)])
