#!/usr/bin/env python3
"""Verify DconnNet integrates DSF and DG-DSF correctly."""
import sys
sys.path.insert(0, 'src')

import torch
from model.DconnNet import DconnNet

x = torch.randn(1, 3, 512, 512)

configs = [
    dict(conn_fusion='dsf',    fusion_branches='io',  label='DSF io'),
    dict(conn_fusion='dsf',    fusion_branches='iof', label='DSF iof (3 branches)'),
    dict(conn_fusion='dg_dsf', fusion_branches='io',  label='DG-DSF io'),
    dict(conn_fusion='dg_dsf', fusion_branches='iof', label='DG-DSF iof (3 branches)'),
]

for cfg in configs:
    label = cfg.pop('label')
    model = DconnNet(num_class=1, conn_num=8, conn_layout='standard8', **cfg).eval()
    with torch.no_grad():
        out = model(x)
    assert 'fused' in out, f'missing fused key'
    assert out['fused'].shape == (1, 8, 512, 512), f"fused shape: {out['fused'].shape}"
    assert 'inner' in out, f'missing inner key'
    if 'dsf' in cfg.get('conn_fusion', ''):
        assert 'fusion_alpha' in out, f'missing fusion_alpha for {label}'
        alpha = out['fusion_alpha']
        assert alpha.dim() == 5, f'alpha should be 5D [B, num_scales, 8, H, W], got {alpha.shape}'
        alpha_sum = alpha.sum(dim=1)
        assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-4), 'alpha not normalized'
    if cfg.get('conn_fusion') == 'dg_dsf':
        assert 'fusion_gate' in out, f'missing fusion_gate (gamma) for {label}'
        gamma = out['fusion_gate']
        assert gamma.dim() == 4, f'gamma should be 4D [B, 8, H, W], got {gamma.shape}'
        assert (gamma >= 0).all() and (gamma <= 1).all(), 'gamma should be sigmoid output in [0,1]'
    print(f'[{label}] fused={tuple(out["fused"].shape)}, alpha={tuple(out["fusion_alpha"].shape)} OK')

print('\nAll DconnNet integration tests passed.')
