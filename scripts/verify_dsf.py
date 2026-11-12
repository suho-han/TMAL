#!/usr/bin/env python3
"""Verify DirectionalScaleFusion and DecoderGuidedDirectionalScaleFusion shapes and gradients."""
import sys
sys.path.insert(0, 'src')

import torch
from model.DconnNet import DirectionalScaleFusion, DecoderGuidedDirectionalScaleFusion

c1 = torch.randn(2, 8, 64, 64)
c2 = torch.randn(2, 8, 64, 64)

# DSF: 2 branches
dsf = DirectionalScaleFusion(conn_ch=8, num_scales=2, hidden_ch=64)
C_fused, alpha = dsf(c1, c2)
assert C_fused.shape == (2, 8, 64, 64), f'C_fused: {C_fused.shape}'
assert alpha.shape == (2, 2, 8, 64, 64), f'alpha: {alpha.shape}'
print(f'[DSF 2-branch] C_fused={tuple(C_fused.shape)}, alpha={tuple(alpha.shape)} OK')

# DSF: 3 branches
dsf3 = DirectionalScaleFusion(conn_ch=8, num_scales=3, hidden_ch=64)
C_fused3, alpha3 = dsf3(c1, c2, torch.randn(2, 8, 64, 64))
assert C_fused3.shape == (2, 8, 64, 64)
assert alpha3.shape == (2, 3, 8, 64, 64)
print(f'[DSF 3-branch] C_fused={tuple(C_fused3.shape)}, alpha={tuple(alpha3.shape)} OK')

# DG-DSF: 2 branches, use_residual=True (default)
D = torch.randn(2, 32, 64, 64)
dg_dsf = DecoderGuidedDirectionalScaleFusion(dec_ch=32, conn_ch=8, num_scales=2, hidden_ch=64)
C_fused2, alpha2, gamma = dg_dsf(D, c1, c2)
assert C_fused2.shape == (2, 8, 64, 64), f'C_fused2: {C_fused2.shape}'
assert alpha2.shape == (2, 2, 8, 64, 64), f'alpha2: {alpha2.shape}'
assert gamma.shape == (2, 8, 64, 64), f'gamma: {gamma.shape}'
print(f'[DG-DSF use_residual=True] C_fused={tuple(C_fused2.shape)}, alpha={tuple(alpha2.shape)}, gamma={tuple(gamma.shape)} OK')

# DG-DSF: use_residual=False
dg_dsf_nores = DecoderGuidedDirectionalScaleFusion(dec_ch=32, conn_ch=8, num_scales=2, use_residual=False)
C_fused_nr, alpha_nr, gamma_nr = dg_dsf_nores(D, c1, c2)
assert gamma_nr is None, f'gamma should be None when use_residual=False'
print(f'[DG-DSF use_residual=False] gamma=None OK')

# Backward pass
loss = C_fused.sum() + C_fused2.sum()
loss.backward()
print('[Backward] Gradient flow OK')

# Softmax property: alpha sums to 1 along scale dim
alpha_sum = alpha.sum(dim=1)
assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5), \
    f'alpha does not sum to 1: max_err={( alpha_sum - 1).abs().max().item():.6f}'
print('[Softmax] alpha sums to 1 per direction per pixel OK')

print('\nAll verification tests passed.')
