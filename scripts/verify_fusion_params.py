#!/usr/bin/env python3
"""Verify the Affinity-Fusion parameter overhead across the scale set S.

The ablation PDF's "Ablation study on S" table compares scale sets
{1}, {2}, {1,2} (proposed), {1,2,3}, {1,2,3,4}. Each maps to a different
`fusion_branches` config and therefore builds a structurally different
Affinity-Fusion module. This script instantiates each model, runs a forward
pass (inference) to confirm the architecture is valid at every S, and counts
the parameters of the Affinity-Fusion block vs. the whole model. It then
cross-checks the measured per-scale delta against the analytic formula, so the
resulting table is self-validating.

Usage:
    python scripts/verify_fusion_params.py                # dg_direct (displayed S table)
    python scripts/verify_fusion_params.py --fusion dg_dsf
"""
import argparse
import os
import sys

sys.path.insert(0, 'src')

import torch
from model.DconnNet import DconnNet

# Names of the optional per-scale connectivity heads created in DconnNet.__init__
# (one extra head appears for each added scale).
_HEAD_ATTRS = (
    'cls_pred_inner_conv',
    'cls_pred_middle_conv',
    'cls_pred_outer_conv',
    'cls_pred_scale3_conv',
    'cls_pred_scale4_conv',
)

# Per-added-scale parameter growth of the Affinity-Fusion block (analytic):
#   shared[0] conv: +conn_ch * hidden_ch * 3*3 = 8*64*9 = 4608
#   one 1x1 head  : +dec_ch * conn_ch + conn_ch = 32*8 + 8 = 264
#   dg_dsf only: scale_attn +hidden_ch*conn_ch + conn_ch = 64*8 + 8 = 520
_PER_SCALE = {
    'dg_direct': 4608 + 264,        # 4872
    'dg':        4608 + 264,        # 4872
    'dg_dsf':    4608 + 264 + 520,  # 5392
    'dsf':       4608 + 264 + 520,  # 5392 (no decoder feature, same scale-dependent terms)
}


def n_params(module):
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def fusion_params(model):
    """Affinity-Fusion submodule + the present per-scale connectivity heads."""
    fusion_mod = getattr(model, 'decoder_guided_fusion', None) or getattr(model, 'dsf_module', None)
    total = n_params(fusion_mod)
    for attr in _HEAD_ATTRS:
        total += n_params(getattr(model, attr, None))
    return total


def build_rows(fusion):
    """S -> config map. {1}/{2} are single-scale conn_fusion='none' baselines."""
    if fusion in ('dg_dsf', 'dsf'):
        multi = [
            ('{1,2} (proposed)', 'standard8', 'io'),
            ('{1,2,3}',          'standard8', 'iop'),
            ('{1,2,3,4}',        'standard8', 'iopf'),
        ]
    else:  # dg_direct / dg -> the displayed S table
        multi = [
            ('{1,2} (proposed)', 'standard8', 'io'),
            ('{1,2,3}',          'standard8', 'iop'),
            ('{1,2,3,4}',        'standard8', 'iopf'),
        ]
    rows = [
        # label, conn_fusion, conn_layout, fusion_branches
        ('{1}', 'none', 'standard8', None),
        ('{2}', 'none', 'out8',      None),
    ]
    for label, layout, branches in multi:
        rows.append((label, fusion, layout, branches))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fusion', default='dg_direct',
                    choices=['dg_direct', 'dg', 'dg_dsf', 'dsf'],
                    help="Fusion variant for the multi-scale rows (default: dg_direct = displayed S table).")
    ap.add_argument('--out', default='output/summary/fusion_param_overhead.tex',
                    help="Where to write the LaTeX table.")
    args = ap.parse_args()

    x = torch.randn(1, 3, 512, 512)
    rows = build_rows(args.fusion)

    results = []  # (label, num_scales, total, fusion, fusion_mod_name)
    proposed_total = proposed_fusion = None

    for label, conn_fusion, conn_layout, branches in rows:
        kwargs = dict(num_class=1, conn_num=8, conn_layout=conn_layout,
                      conn_fusion=conn_fusion, use_seg_aux=True)
        if branches is not None:
            kwargs['fusion_branches'] = branches
        model = DconnNet(**kwargs).eval()

        # --- inference verification ---
        with torch.no_grad():
            out = model(x)
        assert out['fused'].shape == (1, 8, 512, 512), \
            f"{label}: unexpected fused shape {tuple(out['fused'].shape)}"
        num_scales = 1 if branches is None else len(str(branches))
        if conn_fusion in ('dsf', 'dg_dsf'):
            alpha = out['fusion_alpha']
            assert alpha.shape[1] == num_scales, \
                f"{label}: alpha num_scales {alpha.shape[1]} != |S| {num_scales}"

        tot = n_params(model)
        fus = fusion_params(model)
        results.append((label, num_scales, tot, fus))
        if label.startswith('{1,2} '):
            proposed_total, proposed_fusion = tot, fus

    # --- analytic cross-check: consecutive multi-scale rows differ by _PER_SCALE ---
    expected = _PER_SCALE[args.fusion]
    multi = [r for r in results if r[1] >= 2]
    multi.sort(key=lambda r: r[1])
    for (la, sa, _ta, fa), (lb, sb, _tb, fb) in zip(multi, multi[1:]):
        if sb - sa == 1:
            delta = fb - fa
            assert delta == expected * (sb - sa), (
                f"fusion delta {la}->{lb} = {delta}, expected {expected}")

    # --- console table ---
    print(f"Affinity-Fusion parameter overhead across S  (fusion='{args.fusion}')\n")
    hdr = f"{'S':<18}{'|S|':>4}{'fusion':>12}{'Δfusion':>12}{'total':>14}{'fusion%':>10}"
    print(hdr)
    print('-' * len(hdr))
    for label, ns, tot, fus in results:
        dfus = '' if proposed_fusion is None else f"{fus - proposed_fusion:+d}"
        pct = 100.0 * fus / tot if tot else 0.0
        print(f"{label:<18}{ns:>4}{fus:>12,}{dfus:>12}{tot:>14,}{pct:>9.3f}%")
    print(f"\nPer-added-scale fusion growth (analytic, verified): +{expected:,} params")
    if proposed_total:
        print(f"Proposed total params: {proposed_total:,}")

    # --- LaTeX table for the ablation PDF ---
    fusion_tex = args.fusion.replace('_', r'\_')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"$\mathcal{S}$ & $|\mathcal{S}|$ & Fusion params & $\Delta$ & Fusion \% of total \\",
        r"\midrule",
    ]
    for label, ns, tot, fus in results:
        dfus = '--' if proposed_fusion is None else f"{fus - proposed_fusion:+d}"
        pct = 100.0 * fus / tot if tot else 0.0
        tex_label = r"$\mathcal{S}=\{" + label.split(' ')[0].strip('{}') + r"\}$"
        prop = r" (prop.)" if 'proposed' in label else ""
        lines.append(f"{tex_label}{prop} & {ns} & {fus:,} & {dfus} & {pct:.3f}\\% \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\vspace{0.15cm}",
        r"\caption{Affinity-Fusion parameter overhead across $\mathcal{S}$ "
        f"(fusion: \\texttt{{{fusion_tex}}}). "
        r"Each added scale grows the fusion block by a fixed, negligible amount.}",
        r"\label{tab:fusion_param_overhead}", r"\end{table}", "",
    ]
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nLaTeX table written to {args.out}")
    print("All checks passed.")


if __name__ == '__main__':
    main()
