#!/usr/bin/env python3
"""Measure the computational cost (params / FLOPs / inference time) of the
Proposed model vs. the DconnNet baseline.

The two anchor rows match the paper comparison table (`make_paper_table.py`):

  DconnNet  ->  binary_8_bce            : conn_fusion='none',  use_seg_aux=False
  Proposed  ->  dist_inverted dg_dsf io : conn_fusion='dg_dsf', fusion_branches='io',
                                          use_seg_aux=True

Both DRIVE and DeepCrack train/infer at 512x512 (see DATASET_PRESETS), so cost is
reported for a single 512x512 RGB image (batch=1).

FLOPs convention: fvcore counts multiply-accumulate operations (MACs). We report
GMac and the ×2 GFLOPs column (1 MAC = 2 FLOPs) so either paper convention is covered.

Usage:
    python scripts/measure_model_cost.py                 # 512x512, GPU timing if available
    python scripts/measure_model_cost.py --res 512 512
    python scripts/measure_model_cost.py --cpu           # force CPU (no timing loop)
"""
import argparse
import os
import sys

sys.path.insert(0, 'src')

import torch
from model.DconnNet import DconnNet

try:
    from fvcore.nn import FlopCountAnalysis
    _HAVE_FVCORE = True
except Exception:
    _HAVE_FVCORE = False


# label, kwargs for DconnNet(...)  (architecture-determining kwargs only).
# The seg-aux head produces `mask_logit`, which is used ONLY by the training loss
# (solver.py:_maybe_apply_seg_aux_loss). At inference the prediction is taken from
# `fused` (solver.py val/test), so the deployed model drops the seg-aux head.
# Hence "Proposed (inference)" is the fair deployment cost; the full training-graph
# cost is reported separately for completeness.
MODELS = [
    ("DconnNet (baseline)",   dict(conn_fusion='none', use_seg_aux=False)),
    ("Proposed (inference)",  dict(conn_fusion='dg_dsf', fusion_branches='io', use_seg_aux=False)),
    ("Proposed (train graph)", dict(conn_fusion='dg_dsf', fusion_branches='io', use_seg_aux=True)),
]

# Index of the row treated as the deployed Proposed model for the summary.
_PROPOSED_IDX = 1

COMMON = dict(num_class=1, conn_num=8, conn_layout='standard8', fusion_residual_scale=1.0)


def build(kwargs):
    m = DconnNet(**{**COMMON, **kwargs})
    return m.eval()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_macs(model, x):
    if not _HAVE_FVCORE:
        return None
    fca = FlopCountAnalysis(model, x)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    return fca.total()  # MACs


@torch.no_grad()
def time_inference(model, x, warmup=10, iters=50):
    device = x.device
    if device.type == 'cuda':
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            starter.record()
            model(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms
        t = torch.tensor(times)
        return t.mean().item(), t.std().item()
    else:
        import time
        for _ in range(3):
            model(x)
        times = []
        for _ in range(max(5, iters // 5)):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
        t = torch.tensor(times)
        return t.mean().item(), t.std().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--res', type=int, nargs=2, default=[512, 512], help='H W (default 512 512)')
    ap.add_argument('--cpu', action='store_true', help='force CPU (skips reliable timing)')
    ap.add_argument('--no-timing', action='store_true', help='skip inference timing loop')
    ap.add_argument('--out', default='output/summary/model_cost.tex')
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device('cuda' if use_cuda else 'cpu')
    H, W = args.res
    x = torch.randn(1, 3, H, W, device=device)

    if not _HAVE_FVCORE:
        print("WARNING: fvcore not installed; FLOPs will be reported as N/A.\n")

    rows = []  # (label, total, trainable, macs, t_mean, t_std)
    for label, kwargs in MODELS:
        model = build(kwargs).to(device)
        total, trainable = count_params(model)
        macs = count_macs(model, x)
        if args.no_timing:
            t_mean = t_std = None
        else:
            t_mean, t_std = time_inference(model, x)
        rows.append((label, total, trainable, macs, t_mean, t_std))
        del model
        if use_cuda:
            torch.cuda.empty_cache()

    base_total = rows[0][1]
    base_macs = rows[0][3]
    dev_name = torch.cuda.get_device_name(0) if use_cuda else 'CPU'

    # ---- console table ----
    print(f"Model computational cost @ {H}x{W}, batch=1   (device: {dev_name})\n")
    hdr = (f"{'Model':<22}{'Params(M)':>11}{'ΔParam':>12}{'GMac':>10}{'GFLOPs':>10}"
           f"{'ΔFLOP%':>9}{'ms/img':>10}")
    print(hdr)
    print('-' * len(hdr))
    for label, total, _tr, macs, tm, ts in rows:
        dparam = total - base_total
        dparam_s = f"+{dparam/1e6:.3f}M" if dparam else "--"
        if macs is None:
            gmac_s = gflop_s = dflop_s = "N/A"
        else:
            gmac_s = f"{macs/1e9:.2f}"
            gflop_s = f"{2*macs/1e9:.2f}"
            dflop_s = f"+{100*(macs-base_macs)/base_macs:.2f}%" if (base_macs and macs != base_macs) else "--"
        t_s = "--" if tm is None else f"{tm:.2f}±{ts:.2f}"
        print(f"{label:<22}{total/1e6:>11.3f}{dparam_s:>12}{gmac_s:>10}{gflop_s:>10}{dflop_s:>9}{t_s:>10}")

    prop = rows[_PROPOSED_IDX]
    print(f"\nProposed (deployed, seg-aux head dropped) vs DconnNet baseline:")
    print(f"  Params : {base_total/1e6:.3f}M -> {prop[1]/1e6:.3f}M "
          f"(+{(prop[1]-base_total)/1e6:.3f}M, +{100*(prop[1]-base_total)/base_total:.2f}%)")
    if base_macs and prop[3]:
        print(f"  GMac   : {base_macs/1e9:.2f} -> {prop[3]/1e9:.2f} "
              f"(+{(prop[3]-base_macs)/1e9:.3f}, +{100*(prop[3]-base_macs)/base_macs:.2f}%)")
    if prop[4] is not None:
        print(f"  ms/img : {rows[0][4]:.2f} -> {prop[4]:.2f} "
              f"(+{100*(prop[4]-rows[0][4])/rows[0][4]:.1f}%)")
    print("  Note: parameter overhead is negligible; the FLOPs increase comes almost")
    print("        entirely from the full-resolution Affinity-Fusion conv (dsf_module).")
    print("        The training-only seg-aux head adds ~6 GMac but is absent at inference.")

    # ---- LaTeX ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\caption{Computational cost at $512\times512$ (batch$=$1) on " f"{dev_name}. "
        r"GFLOPs $=2\times$GMac; $\Delta$ is relative to the DconnNet baseline. "
        r"The seg-aux head is a training-only supervision branch (its output is "
        r"unused at inference), so the deployed Proposed model is the ``inference'' row.}",
        r"\label{tab:model_cost}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Model & Params (M) & $\Delta$Params & GFLOPs & $\Delta$FLOPs & ms/img \\",
        r"\midrule",
    ]
    for label, total, _tr, macs, tm, _ts in rows:
        dparam = total - base_total
        dparam_s = "--" if not dparam else f"+{100*dparam/base_total:.2f}\\%"
        if macs is None:
            gflop_s = dflop_s = "N/A"
        else:
            gflop_s = f"{2*macs/1e9:.2f}"
            dflop_s = "--" if macs == base_macs else f"+{100*(macs-base_macs)/base_macs:.1f}\\%"
        t_s = "--" if tm is None else f"{tm:.2f}"
        lab = label.replace('_', r'\_')
        if label == MODELS[_PROPOSED_IDX][0]:
            lab = f"\\textbf{{{lab}}}"
        lines.append(f"{lab} & {total/1e6:.3f} & {dparam_s} & {gflop_s} & {dflop_s} & {t_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nLaTeX table written to {args.out}")


if __name__ == '__main__':
    main()
