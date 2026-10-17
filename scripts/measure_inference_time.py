#!/usr/bin/env python3
"""Per-image inference time for all in-repo baselines vs. the Proposed model.

Combines the model set from measure_baseline_params.py (U-Net, Attention U-Net,
CSNet, FRU-Net, DconnNet) with the GPU timing loop from measure_model_cost.py,
and writes a Markdown report (params / GMac / ms per 512x512 image, batch=1).

External baselines with no architecture in this repo (CAPE, V-GAN) are not
timed here. SegDiff/MedSegDiff are diffusion models that denoise over many
sampling steps, so a single forward pass is not the right proxy for their
inference cost; they are excluded by default (--with-diffusion times just
their per-step forward cost, labeled as such).

Usage:
    python scripts/measure_inference_time.py
    python scripts/measure_inference_time.py --res 512 512
    python scripts/measure_inference_time.py --cpu
    python scripts/measure_inference_time.py --with-diffusion
"""
import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, 'src')

import torch
from model.DconnNet import DconnNet
from model.registry import build_model

try:
    from fvcore.nn import FlopCountAnalysis
    _HAVE_FVCORE = True
except Exception:
    _HAVE_FVCORE = False

COMMON = dict(num_class=1, conn_num=8, conn_layout='standard8', fusion_residual_scale=1.0)


def make_args(**over):
    base = dict(num_channels=3, num_class=1, conn_num=8, conn_layout='standard8',
                simple_loss=False, image_size=512, diff_timesteps=100)
    base.update(over)
    return SimpleNamespace(**base)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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


def build_models(with_diffusion):
    """Returns list of (label, model, note)."""
    models = []
    models.append(("DconnNet (baseline)",
                    DconnNet(**COMMON, conn_fusion='none', use_seg_aux=False).eval(), ""))

    for disp, name in [("U-Net", "unet"), ("Attention U-Net", "attention_unet"),
                        ("CSNet", "csnet"), ("FRU-Net", "frunet")]:
        try:
            models.append((disp, build_model(name, make_args()).eval(), ""))
        except Exception as e:
            models.append((disp, None, f"skipped: {type(e).__name__}: {e}"))

    if with_diffusion:
        for disp, name in [("SegDiff", "segdiff"), ("MedSegDiff", "medsegdiff")]:
            try:
                models.append((disp, build_model(name, make_args()).eval(),
                                "single denoising step only, not full sampling"))
            except Exception as e:
                models.append((disp, None, f"skipped: {type(e).__name__}: {e}"))

    models.append(("Proposed (inference)",
                    DconnNet(**COMMON, conn_fusion='dg_dsf', fusion_branches='io',
                              use_seg_aux=False).eval(), ""))
    return models


def forward1(model, x):
    """Baselines return (dist, conn) tuples; DconnNet returns a similar tuple.
    A single call is all the timing loop needs, regardless of output shape."""
    model(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--res', type=int, nargs=2, default=[512, 512], help='H W (default 512 512)')
    ap.add_argument('--cpu', action='store_true', help='force CPU')
    ap.add_argument('--with-diffusion', action='store_true',
                    help='also time SegDiff/MedSegDiff single-step forward (not full sampling)')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--out', default='output/summary/inference_time.md')
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device('cuda' if use_cuda else 'cpu')
    H, W = args.res
    x = torch.randn(1, 3, H, W, device=device)
    dev_name = torch.cuda.get_device_name(0) if use_cuda else 'CPU'

    if not _HAVE_FVCORE:
        print("WARNING: fvcore not installed; GMac will be N/A.\n")

    rows = []  # (label, params, macs, t_mean, t_std, note)
    for label, model, note in build_models(args.with_diffusion):
        if model is None:
            rows.append((label, None, None, None, None, note))
            print(f"[skip] {label}: {note}")
            continue
        model = model.to(device)
        total = count_params(model)
        macs = count_macs(model, x)
        try:
            t_mean, t_std = time_inference(model, x, args.warmup, args.iters)
        except Exception as e:
            t_mean = t_std = None
            note = (note + "; " if note else "") + f"timing failed: {type(e).__name__}: {e}"
        rows.append((label, total, macs, t_mean, t_std, note))
        del model
        if use_cuda:
            torch.cuda.empty_cache()

    # ---- console ----
    print(f"\nPer-image inference time @ {H}x{W}, batch=1   (device: {dev_name})\n")
    hdr = f"{'Model':<24}{'Params(M)':>11}{'ms/img':>14}"
    print(hdr)
    print('-' * len(hdr))
    for label, total, macs, tm, ts, note in rows:
        p_s = "N/A" if total is None else f"{total/1e6:.2f}"
        t_s = "N/A" if tm is None else f"{tm:.2f}±{ts:.2f}"
        print(f"{label:<24}{p_s:>11}{t_s:>14}")
        if note:
            print(f"    note: {note}")

    # ---- markdown ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines = [
        "# Per-image inference time",
        "",
        f"Resolution: {H}x{W}, batch=1, device: {dev_name}.",
        f"Timing: {args.warmup} warmup + {args.iters} timed iterations "
        "(CUDA events, synchronized) per model; mean ± std reported.",
        "",
        "External baselines with no architecture in this repo (CAPE, V-GAN) are not "
        "timed here. SegDiff/MedSegDiff are diffusion models that sample over many "
        "denoising steps" + (
            "; the row below times a single forward step only, not full sampling."
            if args.with_diffusion else
            " and are excluded by default (`--with-diffusion` to include a single-step timing)."
        ),
        "",
        "| Model | Params (M) | ms/img |",
        "|---|---:|---:|",
    ]
    for label, total, macs, tm, ts, note in rows:
        if total is None:
            lines.append(f"| {label} | N/A | N/A |")
            continue
        p_s = f"{total/1e6:.2f}"
        t_s = "N/A" if tm is None else f"{tm:.2f} ± {ts:.2f}"
        lab = f"**{label}**" if label.startswith("Proposed") else label
        lines.append(f"| {lab} | {p_s} | {t_s} |")
        if note:
            lines.append(f"| | | *({note})* |")
    lines.append("")
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nMarkdown report written to {args.out}")


if __name__ == '__main__':
    main()
