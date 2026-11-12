#!/usr/bin/env python3
"""Pick which test sample the ablation-table figure should be built from.

scripts/compare_tp_patches.py compares ROI candidates *within* the sample whose
panels were already exported under output/paper_tables/figures/. This script
goes one level up: it reads every test sample's GT and per-option predictions
straight from the training runs listed in the ablation-paths JSON, runs the same
patch search (scripts/generate_figure.compute_top_tp_patches) on each, and
renders the crop *pair* each sample would give the figure -- since the figure
draws two ROI boxes, a sample is only as good as the weaker of its two crops.

Samples are ranked by that weaker crop's FN-rate gap (how much of the patch's
GT the rivals miss and the highlighted option recovers), and samples whose
whole-image Dice does not follow the figure's option order are dropped.

Usage:
    uv run scripts/compare_tp_samples.py [--dataset DRIVE] [--patches 2]
                                         [--top 6] [sample_id ...]

Switching the figure over to the winner means re-exporting its panels:
    uv run scripts/make_paper_table.py --datasets DRIVE --ablation --ablation-sample <id>
then pointing GROUPS["ablation_table"]["drive_json"] at the new JSON and
clearing TP_PATCH_KEEP (its coordinates belong to the old sample).
"""
import json
import os
import sys

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

from aggregate_results import checkpoint_image_path
from make_paper_table import _dominant_epoch, _model_dir, _parse_sample_metrics_full
from generate_figure import (
    LABEL_MAPS,
    REPO_ROOT,
    RESULTS_DIR,
    SRC,
    TP_PATCH_CRITERION,
    TP_PATCH_MAX_FN_RATE,
    TP_PATCH_MIN_FG,
    TP_PATCH_SIZE,
    compute_top_tp_patches,
)

ABLATION_PATHS_JSON = os.path.join(
    REPO_ROOT, "output", "summary", "summary_experiment_means_ablation_paths.json")
GROUP = "ablation_table"
FOLD = "fold_0"

CELL = 165  # rendered size of one magnified crop, in px
PAD = 10
LABEL_H = 40  # caption strip under each crop, two text lines tall
HEAD_H = 30
ROW_LABEL_W = 165
BOX_COLORS = ("#ff0000", "#008000")


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _plain_label(label):
    """LABEL_MAPS labels are LaTeX; this sheet is drawn with PIL."""
    import re
    return re.sub(r"\\(mathcal|mathbf|text|textbf)\{([^{}]*)\}", r"\2", label).replace("$", "").strip()


def load_options(dataset):
    """The figure's option runs, in figure (LABEL_MAPS) order, plus the index
    of the highlighted one -- the row whose experiment the ablation-paths JSON
    also lists as "proposed"."""
    with open(ABLATION_PATHS_JSON) as f:
        all_paths = json.load(f)
    key = dataset.lower().replace("-", "").replace(" ", "")
    paths = next(v for k, v in all_paths.items()
                 if k.lower().replace("-", "").replace(" ", "") == key)[GROUP]

    options = []
    for row_key, label in LABEL_MAPS[GROUP].items():
        if row_key not in paths:
            continue
        mdir = _model_dir(paths[row_key]["path"], FOLD)
        rows = _parse_sample_metrics_full(os.path.join(mdir, "test_sample_metrics.csv"))
        epoch = _dominant_epoch(rows)
        options.append(dict(
            label=_plain_label(label),
            experiment=paths[row_key]["experiment"],
            model_dir=mdir,
            epoch=epoch,
            rows={r["sample_name"]: r for r in rows if int(r["epoch"]) == epoch},
        ))
    best_index = next(i for i, o in enumerate(options)
                      if o["experiment"] == paths["proposed"]["experiment"])
    return options, best_index


def sample_paths(options, best_index, sample_id):
    """(gt_path, [pred_path per option]) for one test sample."""
    best = options[best_index]
    gt = checkpoint_image_path(best["model_dir"], best["epoch"], "mask",
                               int(best["rows"][sample_id]["batch"]))
    preds = [checkpoint_image_path(o["model_dir"], o["epoch"], "pred",
                                   int(o["rows"][sample_id]["batch"])) for o in options]
    return gt, preds


def _binary(path):
    img = PILImage.open(path).convert("L")
    if img.size != (SRC, SRC):
        img = img.resize((SRC, SRC), PILImage.Resampling.NEAREST)
    return np.array(img) > 127


def box_stats(gt, preds, box):
    """Per-patch fg / (dice, FN rate) per option, inside `box` only."""
    x, y, s = box
    g = gt[y:y + s, x:x + s]
    gt_fg = max(float(g.sum()), 1.0)
    stats = []
    for p in preds:
        pp = p[y:y + s, x:x + s]
        tp = float((g & pp).sum())
        stats.append((2 * tp / max(float(g.sum() + pp.sum()), 1.0), (gt_fg - tp) / gt_fg))
    return g.mean(), stats


def evaluate(options, best_index, sample_id, patches, criterion, min_fg, max_fn_rate):
    """Search this sample's crop pair and score it by the weaker crop's FN gap.
    Returns None when the sample cannot fill every crop under the constraints."""
    gt_path, pred_paths = sample_paths(options, best_index, sample_id)
    boxes = compute_top_tp_patches(gt_path, pred_paths, best_index, count=patches,
                                   criterion=criterion, min_fg=min_fg, max_fn_rate=max_fn_rate)
    if len(boxes) < patches:
        return None

    gt, preds = _binary(gt_path), [_binary(p) for p in pred_paths]
    rows = [box_stats(gt, preds, b) for b in boxes]
    gaps = [min(fn for i, (_, fn) in enumerate(st) if i != best_index) - st[best_index][1]
            for _, st in rows]
    return dict(sample=sample_id, gt=gt, preds=preds, gt_path=gt_path, boxes=boxes,
                rows=rows, gaps=gaps, weakest=min(gaps),
                dices=[o["rows"][sample_id]["dice"] for o in options])


def build_sheet(dataset, results, options, best_index, patches, out_name):
    ncols = 1 + 1 + len(options)  # full GT + GT crop + one crop per option
    w = ROW_LABEL_W + ncols * (CELL + PAD) + PAD
    h = HEAD_H + len(results) * patches * (CELL + LABEL_H + PAD) + PAD
    sheet = PILImage.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    f_small, f_head = _font(13), _font(15)

    titles = ["GT (full)", "GT"] + [o["label"] + (" *" if i == best_index else "")
                                    for i, o in enumerate(options)]
    for ci, title in enumerate(titles):
        draw.text((ROW_LABEL_W + ci * (CELL + PAD), 6), title, fill="black", font=f_head)

    scale = CELL / SRC
    row = 0
    for res in results:
        full = PILImage.open(res["gt_path"]).convert("RGB").resize(
            (CELL, CELL), PILImage.Resampling.BILINEAR)
        for bi, (x, y, s) in enumerate(res["boxes"]):
            ImageDraw.Draw(full).rectangle(
                [x * scale, y * scale, (x + s) * scale, (y + s) * scale],
                outline=BOX_COLORS[bi % len(BOX_COLORS)], width=2)

        for bi, box in enumerate(res["boxes"]):
            top = HEAD_H + row * (CELL + LABEL_H + PAD)
            color = BOX_COLORS[bi % len(BOX_COLORS)]
            if bi == 0:
                draw.text((6, top + 4), "sample {}".format(res["sample"]), fill="black", font=f_head)
                draw.text((6, top + 24), "dice {}".format(
                    "/".join("{:.3f}".format(d) for d in res["dices"])), fill="black", font=f_small)
                draw.text((6, top + 40), "weakest gap {:.3f}".format(res["weakest"]),
                          fill="black", font=f_small)
                sheet.paste(full, (ROW_LABEL_W, top))
            draw.text((6, top + 60 if bi == 0 else top + 4),
                      "box{} ({},{}) gap {:.3f}".format(bi + 1, box[0], box[1], res["gaps"][bi]),
                      fill=color, font=f_small)

            fg, stats = res["rows"][bi]
            cells = [(res["gt"], "fg {:.3f}".format(fg))] + [
                (p, "dice {:.3f}\nFN {:.3f}".format(*st)) for p, st in zip(res["preds"], stats)]
            for ci, (mask, caption) in enumerate(cells):
                left = ROW_LABEL_W + (ci + 1) * (CELL + PAD)
                crop = PILImage.fromarray((mask[box[1]:box[1] + box[2],
                                                box[0]:box[0] + box[2]] * 255).astype(np.uint8))
                sheet.paste(crop.convert("RGB").resize((CELL, CELL), PILImage.Resampling.NEAREST),
                            (left, top))
                draw.rectangle([left, top, left + CELL - 1, top + CELL - 1], outline=color, width=2)
                for li, line in enumerate(caption.split("\n")):
                    draw.text((left, top + CELL + 4 + li * 16), line, fill="black", font=f_small)
            row += 1

    out = os.path.join(RESULTS_DIR, "tmp", out_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    print("wrote", out)
    return out


def _pop_option(args, flag, default, cast=str):
    if flag not in args:
        return args, default
    i = args.index(flag)
    return args[:i] + args[i + 2:], cast(args[i + 1])


def main(argv):
    args = argv[1:]
    args, dataset = _pop_option(args, "--dataset", "DRIVE")
    args, patches = _pop_option(args, "--patches", 2, int)
    args, top = _pop_option(args, "--top", 6, int)
    args, criterion = _pop_option(args, "--criterion", TP_PATCH_CRITERION)
    args, min_fg = _pop_option(args, "--min-fg", TP_PATCH_MIN_FG, float)
    args, max_fn = _pop_option(args, "--max-fn", TP_PATCH_MAX_FN_RATE, float)

    options, best_index = load_options(dataset)
    samples = args or sorted(set.intersection(*[set(o["rows"]) for o in options]))

    results, skipped = [], []
    for sid in samples:
        # A crop pair can only argue the table's ordering if the whole image
        # agrees with it, so samples that invert it are not worth rendering.
        dices = [o["rows"][sid]["dice"] for o in options]
        if sorted(dices) != dices:
            skipped.append((sid, "whole-image dice out of order"))
            continue
        res = evaluate(options, best_index, sid, patches, criterion, min_fg, max_fn)
        if res is None:
            skipped.append((sid, "fewer than {} qualifying patches".format(patches)))
            continue
        results.append(res)

    results.sort(key=lambda r: -r["weakest"])
    print("\n{:<8} {:<24} {:<10} {}".format("sample", "dice " + "/".join(
        o["label"] for o in options), "weak gap", "boxes"))
    for r in results:
        print("{:<8} {:<24} {:<10} {}".format(
            r["sample"], "/".join("{:.3f}".format(d) for d in r["dices"]),
            "{:.3f}".format(r["weakest"]),
            "  ".join("({},{}) gap {:.3f}".format(b[0], b[1], g)
                      for b, g in zip(r["boxes"], r["gaps"]))))
    for sid, why in skipped:
        print("{:<8} skipped: {}".format(sid, why))

    if results:
        build_sheet(dataset, results[:top], options, best_index, patches,
                    "sample_candidates_{}_{}.png".format(dataset.lower(), GROUP))


if __name__ == "__main__":
    main(sys.argv)
