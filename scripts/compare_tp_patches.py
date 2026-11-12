#!/usr/bin/env python3
"""Eyeball the ROI-box candidates the ablation figures search for.

scripts/generate_figure.py picks the crops of a TP_PATCH_GROUPS figure by
ranking patches on how much GT foreground the highlighted option recovers that
every other option misses (see compute_top_tp_patches). This script renders the
whole ranked candidate list -- one row per candidate, one column per panel (GT
plus each compared option) -- so the two ranks wired into TP_PATCH_PICKS can be
chosen by looking rather than by trusting the score.

Per-patch Dice, recall and FN rate (against GT, inside the patch only) are
printed under each crop, and the highlighted option's column is marked with a
star.

Usage:
    uv run scripts/compare_tp_patches.py [--criterion tp_gain|fn_rate] [--count N]
                                        [--min-fg F] [--max-fn F] [group_key ...]
"""
import os
import re
import sys

from PIL import Image as PILImage, ImageDraw, ImageFont

from generate_figure import (
    GROUPS,
    PAPER_TABLES_DIR,
    RESULTS_DIR,
    SRC,
    TP_PATCH_CANDIDATES,
    TP_PATCH_CRITERION,
    TP_PATCH_KEEP,
    TP_PATCH_MAX_FN_RATE,
    TP_PATCH_MIN_FG,
    TP_PATCH_PICKS,
    _load_binary,
    ablation_raw_panels,
    load_ablation_json,
    search_option_gap_patches,
)

CELL = 190  # rendered size of one magnified crop, in px
PAD = 10
LABEL_H = 40  # caption strip under each crop, two text lines tall
HEAD_H = 30  # column-header strip
ROW_LABEL_W = 150  # left gutter holding the candidate rank / coordinates
PICK_COLORS = ("#ff0000", "#008000")  # matches generate_figure.BOX_COLORS


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def patch_stats(gt, pred, box):
    x, y, s = box
    g = gt[y:y + s, x:x + s]
    p = pred[y:y + s, x:x + s]
    tp = float((g & p).sum())
    gt_fg = max(float(g.sum()), 1.0)
    dice = 2 * tp / max(float(g.sum() + p.sum()), 1.0)
    return dice, tp / gt_fg, (gt_fg - tp) / gt_fg  # dice, recall, FN rate


def crop_image(path, box):
    x, y, s = box
    img = PILImage.open(path).convert("RGB")
    if img.size != (SRC, SRC):
        img = img.resize((SRC, SRC), PILImage.Resampling.BILINEAR)
    return img.crop((x, y, x + s, y + s)).resize((CELL, CELL), PILImage.Resampling.NEAREST)


def build_sheet(group_key, json_path, criterion=TP_PATCH_CRITERION, count=TP_PATCH_CANDIDATES,
                **search_kwargs):
    data = load_ablation_json(json_path)
    raw_panels = ablation_raw_panels(group_key, data, no_image=True)
    boxes = search_option_gap_patches(data, raw_panels, count=count, criterion=criterion,
                                      **search_kwargs)
    kept = {(b[0], b[1]) for b in TP_PATCH_KEEP}

    # Columns: GT first, then every option panel, in figure order.
    cols = [p for p in raw_panels if p["label"] == "GT"] + [p for p in raw_panels if p["metrics"]]
    col_titles = [_plain_label(p["label"]) + (" *" if p["highlighted"] else "") for p in cols]
    masks = {p["img"]: _load_binary(os.path.join(PAPER_TABLES_DIR, p["img"])) for p in cols}
    gt_mask = masks[next(p["img"] for p in cols if p["metrics"] is None)]

    w = ROW_LABEL_W + len(cols) * (CELL + PAD) + PAD
    h = HEAD_H + len(boxes) * (CELL + LABEL_H + PAD) + PAD
    sheet = PILImage.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    f_small, f_head = _font(14), _font(16)

    head = "criterion: " + criterion
    if criterion == "fn_rate":
        head += "  fg>={:.2f} FN<={:.2f}".format(
            search_kwargs.get("min_fg", TP_PATCH_MIN_FG),
            search_kwargs.get("max_fn_rate", TP_PATCH_MAX_FN_RATE))
    draw.text((6, 6), head, fill="black", font=f_small)
    for ci, title in enumerate(col_titles):
        draw.text((ROW_LABEL_W + ci * (CELL + PAD), 6), title, fill="black", font=f_head)

    # The picked-box highlight only means anything when this sheet ranks the
    # way the renderer does; a different criterion or relaxed thresholds shift
    # every rank, so TP_PATCH_PICKS would point at the wrong rows.
    same_search = (criterion == TP_PATCH_CRITERION
                   and search_kwargs.get("min_fg", TP_PATCH_MIN_FG) == TP_PATCH_MIN_FG
                   and search_kwargs.get("max_fn_rate", TP_PATCH_MAX_FN_RATE) == TP_PATCH_MAX_FN_RATE)
    picks = TP_PATCH_PICKS if same_search else ()
    for ri, box in enumerate(boxes):
        top = HEAD_H + ri * (CELL + LABEL_H + PAD)
        pick = picks.index(ri) if ri in picks else None
        color = PICK_COLORS[pick] if pick is not None else "black"
        tag = "(picked)" if pick is not None else ""
        if (box[0], box[1]) in kept:
            tag = ("kept " + tag).strip()
        draw.text((6, top + 4), "#{} {}".format(ri, tag), fill=color, font=f_head)
        draw.text((6, top + 26), "x={} y={} s={}".format(*box), fill="black", font=f_small)

        for ci, p in enumerate(cols):
            left = ROW_LABEL_W + ci * (CELL + PAD)
            sheet.paste(crop_image(os.path.join(PAPER_TABLES_DIR, p["img"]), box), (left, top))
            if pick is not None:
                draw.rectangle([left, top, left + CELL - 1, top + CELL - 1], outline=color, width=3)
            if p["metrics"] is None:
                fg = masks[p["img"]][box[1]:box[1] + box[2], box[0]:box[0] + box[2]].mean()
                lines = ["fg {:.3f}".format(fg)]
            else:
                dice, recall, fn_rate = patch_stats(gt_mask, masks[p["img"]], box)
                # Two lines: one caption on a single line would run under the
                # next column's crop.
                lines = ["dice {:.3f}  rec {:.3f}".format(dice, recall), "FN {:.3f}".format(fn_rate)]
            for li, line in enumerate(lines):
                draw.text((left, top + CELL + 4 + li * 16), line, fill="black", font=f_small)

    out = os.path.join(RESULTS_DIR, "tmp",
                       "tp_patch_candidates_{}_{}_{}.png".format(group_key, data["id"], criterion))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    print("wrote", out)
    return out


def _plain_label(label):
    """LABEL_MAPS labels are LaTeX ("$\\mathcal{D}^{\\text{inv}}$"); this sheet
    is drawn with PIL, so reduce them to plain text ("D^inv")."""
    out = re.sub(r"\\(mathcal|mathbf|text|textbf)\{([^{}]*)\}", r"\2", label)
    return out.replace("$", "").replace("\xa0", " ").strip()


def _pop_option(args, flag, default, cast=str):
    if flag not in args:
        return args, default
    i = args.index(flag)
    return args[:i] + args[i + 2:], cast(args[i + 1])


def main(argv):
    args = argv[1:]
    args, criterion = _pop_option(args, "--criterion", TP_PATCH_CRITERION)
    args, count = _pop_option(args, "--count", TP_PATCH_CANDIDATES, int)
    # "fn_rate" only; relaxing these surfaces more (weaker) candidates.
    args, min_fg = _pop_option(args, "--min-fg", TP_PATCH_MIN_FG, float)
    args, max_fn = _pop_option(args, "--max-fn", TP_PATCH_MAX_FN_RATE, float)
    for key in args or ["ablation_table"]:
        # A group can be built from several samples stacked in one figure.
        paths = GROUPS[key]["drive_json"]
        for path in ([paths] if isinstance(paths, str) else paths):
            build_sheet(key, path, criterion=criterion, count=count,
                        min_fg=min_fg, max_fn_rate=max_fn)


if __name__ == "__main__":
    main(sys.argv)
