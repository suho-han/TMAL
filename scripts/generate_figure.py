#!/usr/bin/env python3
"""Render the paper's qualitative HTML figures to PDF/PNG.

Two families of figures are produced from the panels/JSON that
scripts/make_paper_table.py writes under output/paper_tables/figures/:

1. Model comparison (see PANELS / DATASETS) — per dataset, Image and GT
   (unlettered) stacked in their own first column, split off by a vertical
   rule from a 5x2 grid of model panels lettered alphabetically. Each panel
   shows the full 512x512 image/mask with two ROI boxes (red, green) drawn on
   it plus two matching zoomed crops (each 50% of the main-image size): the
   "below" layout puts them in a row underneath, the "inset" layout attaches
   them to the right, stacked top/bottom. Prediction panels overlay their
   Dice / beta_0 error in yellow at the bottom-right.
   Also renders the 2x2 DRIVE distance-map figure (see DISTANCE_PANELS).
2. Ablation (see GROUPS / LABEL_MAPS) — one figure per ablation group, with
   the row labels typeset as LaTeX via KaTeX.

Outputs (HTML + PDF in results/tmp/, PNG in results/) stay under
output/paper_tables/, unchanged from when this script lived there.

Usage:
    uv run scripts/generate_figure.py
"""
import glob
import json
import math
import os
import subprocess

import matplotlib
import numpy as np
from PIL import Image as PILImage
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PAPER_TABLES_DIR = os.path.join(REPO_ROOT, "output", "paper_tables")
FIGURES_DIR = os.path.join(PAPER_TABLES_DIR, "figures")
RESULTS_DIR = os.path.join(PAPER_TABLES_DIR, "results")
TMP_DIR = os.path.join(RESULTS_DIR, "tmp")

# The generated HTML lives in results/tmp/, two levels below paper_tables/.
FIG_REL = "../../figures/"

# ── Shared geometry ──────────────────────────────────────────────────────────

MAIN_W = 260  # display width of the main (512x512) image, in px
SRC = 512  # source image size in px
SCALE = MAIN_W / SRC
CROP_W = MAIN_W / 2  # each crop at 50% of the main-image size
TOTAL_W = MAIN_W + CROP_W  # main image + side crop column
CROP_ZOOM = 2.0  # saved crop.jpg resolution = box size * CROP_ZOOM

# Size of the yellow "dice / beta_0 error" readout overlaid on each prediction
# panel; shared so the comparison and ablation figures stay consistent.
METRIC_FONT_SIZE = 26

# Panel caption ("(a) $\\mathcal{S}_1$", ...) size in the ablation figures, and
# the fixed strip reserved for it. The strip has to clear the tallest label --
# KaTeX subscripts like $\\mathcal{A}_{S_{1,2}}$ run well past the font size --
# so it is kept proportional to the font size rather than pinned in px.
ABLATION_TITLE_FONT_SIZE = 34
ABLATION_TITLE_H = round(ABLATION_TITLE_FONT_SIZE * 2.4)

# crop1 (first ROI box) -> red, crop2 (second ROI box) -> green; used for both
# the ROI outline drawn on the main image and the matching crop's border.
BOX_COLORS = ("#ff0000", "#008000")

# ROI boxes (x, y, size) in SRCxSRC source-image pixels, keyed by sample id and
# shared between the comparison and ablation figures so their crop insets line
# up. Ids handled in arrow mode ignore this (see compute_arrow_target).
DATASET_BOXES = {
    "19": [(200, 180, 60), (200, 270, 60)],          # drive (comparison)
    # "14" (drive / ablation_table) has no entry: its boxes are searched at
    # render time instead, see TP_PATCH_GROUPS / compute_top_tp_patches.
    "04": [(10, 300, 75), (430, 110, 75)],           # drive (s_variation) -- patch-search: largest proposed-vs-worst TP recall gap
    "06": [(10, 130, 70), (340, 210, 70)],           # drive (imo_variation) -- patch-search: largest proposed-vs-worst TP recall gap
    "11269-5": [(256, 224, 100), (128, 224, 100)],   # deepcrack (unused when arrow mode is active)
}

# Arrow-mode datasets draw a single red arrow instead of the two ROI boxes +
# magnified crops -- pointing at the highest-foreground-TP ARROW_PATCH_SIZE-
# square patch (GT vs. the proposed/highlighted prediction), matching
# notebooks/paper_figures.ipynb's DeepCrack cell, whose ARROW_ANGLE_DEG /
# ARROW_LENGTH these match.
ARROW_PATCH_SIZE = 100
ARROW_ANGLE_DEG = 45  # 0 = tail directly right of tip, 90 = tail directly above
ARROW_LENGTH = 70     # source-pixel tail-to-tip distance
ARROW_COLOR = "#ff0000"

# Ablation groups whose ROI boxes are searched at render time (see
# compute_top_tp_patches) instead of being read from DATASET_BOXES, so the
# magnified crops always sit where the highlighted option recovers foreground
# the other options miss.
TP_PATCH_GROUPS = {"ablation_table"}
TP_PATCH_SIZE = 75  # source-pixel side of a searched ROI box
TP_PATCH_CANDIDATES = 10  # how many ranked candidates the search returns
# Keep only windows whose per-patch Dice rises strictly along the figure's
# option order, so a crop never shows the highlighted option scoring worse
# than the options it is supposed to beat.
TP_PATCH_REQUIRE_DICE_ORDER = True
# How candidates are ranked (see compute_top_tp_patches):
#   "tp_gain" -- by TP count recovered over the hardest rival. Note this is the
#                same ordering as ranking by FN count, since FN = GT_fg - TP and
#                GT_fg is shared by every option in a given window.
#   "fn_rate" -- by *missed fraction* of the patch's GT instead, which is what
#                actually differs from "tp_gain": it ignores how much vessel a
#                window holds and asks how much of it each option drops, so
#                thin-vessel windows the rivals fragment can outrank dense ones.
TP_PATCH_CRITERION = "fn_rate"
# Final ROI boxes per group and sample id, overriding the search for that
# sample (the search still backs any sample not listed here). Keyed by group
# too, since the same sample carries one box in the two-sample layout and two
# in the one-sample one.
TP_PATCH_BOXES = {
    # Each box is the sample's own fn_rate rank-0 window nudged 10px up by
    # hand -- both stay within the Dice-order constraint there, at a small
    # cost in FN gap (02: 0.115 -> 0.096, 07: 0.098 -> 0.089).
    "ablation_table": {
        "02": [(261, 21, TP_PATCH_SIZE)],
        "07": [(251, 153, TP_PATCH_SIZE)],
    },
    # Same nudged rank-0 box, plus the sample's rank-1 window as the second
    # crop -- shifted 15px right, where the highlighted option's own error is
    # lowest (FN+FP 0.307 -> 0.220 of the patch's GT, FP 0.137 -> 0.088) while
    # the rivals keep degrading (Y stays at FN 0.47).
    "ablation_table_1x2_02": {
        "02": [(261, 21, TP_PATCH_SIZE), (50, 306, TP_PATCH_SIZE)],
    },
    # The best-balanced pair of any DRIVE sample (gaps 0.099 / 0.095), both
    # straight from the search.
    "ablation_table_1x2_08": {
        "08": [(81, 25, TP_PATCH_SIZE), (318, 120, TP_PATCH_SIZE)],
    },
}
# Boxes pinned by hand for a *searched* sample: they head the candidate list
# (so rank 0, 1, ... are these) and seed the search's suppression, so the
# remaining ranks are elsewhere in the image. Coordinates are sample-specific.
TP_PATCH_KEEP = []
# "fn_rate" only: windows must hold at least this fraction of GT foreground
# (otherwise the rate is computed off a handful of pixels), and the highlighted
# option must itself miss no more than this fraction of the patch's GT. Tight
# values (0.10 / 0.15) leave only 7 qualifying windows on DRIVE 14; these are
# loosened enough to give a real shortlist to choose a partner crop from.
TP_PATCH_MIN_FG = 0.07
TP_PATCH_MAX_FN_RATE = 0.20
# Which two candidates (0-based rank) are drawn as the red / green ROI. Change
# these after eyeballing `uv run scripts/compare_tp_patches.py`, which renders
# all TP_PATCH_CANDIDATES side by side.
TP_PATCH_PICKS = (0, 1)


def ensure_crop(src_abs, box, suffix):
    """Return the HTML-relative src of a magnified ROI crop of `src_abs`,
    (re)generating it from `box` every time so a changed ROI position/size can
    never be masked by a stale cached file that merely happens to match the
    expected output pixel size. The source is resized to SRCxSRC first if it
    isn't already, since `box` is defined in that coordinate space."""
    crop_name = "{}{}.jpg".format(os.path.splitext(os.path.basename(src_abs))[0], suffix)
    crop_abs = os.path.join(FIGURES_DIR, crop_name)
    x, y, size = box
    img = PILImage.open(src_abs)
    if img.size != (SRC, SRC):
        img = img.resize((SRC, SRC), PILImage.Resampling.BILINEAR)
    out_size = round(size * CROP_ZOOM)
    crop = img.crop((x, y, x + size, y + size))
    crop = crop.resize((out_size, out_size), PILImage.Resampling.BILINEAR).convert("RGB")
    crop.save(crop_abs)
    print("wrote", crop_abs)
    return FIG_REL + crop_name


def box_divs(boxes, flip=False):
    out = []
    for i, (x, y, s) in enumerate(boxes):
        if flip:  # image displayed flipped 180deg -> point-reflect the ROI
            x, y = SRC - x - s, SRC - y - s
        out.append(
            '<div class="roi" style="left:{l:.1f}px;top:{t:.1f}px;'
            'width:{w:.1f}px;height:{w:.1f}px;border-color:{c};"></div>'.format(
                l=x * SCALE, t=y * SCALE, w=s * SCALE, c=BOX_COLORS[i]
            )
        )
    return "".join(out)


def _load_binary(path):
    """Load a saved mask/prediction JPG as a boolean SRCxSRC foreground map."""
    img = PILImage.open(path).convert("L")
    if img.size != (SRC, SRC):
        img = img.resize((SRC, SRC), PILImage.Resampling.NEAREST)
    return np.array(img) > 127


def _window_sums(mask, ps):
    """Sum of `mask` over every ps-square window, via an integral image.
    Result[y, x] is the sum of the window whose top-left corner is (x, y)."""
    integral = np.pad(np.cumsum(np.cumsum(mask.astype(np.float64), axis=0), axis=1), ((1, 0), (1, 0)))
    return integral[ps:, ps:] - integral[:-ps, ps:] - integral[ps:, :-ps] + integral[:-ps, :-ps]


def compute_arrow_target(gt_abs, pred_abs):
    """Highest-foreground-TP ARROW_PATCH_SIZE-square patch center, in source
    (SRCxSRC) pixel space -- slides the patch over the full image and picks the
    window where GT and the given prediction are both foreground the most."""
    patch_sum = _window_sums(_load_binary(gt_abs) & _load_binary(pred_abs), ARROW_PATCH_SIZE)
    y0, x0 = np.unravel_index(np.argmax(patch_sum), patch_sum.shape)
    return x0 + ARROW_PATCH_SIZE / 2.0, y0 + ARROW_PATCH_SIZE / 2.0


def _window_dice(gt, pred, size, gt_fg):
    """Per-window Dice of `pred` against `gt`, over every ps-square window."""
    tp = _window_sums(gt & pred, size)
    return tp, 2 * tp / np.maximum(gt_fg + _window_sums(pred, size), 1.0)


def compute_top_tp_patches(gt_abs, pred_abses, best_index, size=TP_PATCH_SIZE, count=2,
                           require_dice_order=TP_PATCH_REQUIRE_DICE_ORDER,
                           criterion=TP_PATCH_CRITERION, keep=(),
                           min_fg=TP_PATCH_MIN_FG, max_fn_rate=TP_PATCH_MAX_FN_RATE):
    """`count` non-overlapping (x, y, size) ROI boxes in source (SRCxSRC) pixel
    space, ranked so the magnified crops show the highlighted option at its
    most convincing: windows holding GT foreground that `pred_abses[best_index]`
    recovers and *every* other option misses.

    Under the "tp_gain" `criterion` a window scores the highlighted option's
    foreground-TP count minus the best (i.e. hardest-to-beat) competitor's, so
    a window only ranks high when all the other options degrade there -- the
    earlier mean-TP score instead favoured windows every option got right,
    where the panels look identical. Ties are broken toward more GT foreground.
    Under "fn_rate" the same comparison is made on the fraction of the patch's
    GT each option misses, restricted to windows with at least TP_PATCH_MIN_FG
    foreground where the highlighted option itself misses at most
    TP_PATCH_MAX_FN_RATE of it.

    Either way each pick suppresses one full patch size around it so the boxes
    cover distinct regions. Boxes passed in `keep` are hand-picked off an
    earlier candidate sheet: they head the returned list unconditionally
    (bypassing the ranking) and seed that suppression, so every searched rank
    after them sits elsewhere in the image.

    With `require_dice_order`, windows are additionally restricted to those
    whose per-patch Dice increases strictly along `pred_abses`' figure order
    (so for ablation_table: Dice(Y) < Dice(D) < Dice(D_inv)). TP gain alone can
    rank a window where the highlighted option recovers the most vessel yet
    still scores the *worst* patch Dice, because the extra foreground it paints
    outside GT costs more than the recovered TP gains -- a crop that argues
    against the very ordering the table reports.
    """
    gt = _load_binary(gt_abs)
    gt_fg = _window_sums(gt, size)
    tps, dices = zip(*(_window_dice(gt, _load_binary(p), size, gt_fg) for p in pred_abses))

    rival_tp = np.maximum.reduce([tp for i, tp in enumerate(tps) if i != best_index])
    fn_rates = [(gt_fg - tp) / np.maximum(gt_fg, 1.0) for tp in tps]
    rival_fn_rate = np.minimum.reduce([f for i, f in enumerate(fn_rates) if i != best_index])

    if criterion == "tp_gain":
        score = tps[best_index] - rival_tp + 1e-3 * gt_fg
    elif criterion == "fn_rate":
        eligible = (gt_fg >= min_fg * size * size) & (fn_rates[best_index] <= max_fn_rate)
        score = np.where(eligible, rival_fn_rate - fn_rates[best_index], -np.inf)
    else:
        raise ValueError("unknown TP patch criterion: {!r}".format(criterion))
    if require_dice_order:
        ordered = np.logical_and.reduce([a < b for a, b in zip(dices, dices[1:])])
        score = np.where(ordered, score, -np.inf)

    def report(x, y, tag):
        print("tp-patch[{}]: {} box ({}, {}, {}) TP gain {:+.4f} of patch, GT fg {:.3f}, "
              "FN rate {}, dice {}".format(
                  criterion, tag, x, y, size,
                  (tps[best_index][y, x] - rival_tp[y, x]) / (size * size), gt_fg[y, x] / (size * size),
                  "/".join("{:.3f}".format(f[y, x]) for f in fn_rates),
                  "/".join("{:.3f}".format(d[y, x]) for d in dices)))

    def suppress(x, y):
        score[max(0, y - size):y + size, max(0, x - size):x + size] = -np.inf

    boxes = []
    for x, y, _ in keep:
        boxes.append((x, y, size))
        report(x, y, "kept")
        suppress(x, y)

    while len(boxes) < count:
        y, x = np.unravel_index(np.argmax(score), score.shape)
        if not np.isfinite(score[y, x]):
            print("tp-patch[{}]: only {} candidate(s) satisfy the search constraints".format(
                criterion, len(boxes)))
            break
        boxes.append((int(x), int(y), size))
        report(x, y, "rank {}".format(len(boxes) - 1))
        suppress(x, y)
    return boxes


def search_option_gap_patches(data, raw_panels, count=TP_PATCH_CANDIDATES,
                              criterion=TP_PATCH_CRITERION, keep=None, **search_kwargs):
    """ROI-box candidates for one ablation figure: TP_PATCH_KEEP's hand-picked
    boxes first, then ranked windows where the highlighted option beats the
    other option panels the figure shows (i.e. the ones carrying metrics), in
    figure order. Shared by the renderer and scripts/compare_tp_patches.py so
    both rank the same way."""
    preds = [p for p in raw_panels if p["metrics"]]
    return compute_top_tp_patches(
        os.path.join(PAPER_TABLES_DIR, data["gt"]),
        [os.path.join(PAPER_TABLES_DIR, p["img"]) for p in preds],
        next(i for i, p in enumerate(preds) if p["highlighted"]),
        count=count,
        criterion=criterion,
        keep=TP_PATCH_KEEP if keep is None else keep,
        **search_kwargs,
    )


def _arrow_tail(tip_x, tip_y, angle_deg=ARROW_ANGLE_DEG, length=ARROW_LENGTH):
    """Tail point up-and-left of (tip_x, tip_y) at a fixed angle/length, so
    every arrow has the same look regardless of where the target lands."""
    theta = math.radians(angle_deg)
    return tip_x - length * math.cos(theta), tip_y - length * math.sin(theta)


def arrow_svg(target, uid, flip=False):
    tip_x, tip_y = target
    if flip:  # image displayed flipped 180deg -> point-reflect the target
        tip_x, tip_y = SRC - tip_x, SRC - tip_y
    tail_x, tail_y = _arrow_tail(tip_x, tip_y)
    marker_id = "arrowhead-{}".format(uid)
    return (
        '<svg class="arrow" viewBox="0 0 {src} {src}" preserveAspectRatio="none">'
        '<defs><marker id="{mid}" markerWidth="8" markerHeight="8" refX="6" refY="4" '
        'orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="{color}"/></marker></defs>'
        '<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        'stroke="{color}" stroke-width="4" marker-end="url(#{mid})"/>'
        "</svg>"
    ).format(src=SRC, mid=marker_id, x1=tail_x, y1=tail_y, x2=tip_x, y2=tip_y, color=ARROW_COLOR)


PRINT_CSS = """
@media print {
  body {
    padding: 0;
  }
  .figure {
    margin: 0;
    zoom: 1;
    break-inside: avoid;
    break-after: page;
    page-break-after: always;
  }
  .figure:last-child {
    break-after: auto;
    page-break-after: auto;
  }
}
"""


def wrap_html(body, title, css):
    """Wrap a figure body in a document, with KaTeX loaded so LaTeX labels
    like $\\mathcal{A}_{S_1}$ are typeset at render time."""
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>{title}</title>\n"
        "<style>{css}</style>\n"
        "<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css\">\n"
        "<script src=\"https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js\"></script>\n"
        "<script src=\"https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js\"></script>\n"
        "<script>document.addEventListener('DOMContentLoaded', function() {{ renderMathInElement(document.body, {{delimiters: [{{left: '$', right: '$', display: false}}]}}); }});</script>\n"
        "</head>\n<body>\n{body}\n</body>\n</html>\n"
    ).format(title=title, css=css, body=body)


# ── html -> pdf -> png rendering ─────────────────────────────────────────────
#
# Each HTML report is rendered with a real browser engine (Playwright +
# Chromium); the actual rendered size of each ".figure" section under print
# media is measured and an exact @page size injected for it (see
# _fit_page_to_figures) so PDF pages hug the figure with no leftover
# whitespace or clipping. Each PDF page is then rasterized with pdftoppm.

PNG_DPI = 192  # 2x the 96 CSS-px/inch baseline the @page sizes assume
PAGE_MARGIN = 12  # px of slack around each figure to avoid edge clipping
SIZE_SLACK = 12  # px added to the measured size: getBoundingClientRect() and
# Chromium's PDF page-break layout round sub-pixel heights slightly
# differently, so an exact-fit page can spill one row onto an extra page.


def render_html_files(html_paths, results_dir, dpi=PNG_DPI):
    """Render each given HTML file to a PDF (kept alongside it) and then
    rasterize it to PNG page(s) written into `results_dir`.
    """
    if not html_paths:
        return

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for html_path in html_paths:
            stem = os.path.splitext(os.path.basename(html_path))[0]
            pdf_path = os.path.join(os.path.dirname(html_path), stem + ".pdf")

            page.goto("file://" + html_path)
            page.wait_for_load_state("networkidle")
            page.emulate_media(media="print")
            _fit_page_to_figures(page)
            page.pdf(path=pdf_path, print_background=True, prefer_css_page_size=True)
            print("wrote", pdf_path)

            _pdf_to_png(pdf_path, results_dir, stem, dpi)
        browser.close()


def _fit_page_to_figures(page):
    """Inject @page rule(s) sized to fit each ".figure" section (plus
    PAGE_MARGIN slack), so the PDF page has no leftover whitespace or
    clipping regardless of panel counts / row layout.

    A ".figure"'s own getBoundingClientRect() only covers panels and gaps —
    a caption centered under a fixed-width panel with `white-space: nowrap`
    can render wider than the panel, overflowing past the figure's own box on
    the outer edge of the first/last column.

    Growing the *page* to fit that overflow doesn't work: Chromium's print
    pagination clips content at the body element's own coordinate origin,
    not at the page's outer edge, so anything with a negative offset
    relative to body (i.e. poking out past its left/top edge) is dropped
    regardless of how much @page margin surrounds it (verified empirically —
    even an 80px margin left the clip untouched). The only fix is to give
    `body` real left/right padding (in-flow space, so nothing sits at a
    negative coordinate relative to it) sized to the worst-case overflow
    across every figure in the document; @page margin stays purely cosmetic
    slack outside that already-safe box.

    Only the first figure's size can differ from the rest (via `@page
    :first`), which covers every current report: each HTML file has one or
    two ".figure" sections (one per dataset), and their sizes only ever
    differ pairwise.
    """
    sizes = page.eval_on_selector_all(
        ".figure",
        """els => els.map(fig => {
            const f = fig.getBoundingClientRect();
            let overLeft = 0, overRight = 0;
            for (const el of fig.querySelectorAll('*')) {
                // KaTeX renders a visually-hidden MathML copy for screen
                // readers (clipped via CSS `clip`, not layout) — it can
                // report a large bounding rect despite never being painted,
                // so it must not inflate the padding we size for real
                // (visible) overflow like long nowrap captions.
                if (el.closest('.katex-mathml')) continue;
                const r = el.getBoundingClientRect();
                if (r.width === 0 && r.height === 0) continue;
                overLeft = Math.max(overLeft, f.left - r.left);
                overRight = Math.max(overRight, r.right - f.right);
            }
            return [f.width, f.height, overLeft, overRight];
        })""",
    )
    if not sizes:
        return

    pad_left = math.ceil(max(s[2] for s in sizes))
    pad_right = math.ceil(max(s[3] for s in sizes))

    def page_rule(selector, w, h):
        return "@page {sel} {{ size: {pw}px {ph}px; margin: {m}px; }}".format(
            sel=selector,
            pw=math.ceil(w) + pad_left + pad_right + SIZE_SLACK + 2 * PAGE_MARGIN,
            ph=math.ceil(h) + SIZE_SLACK + 2 * PAGE_MARGIN,
            m=PAGE_MARGIN,
        )

    first_w, first_h = sizes[0][0], sizes[0][1]
    if len(sizes) == 1:
        css = page_rule("", first_w, first_h)
    else:
        rest_w, rest_h = sizes[-1][0], sizes[-1][1]
        css = page_rule(":first", first_w, first_h) + "\n" + page_rule("", rest_w, rest_h)
    css += "\nbody {{ padding-left: {l}px !important; padding-right: {r}px !important; }}".format(
        l=pad_left, r=pad_right
    )
    page.add_style_tag(content=css)


def _pdf_to_png(pdf_path, results_dir, stem, dpi):
    prefix = os.path.join(results_dir, stem)
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), pdf_path, prefix], check=True)

    # Single-page PDFs still get a "-1" suffix from pdftoppm; drop it so a
    # one-figure report yields a plain "<stem>.png".
    pages = sorted(glob.glob(prefix + "-*.png"))
    if len(pages) == 1:
        os.replace(pages[0], prefix + ".png")
        pages = [prefix + ".png"]
    for out in pages:
        print("wrote", out)


# ── Model-comparison figures ─────────────────────────────────────────────────

# Column order (matches the reference figure).
# main  : filename stem after "{id}_"
# crop  : stem used for "-crop.jpg" / "-crop2.jpg" (None => no crops)
# model : key into the JSON "models" dict (None => no caption)
PANELS = [
    dict(label="Image",           main="1_image-resized",   crop="1_image",           model=None),
    dict(label="GT",              main="2_gt-resized",       crop="2_gt",              model=None),
    dict(label="U-Net",           main="4_u-net",            crop="4_u-net",           model="U-Net"),
    dict(label="FR-UNet",         main="6_fru-net",          crop="6_fru-net",         model="FRU-Net"),
    dict(label="CS-Net",          main="7_csnet",            crop="7_csnet",           model="CSNet"),
    dict(label="V-GAN",           main="8_v-gan",            crop="8_v-gan",           model="VGAN"),
    dict(label="SegDiff",         main="9_segdiff",          crop="9_segdiff",         model="SegDiff"),
    dict(label="MedSegDiff",      main="10_medsegdiff",      crop="10_medsegdiff",     model="MedSegDiff"),
    dict(label="clDice",          main="11_cldice",          crop="11_cldice",         model="clDice"),
    dict(label="CAPE",            main="12_cape",            crop="12_cape",           model="CAPE"),
    dict(label="DconnNet",        main="13_dconnnet",        crop="13_dconnnet",       model="DconnNet"),
    dict(label="TMAL(Proposed)",      main="3_proposed",         crop="3_proposed",        model="Proposed"),
]

MODEL_COLS = 5  # Image/GT sit in their own first column, then 5x2 model panels

DATASETS = [
    dict(
        id="drive_19",
        json="drive_19.json",
        title="DRIVE — Retinal Vessel Segmentation",
        boxes=DATASET_BOXES["19"],
    ),
    dict(
        id="deepcrack_11269-5",
        json="deepcrack_11269-5.json",
        title="DeepCrack — Crack Segmentation",
        use_arrow=True,
    ),
]

# Single-row (1x4) "Image / Y / D / D_inv" figure illustrating the DRIVE
# distance-map targets (see notebooks/distance_map_drive.ipynb, which
# generates masks_dist/*_dist.npy and masks_dist_inverted/*_dist_inverted.npy
# from data/DRIVE/{split}/masks via scipy's distance_transform_edt). Reuses
# the same DRIVE test id ("19") as the drive_19 qualitative-comparison
# figure above.
DISTANCE_MAP_ID = "19"
DISTANCE_MAP_SPLIT = "test"
DISTANCE_DATA_DIR = os.path.join(REPO_ROOT, "data", "DRIVE", DISTANCE_MAP_SPLIT)

DISTANCE_PANELS = [
    dict(letter="a", label="Image"),
    dict(letter="b", label=r"$Y$"),
    dict(letter="c", label=r"$\mathcal{D}$"),
    dict(letter="d", label=r"$\bar{\mathcal{D}}$"),
]

# Panels per row: 2 gives the 2x2 grid, 4 would restore the original 1x4 strip.
DISTANCE_COLS = 2
# Fixed title strip (px) so plain-text and KaTeX labels take the same room, and
# the strut height that fixes the baseline within it (see .panel.dist in the
# CSS). The strip has to clear the strut plus the label's descenders.
DISTANCE_STRUT_H = 34
DISTANCE_TITLE_H = 46


def panel_html(ds, p, models, layout="below"):
    idp = ds["id"]
    label = p["label"]
    main_src = "{}{}_{}.jpg".format(FIG_REL, idp, p["main"])

    flip = label in ds.get("flip_panels", set())
    use_arrow = ds.get("use_arrow", False)

    boxes = arrow = ""
    c1 = c2 = None
    if p["crop"] and use_arrow:
        arrow = arrow_svg(ds["arrow_target"], uid="{}-{}".format(idp, p["crop"]), flip=flip)
    elif p["crop"]:
        crop_abs = os.path.join(FIGURES_DIR, "{}_{}.jpg".format(idp, p["crop"]))
        boxes = box_divs(ds["boxes"], flip=flip)
        c1 = ensure_crop(crop_abs, ds["boxes"][0], "-crop")
        c2 = ensure_crop(crop_abs, ds["boxes"][1], "-crop2")

    # "inset" layout attaches the two crops to the right of the main image,
    # stacked top/bottom; "below" places them in a row underneath. Arrow-mode
    # datasets have no crops at all, in either layout.
    side_crops = ""
    crops = ""
    if p["crop"] and not use_arrow and layout == "inset":
        side_crops = (
            '<div class="side-crops">'
            '<img class="crop1" src="{c1}" alt="crop1">'
            '<img class="crop2" src="{c2}" alt="crop2">'
            "</div>"
        ).format(c1=c1, c2=c2)
    elif p["crop"] and not use_arrow:
        crops = (
            '<div class="crops">'
            '<img class="crop1" src="{c1}" alt="crop1">'
            '<img class="crop2" src="{c2}" alt="crop2">'
            "</div>"
        ).format(c1=c1, c2=c2)

    metric = ""
    if p["model"] is not None:
        m = models[p["model"]]
        metric = '<div class="metric">{d:.4f} / {b:.2f}</div>'.format(d=m["dice"], b=m["betti_error_0"])

    main = (
        '<div class="main">'
        '<img src="{src}" width="{w}" height="{w}" alt="{lbl}">'
        "{boxes}{arrow}{metric}"
        "</div>"
    ).format(src=main_src, w=MAIN_W, lbl=label, boxes=boxes, arrow=arrow, metric=metric)

    panel_class = "panel"
    if layout == "inset" and p["crop"] and not use_arrow:
        panel_class += " wide"
        body = '<div class="main-wrap">{main}{side_crops}</div>'.format(main=main, side_crops=side_crops)
    else:
        body = main + crops

    return (
        '<div class="{cls}">'
        '<div class="title">{title}</div>'
        "{body}"
        "</div>"
    ).format(cls=panel_class, title=label, body=body)


def fix_segdiff_orientation(idp):
    """SegDiff's saved prediction can be diagonally transposed relative to
    the reference image (rows/cols swapped -- e.g. DeepCrack's, confirmed
    against notebooks/paper_figures.ipynb's DeepCrack cell, which applies
    the same fix but only in-memory for display). This pipeline reads the
    file straight off disk, so fix it there once; idempotent, since a
    second run finds the shapes already matching and no-ops.

    Compares against the *raw* (pre-"-resized") reference image: the
    "-resized" copy is always square (SRCxSRC), so it can never reveal a
    W/H swap -- only the original aspect-ratio image can.
    """
    ref_path = os.path.join(FIGURES_DIR, "{}_1_image.jpg".format(idp))
    segdiff_path = os.path.join(FIGURES_DIR, "{}_9_segdiff.jpg".format(idp))
    if not (os.path.isfile(ref_path) and os.path.isfile(segdiff_path)):
        return
    ref_w, ref_h = PILImage.open(ref_path).size
    seg_img = PILImage.open(segdiff_path)
    if ref_w != ref_h and seg_img.size == (ref_h, ref_w):  # W/H swapped -> transpose fixes it
        seg_img.transpose(PILImage.Transpose.TRANSPOSE).save(segdiff_path)
        print("fixed orientation:", segdiff_path)


def _magma_jpg(arr01, out_path):
    """Save a [0,1] float array as an RGB JPG using the magma colormap,
    matching notebooks/distance_map_drive.ipynb's `cmap="magma"` display
    (its own saved *_dist.png / *_dist_inverted.npy previews are plain
    grayscale -- the magma coloring only happens at display time there)."""
    rgba = matplotlib.colormaps["magma"](arr01)
    PILImage.fromarray((rgba[..., :3] * 255).astype(np.uint8)).save(out_path)


def ensure_distance_map_assets(idp=DISTANCE_MAP_ID):
    """(Re)generate the Image / Y / D / D_inv panel JPGs for the DRIVE
    distance-map figure straight from data/DRIVE/{split}, every time, so a
    changed source mask/image can never be masked by a stale cached file.
    Returns {letter: figures-relative filename}."""
    image = PILImage.open(
        os.path.join(DISTANCE_DATA_DIR, "images", "{}.tif".format(idp))
    ).convert("RGB").resize((SRC, SRC), PILImage.Resampling.BILINEAR)
    binary = PILImage.open(
        os.path.join(DISTANCE_DATA_DIR, "masks", "{}.gif".format(idp))
    ).convert("L").resize((SRC, SRC), PILImage.Resampling.NEAREST)
    dist = np.load(os.path.join(DISTANCE_DATA_DIR, "masks_dist", "{}_dist.npy".format(idp)))
    dist_inv = np.load(
        os.path.join(DISTANCE_DATA_DIR, "masks_dist_inverted", "{}_dist_inverted.npy".format(idp))
    )

    filenames = {}

    filenames["a"] = "distmap_{}_a_image.jpg".format(idp)
    image.save(os.path.join(FIGURES_DIR, filenames["a"]))

    filenames["b"] = "distmap_{}_b_binary.jpg".format(idp)
    binary.convert("RGB").save(os.path.join(FIGURES_DIR, filenames["b"]))

    filenames["c"] = "distmap_{}_c_dist.jpg".format(idp)
    _magma_jpg(
        np.array(PILImage.fromarray(dist).resize((SRC, SRC), PILImage.Resampling.BILINEAR)),
        os.path.join(FIGURES_DIR, filenames["c"]),
    )

    filenames["d"] = "distmap_{}_d_dist_inv.jpg".format(idp)
    _magma_jpg(
        np.array(PILImage.fromarray(dist_inv).resize((SRC, SRC), PILImage.Resampling.BILINEAR)),
        os.path.join(FIGURES_DIR, filenames["d"]),
    )

    for f in filenames.values():
        print("wrote", os.path.join(FIGURES_DIR, f))
    return filenames


def distance_map_panel_html(letter, label, filename):
    main = (
        '<div class="main">'
        '<img src="{rel}{src}" width="{w}" height="{w}" alt="{lbl}">'
        "</div>"
    ).format(rel=FIG_REL, src=filename, w=MAIN_W, lbl=label)
    return (
        '<div class="panel dist">'
        '<div class="title">({letter}) {label}</div>'
        "{main}"
        "</div>"
    ).format(letter=letter, label=label, main=main)


def distance_map_html(idp=DISTANCE_MAP_ID):
    filenames = ensure_distance_map_assets(idp)
    cells = [distance_map_panel_html(p["letter"], p["label"], filenames[p["letter"]]) for p in DISTANCE_PANELS]
    rows = "".join(
        '<div class="row">{}</div>'.format("".join(cells[i:i + DISTANCE_COLS]))
        for i in range(0, len(cells), DISTANCE_COLS)
    )
    return '<section class="figure">{}</section>'.format(rows)


def dataset_html(ds, layout="below"):
    fix_segdiff_orientation(ds["id"])

    with open(os.path.join(FIGURES_DIR, ds["json"]), encoding="utf-8") as f:
        models = json.load(f)["models"]

    if ds.get("use_arrow"):
        ds = dict(ds, arrow_target=compute_arrow_target(
            os.path.join(FIGURES_DIR, "{}_2_gt-resized.jpg".format(ds["id"])),
            os.path.join(FIGURES_DIR, "{}_3_proposed.jpg".format(ds["id"])),
        ))

    # Image, GT (unlettered), then the model panels lettered alphabetically
    # starting at the first one.
    image_panel, gt_panel = PANELS[0], PANELS[1]
    model_panels = PANELS[2:]

    ref_cells = [panel_html(ds, image_panel, models, layout), panel_html(ds, gt_panel, models, layout)]
    model_cells = []
    for idx, p in enumerate(model_panels):
        letter = chr(97 + idx)
        p_copied = p.copy()
        p_copied["label"] = "({}) {}".format(letter, p["label"])
        model_cells.append(panel_html(ds, p_copied, models, layout))

    # Reference column: GT stacked directly under Image, split off from the
    # model panels by a vertical rule; the models then fill 5 columns x 2 rows.
    model_rows = "".join(
        '<div class="row">{}</div>'.format("".join(model_cells[i:i + MODEL_COLS]))
        for i in range(0, len(model_cells), MODEL_COLS)
    )
    body = (
        '<div class="split">'
        '<div class="ref-col">{ref}</div>'
        '<div class="col-divider"></div>'
        '<div class="model-grid">{rows}</div>'
        "</div>"
    ).format(ref="".join(ref_cells), rows=model_rows)

    return (
        '<section class="figure">'
        "{body}"
        "</section>"
    ).format(body=body)


def get_css():
    base = """
* {{ box-sizing: border-box; }}
body {{
  font-family: 'Times New Roman', Georgia, serif;
  background: #ffffff; color: #000;
  margin: 0; padding: 32px;
}}
h2 {{
  font-size: 22px; font-weight: 700; margin: 0 0 14px;
  text-align: left;
}}
.figure {{ margin: 0 auto 48px; width: fit-content; }}
.figure:last-child {{ margin-bottom: 0; }}
.row {{ display: flex; gap: 48px; margin-bottom: 12px; align-items: flex-start; }}
/* Image/GT stacked in a first column, then a vertical rule, then the model
   grid. The 24px gaps on either side of the 2px rule add up to the same ~48px
   the panels within a row are spaced by. */
.split {{ display: flex; gap: 24px; align-items: flex-start; }}
.ref-col {{ display: flex; flex-direction: column; gap: 12px; }}
.col-divider {{ align-self: stretch; width: 2px; background: #000; flex: none; }}
.model-grid {{ display: flex; flex-direction: column; }}
.model-grid .row:last-child {{ margin-bottom: 0; }}
.panel {{
  width: {mw}px; text-align: center;
  display: flex; flex-direction: column; align-items: center;
}}
.panel.wide {{ width: {tw:.1f}px; }}
.title {{ font-weight: 700; font-size: 26px; margin-bottom: 4px; width: {mw}px; }}
.panel.wide .title {{ width: {tw:.1f}px; }}
/* KaTeX-typeset titles ($Y$, $\\mathcal{{D}}$) are taller than plain-text ones
   ("Image"), so leaving them in flow would shift the images in a row off each
   other, and pinning them by either edge would leave their baselines off by
   the ascent difference. Instead: reserve a fixed strip at the top of each
   panel for an out-of-flow title, and start that title with a zero-width
   strut taller than any label. The strut's own bottom is its baseline, so it
   dominates the line box and puts every title's baseline at the same height,
   while letter and label still lay out inline (i.e. on a shared baseline). */
.panel.dist {{ position: relative; padding-top: {dth}px; }}
.panel.dist .title {{ position: absolute; top: 0; left: 0; margin: 0; }}
.panel.dist .title::before {{
  content: ""; display: inline-block; width: 0; height: {sth}px;
  vertical-align: baseline;
}}
.main-wrap {{ display: flex; align-items: flex-start; }}
.panel.wide .main-wrap {{ width: {tw:.1f}px; }}
.main {{ position: relative; width: {mw}px; height: {mw}px; line-height: 0; flex: none; }}
.main img {{ display: block; width: {mw}px; height: {mw}px; object-fit: cover; }}
.roi {{
  position: absolute; border: 2px solid; pointer-events: none;
}}
.arrow {{
  position: absolute; top: 0; left: 0; width: {mw}px; height: {mw}px;
  pointer-events: none; z-index: 1;
}}
.crops {{ display: flex; width: {mw}px; line-height: 0; }}
.crops img {{ width: {cw:.1f}px; height: {cw:.1f}px; object-fit: cover; }}
.crops img.crop1 {{ border: 3px solid {c1}; }}
.crops img.crop2 {{ border: 3px solid {c2}; }}
.side-crops {{ display: flex; flex-direction: column; width: {cw:.1f}px; flex: none; line-height: 0; }}
.side-crops img {{ display: block; width: {cw:.1f}px; height: {cw:.1f}px; object-fit: cover; }}
.side-crops img.crop1 {{ border: 3px solid {c1}; }}
.side-crops img.crop2 {{ border: 3px solid {c2}; }}
.metric {{
  position: absolute; right: 4px; bottom: 4px; z-index: 2;
  font-size: {mf}px; font-weight: 700; color: #ffff00; text-align: right; line-height: 1.2;
  text-shadow: 1px 1px 2px #000, -1px -1px 2px #000, 1px -1px 2px #000, -1px 1px 2px #000;
}}
""".format(mw=MAIN_W, cw=CROP_W, tw=TOTAL_W, c1=BOX_COLORS[0], c2=BOX_COLORS[1],
           mf=METRIC_FONT_SIZE, dth=DISTANCE_TITLE_H, sth=DISTANCE_STRUT_H)

    return base + PRINT_CSS


def build_comparison_htmls():
    """Write the comparison + distance-map HTML reports; return their paths."""
    title = "Segmentation Comparison Figures"
    html_paths = []
    for layout, fname in [("below", "figure.html"), ("inset", "figure_inset.html")]:
        body = "".join(dataset_html(ds, layout) for ds in DATASETS)
        out = os.path.join(TMP_DIR, fname)
        with open(out, "w", encoding="utf-8") as f:
            f.write(wrap_html(body, title, get_css()))
        print("wrote", out)
        html_paths.append(out)

    out = os.path.join(TMP_DIR, "figure_distance_map.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(wrap_html(distance_map_html(), title, get_css()))
    print("wrote", out)
    html_paths.append(out)
    return html_paths


# ── Ablation figures ─────────────────────────────────────────────────────────

GROUPS = {
    "ablation_table": {
        "title": "Qualitative Comparison of Ablation Study on Proposed Components",
        # Two DRIVE samples, stacked as two rows of one figure with a single
        # ROI crop each (see TP_PATCH_BOXES), rather than one sample carrying
        # two crops: no single test image offers two windows that both separate
        # the three options well (see scripts/compare_tp_samples.py).
        "drive_json": [
            "drive_ablation_ablation_table_02.json",
            "drive_ablation_ablation_table_07.json",
        ],
        "deepcrack_json": "deepcrack_ablation_ablation_table_11269-5.json",
    },
    # Alternative layouts of the same ablation, kept alongside the two-sample
    # figure above so they can be compared before one is chosen for the paper:
    # one sample carrying both ROI crops. "labels" points at the group whose
    # LABEL_MAPS / row layout / patch-search settings apply.
    "ablation_table_1x2_02": {
        "title": "Qualitative Comparison of Ablation Study on Proposed Components",
        "labels": "ablation_table",
        "drive_json": "drive_ablation_ablation_table_02.json",
    },
    "ablation_table_1x2_08": {
        "title": "Qualitative Comparison of Ablation Study on Proposed Components",
        "labels": "ablation_table",
        "drive_json": "drive_ablation_ablation_table_08.json",
    },
    "imo_variation": {
        "title": "Qualitative Comparison of Ablation Study on Rotating the Affinity Representation",
        "drive_json": "drive_ablation_imo_variation_06.json",
        "deepcrack_json": "deepcrack_ablation_imo_variation_11269-5.json",
    },
    "s_variation": {
        "title": "Qualitative Comparison of Ablation Study on Affinity Scale S",
        "drive_json": "drive_ablation_s_variation_04.json",
        "deepcrack_json": "deepcrack_ablation_s_variation_11269-5.json",
    }
}

DATASET_TITLES = {
    "19": "DRIVE — Retinal Vessel Segmentation",
    "11269-5": "DeepCrack — Crack Segmentation"
}

ARROW_DATASET_IDS = {"11269-5"}

# LaTeX equations used as titles so users can easily edit them in code
LABEL_MAPS = {
    "ablation_table": {
        "binary+madf=True+segaux=True": "$Y$",
        "dist+madf=True+segaux=True": "$\\mathcal{D}$",
        "dist_inverted+madf=True+segaux=True": "$\\bar{\\mathcal{D}}$",
    },
    "imo_variation": {
        "A": "$\\mathcal{A}_{S_1}$",
        "\\textbf{\\{A,B\\}}": "$\\mathcal{A}_{S_{1,2}}$ $\\star$",
        # The rotated variant primes the scale index (2') rather than the
        # symbol, so it folds into the shared subscript like the others.
        "\\{A,C\\}": "$\\mathcal{A}_{S_{1,2'}}$",
        "\\{A,B,C\\}": "$\\mathcal{A}_{S_{1,2,2'}}$",
        "D": "$\\mathcal{A}_{24}$"
    },
    "s_variation": {
        "1": "$\\mathcal{S}_1$",
        "2": "$\\mathcal{S}_2$",
        "\\textbf{\\{1,2\\}}": "$\\mathcal{S}_{1,2}$ $\\star$",
        "\\{1,2,3\\}": "$\\mathcal{S}_{1,2,3}$",
        "\\{1,2,3,4\\}": "$\\mathcal{S}_{1,2,3,4}$"
    }
}

# Each entry is one rendered HTML variant per group. To also add crop
# insets to e.g. the empty/no_image/no_image_empty variants, add matching
# entries here with crop=True (mirrors the "_crop" one below).
VARIANTS = [
    dict(suffix="", empty=False, no_image=False, crop=False),
    dict(suffix="_no_image", empty=False, no_image=True, crop=False),
    dict(suffix="_crop", empty=False, no_image=False, crop=True),
    dict(suffix="_no_image_crop", empty=False, no_image=True, crop=True),
]


def get_ablation_css():
    return """
* {{ box-sizing: border-box; }}
body {{
  font-family: 'Times New Roman', Georgia, serif;
  background: #ffffff; color: #000;
  margin: 0; padding: 32px;
}}
h2 {{
  font-size: 22px; font-weight: 700; margin: 0 0 14px;
  text-align: left;
}}
.figure {{ margin: 0 auto 48px; width: fit-content; }}
.figure:last-child {{ margin-bottom: 0; }}
.row {{ display: flex; gap: 56px; margin-bottom: 12px; align-items: flex-start; }}
.panel {{
  width: {mw}px; text-align: center;
  display: flex; flex-direction: column; align-items: center;
}}
.title {{ font-weight: bold; font-size: {tf}px; margin-bottom: 6px; width: {mw}px; height: {th}px; display: flex; align-items: center; justify-content: center; line-height: 1.2; }}
.main-wrap {{ display: flex; flex-direction: column; align-items: center; width: {mw}px; }}
.main {{ position: relative; width: {mw}px; height: {mw}px; line-height: 0; flex: none; }}
.main img {{ display: block; width: {mw}px; height: {mw}px; object-fit: cover; border: 1px solid #ddd; }}
.panel.highlighted .title {{
  font-weight: bold;
}}
.roi {{ position: absolute; border: 2px solid; pointer-events: none; }}
.arrow {{
  position: absolute; top: 0; left: 0; width: {mw}px; height: {mw}px;
  pointer-events: none; z-index: 1;
}}
.metric {{
  position: absolute; right: 4px; bottom: 4px; z-index: 2;
  font-size: {mf}px; font-weight: 700; color: #ffff00; text-align: right; line-height: 1.2;
  text-shadow: 1px 1px 2px #000, -1px -1px 2px #000, 1px -1px 2px #000, -1px 1px 2px #000;
}}
/* Magnified ROI crops sit in a row directly under the main image, each half
   its width, so the panel stays exactly MAIN_W wide. A lone crop (groups
   drawing a single ROI box) is centered instead of hugging the left edge. */
.crops {{ display: flex; justify-content: center; width: {mw}px; flex: none; line-height: 0; }}
.crops img {{ display: block; width: {cw:.1f}px; height: {cw:.1f}px; object-fit: cover; }}
.crops img.crop1 {{ border: 3px solid {c1}; }}
.crops img.crop2 {{ border: 3px solid {c2}; }}
""".format(mw=MAIN_W, cw=CROP_W, c1=BOX_COLORS[0], c2=BOX_COLORS[1],
           mf=METRIC_FONT_SIZE, tf=ABLATION_TITLE_FONT_SIZE, th=ABLATION_TITLE_H) + PRINT_CSS


def ablation_panel_html(label, img_path, metrics=None, highlighted=False, empty=False, boxes_html="",
                        crop_srcs=None, show_title=True):
    # img_path is stored as "figures/..." in the JSON.
    src_path = img_path.replace("figures/", FIG_REL)

    # One crop per ROI box, in box order, so a group drawing a single box gets
    # a single (centered) crop and one drawing two gets the side-by-side pair.
    crops_html = ""
    if crop_srcs:
        crops_html = '<div class="crops">{}</div>'.format("".join(
            '<img class="crop{i}" src="{src}" alt="crop{i}">'.format(i=i + 1, src=src)
            for i, src in enumerate(crop_srcs)))

    # Yellow Dice/beta_0 readout overlaid on the image itself (bottom-right),
    # not a separate caption below the image.
    metric_html = ""
    if metrics and not empty:
        metric_html = '<div class="metric">{d:.4f} / {b:.2f}</div>'.format(
            d=metrics["dice"], b=metrics["betti_error_0"]
        )

    main_div = (
        '<div class="main-wrap">'
        '<div class="main"><img src="{src}" alt="">{boxes}{metric}</div>'
        "{crops}"
        "</div>"
    ).format(src=src_path, boxes=boxes_html, metric=metric_html, crops=crops_html)

    panel_class = "panel highlighted" if highlighted else "panel"
    if empty:
        title_html = "&nbsp;"
    else:
        title_html = "<strong>{}</strong>".format(label) if highlighted else label

    # Stacked-sample figures label the columns once, on the top row only; the
    # rows below drop the title strip entirely rather than reserving blank
    # space for it.
    title_div = '<div class="title">{}</div>'.format(title_html) if show_title else ""
    return """
  <div class="{cls}">
    {title}
    {main}
  </div>""".format(cls=panel_class, title=title_div, main=main_div)


def base_group(group_key):
    """The group whose LABEL_MAPS / row layout / patch-search settings apply --
    alternative layouts of one ablation point at it via "labels"."""
    return GROUPS[group_key].get("labels", group_key)


def load_ablation_json(json_path):
    with open(os.path.join(FIGURES_DIR, json_path), encoding="utf-8") as f:
        return json.load(f)


def ablation_raw_panels(group_key, data, no_image=False):
    """The figure's panels, in render order, before lettering: Image (unless
    dropped), GT, then one panel per LABEL_MAPS row present in `data`."""
    raw_panels = []
    # 1. Image panel -- annotated (ROI box + magnified crop, or arrow) like every
    # other panel, matching the comparison figure's convention, so the crop row
    # shows what the region looks like in the fundus/crack photo itself.
    if not no_image:
        raw_panels.append({
            "label": "Image",
            "img": data["image"],
            "metrics": None,
            "highlighted": False,
            "crop_eligible": True,
        })
    # 2. GT panel
    raw_panels.append({
        "label": "GT",
        "img": data["gt"],
        "metrics": None,
        "highlighted": False,
        "crop_eligible": True,
    })
    # 3. Row panels (Filter & Map using LABEL_MAPS) -- iterate in LABEL_MAPS'
    # order (not data["rows"]'s, which is whatever order the ablation JSON
    # happened to write its rows in), so panel order is controlled entirely
    # by LABEL_MAPS here.
    maps = LABEL_MAPS[base_group(group_key)]
    for key, label in maps.items():
        row = data["rows"].get(key)
        if row is None:
            continue
        raw_panels.append({
            "label": label,
            "img": row["pred"],
            "metrics": {
                "dice": row["dice"],
                "betti_error_0": row["betti_error_0"]
            },
            "highlighted": row.get("highlighted", False),
            "crop_eligible": True,
        })
    return raw_panels


def ablation_dataset_rows(group_key, json_path, empty=False, no_image=False, crop=False,
                          show_titles=True):
    """The `.row` divs for one sample's panels (no enclosing section), so a
    group can stack several samples inside one figure."""
    data = load_ablation_json(json_path)
    ds_id = data["id"]
    use_arrow = crop and ds_id in ARROW_DATASET_IDS
    raw_panels = ablation_raw_panels(group_key, data, no_image=no_image)

    # ROI boxes: the sample's final hand-set boxes if it has any, else the
    # group's searched candidates -- of which TP_PATCH_PICKS selects the ones
    # that get drawn (see scripts/compare_tp_patches.py to eyeball the full
    # candidate list) -- else the hand-picked DATASET_BOXES entry.
    ds_boxes = None
    if crop and not use_arrow:
        if ds_id in TP_PATCH_BOXES.get(group_key, {}):
            ds_boxes = TP_PATCH_BOXES[group_key][ds_id]
        elif base_group(group_key) in TP_PATCH_GROUPS:
            candidates = search_option_gap_patches(data, raw_panels)
            ds_boxes = [candidates[i] for i in TP_PATCH_PICKS]
        else:
            ds_boxes = DATASET_BOXES.get(ds_id)

    # Arrow target: highest-foreground-TP ARROW_PATCH_SIZE patch, found once
    # from GT + the highlighted (proposed) row's prediction and reused (same
    # pixel location) on every panel -- mirrors the comparison figure using
    # "Proposed".
    arrow_target = None
    if use_arrow:
        highlighted_pred = next((p["img"] for p in raw_panels if p.get("highlighted")), None)
        if highlighted_pred is not None:
            arrow_target = compute_arrow_target(
                os.path.join(PAPER_TABLES_DIR, data["gt"]),
                os.path.join(PAPER_TABLES_DIR, highlighted_pred),
            )

    # Add alphabetical prefix to labels, starting at the first row panel
    # (Image/GT are left unlettered). Uses a non-breaking space, not a
    # regular one: Chromium collapses a plain space to zero width when it
    # sits right before KaTeX's inline-block math span, silently swallowing
    # the gap for every math-mode label (e.g. "(c) $\\mathcal{S}_1$").
    panels = []
    letter_idx = 0
    for p in raw_panels:
        p_copied = p.copy()
        if p["label"] not in ("Image", "GT"):
            p_copied["label"] = "({})\xa0{}".format(chr(97 + letter_idx), p["label"])
            letter_idx += 1
        panels.append(p_copied)

    def render_panel(idx, p):
        annotation_html, crop_srcs = "", None
        if use_arrow and arrow_target is not None and p["crop_eligible"]:
            annotation_html = arrow_svg(arrow_target, uid="{}-{}-{}".format(ds_id, group_key, idx))
        elif ds_boxes and p["crop_eligible"]:
            img_abs = os.path.join(PAPER_TABLES_DIR, p["img"])
            annotation_html = box_divs(ds_boxes)
            crop_srcs = [
                ensure_crop(img_abs, box, "-crop" if i == 0 else "-crop{}".format(i + 1))
                for i, box in enumerate(ds_boxes)
            ]
        return ablation_panel_html(p["label"], p["img"], p["metrics"], p["highlighted"], empty=empty,
                                   boxes_html=annotation_html, crop_srcs=crop_srcs,
                                   show_title=show_titles)

    # imo_variation / s_variation have too many panels (Image, GT + 5
    # variants) for one row, so Image (when present) gets its own row and the
    # remaining 6 -- GT + the 5 lettered variants -- form a 3x2 grid. In the
    # no_image variants those 6 are the whole figure, and go on a single row.
    # Other groups (e.g. ablation_table, with only 3 variants) keep everything
    # in a single row.
    indexed_panels = list(enumerate(panels))
    if base_group(group_key) in ("imo_variation", "s_variation"):
        image_row = [(idx, p) for idx, p in indexed_panels if p["label"] == "Image"]
        grid = [(idx, p) for idx, p in indexed_panels if p["label"] != "Image"]
        cols = 3 if image_row else max(len(grid), 1)
        row_groups = [image_row] if image_row else []
        row_groups += [grid[i:i + cols] for i in range(0, len(grid), cols)]
        row_groups = [g for g in row_groups if g]
    else:
        row_groups = [indexed_panels]

    return "".join(
        '<div class="row">' + "".join(render_panel(idx, p) for idx, p in group) + "</div>"
        for group in row_groups
    )


def ablation_dataset_html(group_key, json_paths, empty=False, no_image=False, crop=False):
    """One figure section for a dataset. `json_paths` may be a single sample's
    JSON or a list of them, in which case the samples are stacked as extra rows
    of the same figure and only the top row carries the column titles."""
    if isinstance(json_paths, str):
        json_paths = [json_paths]
    rows = "".join(
        ablation_dataset_rows(group_key, path, empty=empty, no_image=no_image, crop=crop,
                              show_titles=(i == 0))
        for i, path in enumerate(json_paths)
    )
    return """
  <section class="figure">
    {rows}
  </section>""".format(rows=rows)


def build_ablation_group_html(group_key, group_meta, empty=False, no_image=False, crop=False):
    drive_html = ablation_dataset_html(group_key, group_meta["drive_json"], empty=empty, no_image=no_image, crop=crop)

    # TABLE III / ablation_table is only evaluated on the DRIVE Dataset in the paper
    if base_group(group_key) == "ablation_table":
        deepcrack_html = ""
    else:
        deepcrack_html = ablation_dataset_html(group_key, group_meta["deepcrack_json"], empty=empty,
                                               no_image=no_image, crop=crop)

    return wrap_html(drive_html + deepcrack_html, group_meta["title"], get_ablation_css())


def build_ablation_htmls():
    """Write the ablation HTML reports; return their paths."""
    html_paths = []
    for group_key, meta in GROUPS.items():
        for variant in VARIANTS:
            html_content = build_ablation_group_html(
                group_key, meta, empty=variant["empty"], no_image=variant["no_image"], crop=variant["crop"]
            )
            out_path = os.path.join(TMP_DIR, "figure_{}{}.html".format(group_key, variant["suffix"]))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print("wrote", out_path)
            html_paths.append(out_path)
    return html_paths


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    html_paths = build_comparison_htmls() + build_ablation_htmls()
    render_html_files(html_paths, RESULTS_DIR)


if __name__ == "__main__":
    main()
