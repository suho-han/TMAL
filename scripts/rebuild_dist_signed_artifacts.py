#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from connect_loss import Bilateral_voting, Bilateral_voting_kxk, distance_affinity_matrix, normalize_conn_layout, resolve_connectivity_layout
from data_loader.GetDataset_CHASE import MyDataset_CHASE
from model.DconnNet import DconnNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@dataclass
class EpochResult:
    checkpoint: str
    epoch_key: str
    logits_mean: float
    logits_std: float
    sig_gt_05_ratio: float
    logit_gt_0_ratio: float
    vote_pos_sig05: float
    vote_pos_logit0: float
    gt_pos_0_ratio: float
    dir_logit0_0: float
    dir_logit0_1: float
    dir_logit0_2: float
    dir_logit0_3: float
    dir_logit0_4: float
    dir_logit0_5: float
    dir_logit0_6: float
    dir_logit0_7: float
    pair_0_7: float
    pair_1_6: float
    pair_2_5: float
    pair_3_4: float
    collapsed_voting: int


def parse_args():
    p = argparse.ArgumentParser(
        "Unified dist debug: single-checkpoint analysis + epochwise CSV + markdown report"
    )
    p.add_argument("--output_root", type=str, default="output/dist")
    p.add_argument("--exp_index", type=int, default=1, help="models/<exp_index>")
    p.add_argument("--data_root", type=str, default="data/DRIVE")
    p.add_argument("--label_mode", type=str, default="dist",
                   choices=["binary", "dist", "dist_inverted"])
    p.add_argument("--exp_id", type=int, default=0, help="DRIVE split id (0-based)")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--resize", type=int, nargs=2, default=[960, 960])
    p.add_argument("--num_class", type=int, default=1)
    p.add_argument("--conn_num", type=int, default=8, choices=[8, 24])
    p.add_argument("--conn_layout", type=str, default=None, choices=["standard8", "full24", "out8"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=6)
    p.add_argument("--max_checkpoints", type=int, default=0, help="0 means all")
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--include_best", action="store_true", default=True)
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--detailed_checkpoint", type=str, default="best_model.pth")
    p.add_argument("--epochwise_csv_name", type=str, default="epochwise_voting_debug.csv")
    p.add_argument("--meta_name", type=str, default="best_model_meta.txt")
    p.add_argument("--report_name", type=str, default="debug_report.md")
    args = p.parse_args()
    args.conn_layout = normalize_conn_layout(args.conn_num, args.conn_layout)
    args.connectivity_layout = resolve_connectivity_layout(args.conn_num, args.conn_layout)
    args.conn_channels = args.connectivity_layout["channel_count"]
    if args.num_class != 1 and args.conn_layout != "standard8":
        raise ValueError("rebuild_dist_signed_artifacts currently supports non-standard conn_layout only for single-class runs")
    return args


def make_chase_split(exp_id: int) -> Tuple[List[str], List[str]]:
    overall = ["01", "02", "03", "04", "05", "06", "07",
               "08", "09", "10", "11", "12", "13", "14"]
    test_ids = overall[3 * exp_id:3 * (exp_id + 1)]
    train_ids = list(set(overall) - set(test_ids))
    return train_ids, test_ids


def shift_2d(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    if dy >= 0:
        src_y = slice(0, x.shape[2] - dy)
        dst_y = slice(dy, x.shape[2])
    else:
        src_y = slice(-dy, x.shape[2])
        dst_y = slice(0, x.shape[2] + dy)
    if dx >= 0:
        src_x = slice(0, x.shape[3] - dx)
        dst_x = slice(dx, x.shape[3])
    else:
        src_x = slice(-dx, x.shape[3])
        dst_x = slice(0, x.shape[3] + dx)
    out[:, :, dst_y, dst_x] = x[:, :, src_y, src_x]
    return out


def pair_alignment_scores(class_pred_bin: torch.Tensor, offsets: List[Tuple[int, int]]) -> np.ndarray:
    a = class_pred_bin[:, 0]
    offset_to_idx = {offset: idx for idx, offset in enumerate(offsets)}
    pair_values: List[float] = []
    visited: set[Tuple[int, int]] = set()
    for offset in offsets:
        if offset in visited:
            continue
        reverse_offset = (-offset[0], -offset[1])
        if reverse_offset not in offset_to_idx:
            raise ValueError(
                f"pair_alignment_scores requires every offset to have a reverse pair, "
                f"missing {reverse_offset} for offset {offset}"
            )
        reverse_idx = offset_to_idx[reverse_offset]
        idx = offset_to_idx[offset]
        pair_values.append(
            float((a[:, idx:idx + 1] * shift_2d(a[:, reverse_idx:reverse_idx + 1], dy=offset[0], dx=offset[1])).mean().item())
        )
        visited.add(offset)
        visited.add(reverse_offset)
        if len(pair_values) == 4:
            break
    if len(pair_values) != 4:
        raise ValueError(
            f"pair_alignment_scores expects exactly 4 reverse-offset pairs, got {len(pair_values)} "
            f"from offsets={offsets}"
        )
    return np.array(pair_values, dtype=np.float64)


def apply_connectivity_voting(
    conn_map: torch.Tensor,
    hori: torch.Tensor,
    vert: torch.Tensor,
    conn_num: int,
    conn_layout: str | None = None,
):
    layout = resolve_connectivity_layout(conn_num, conn_layout)
    if layout["name"] == "standard8":
        return Bilateral_voting(conn_map, hori, vert)
    return Bilateral_voting_kxk(
        conn_map,
        hori,
        vert,
        conn_num=layout['kernel_size'],
        offsets=layout['offsets'],
    )


def build_translation(num_class: int, h: int, w: int, device: torch.device):
    hori = torch.zeros([1, num_class, w, w], device=device, dtype=torch.float32)
    vert = torch.zeros([1, num_class, h, h], device=device, dtype=torch.float32)
    for i in range(w - 1):
        hori[:, :, i, i + 1] = 1.0
    for j in range(h - 1):
        vert[:, :, j, j + 1] = 1.0
    return hori, vert


def checkpoint_sort_key(name: str):
    if name == "best_model.pth":
        return (1, 10**9)
    m = re.match(r"(\d+)_model\.pth$", name)
    if m:
        return (0, int(m.group(1)))
    return (2, name)


def collect_checkpoints(ckpt_dir: Path, include_best: bool) -> List[Path]:
    names: List[str] = []
    for n in os.listdir(ckpt_dir):
        if re.match(r"\d+_model\.pth$", n):
            names.append(n)
        elif include_best and n == "best_model.pth":
            names.append(n)
    names.sort(key=checkpoint_sort_key)
    return [ckpt_dir / n for n in names]


def parse_results_csv(path: Path) -> Dict[int, Dict[str, float]]:
    metrics: Dict[int, Dict[str, float]] = {}
    if not path.exists():
        return metrics
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] == "epoch":
                continue
            if len(row) < 6:
                continue
            ep = int(row[0])
            cldc_raw = row[5].strip().lower()
            cldc = float("nan") if cldc_raw == "nan" else float(row[5])
            metrics[ep] = {
                "train_loss": float(row[1]),
                "val_loss": float(row[2]),
                "dice": float(row[3]),
                "jac": float(row[4]),
                "cldice": cldc,
            }
    return metrics


def select_best_epoch(results: Dict[int, Dict[str, float]]) -> Optional[Tuple[int, Dict[str, float]]]:
    if not results:
        return None
    best_epoch = max(results.keys(), key=lambda ep: results[ep]["dice"])
    return best_epoch, results[best_epoch]


def write_epochwise_csv(path: Path, rows: List[EpochResult]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EpochResult.__annotations__.keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def evaluate_epoch_checkpoint(
    model: DconnNet,
    ckpt_path: Path,
    loader: DataLoader,
    device: torch.device,
    hori: torch.Tensor,
    vert: torch.Tensor,
    conn_num: int,
    conn_channels: int,
    conn_layout: str,
    max_batches: int,
) -> EpochResult:
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    layout = resolve_connectivity_layout(conn_num, conn_layout)

    logits_means: List[float] = []
    logits_stds: List[float] = []
    sig_gt_05: List[float] = []
    logit_gt_0: List[float] = []
    vote_sig05: List[float] = []
    vote_logit0: List[float] = []
    gt_pos0: List[float] = []
    dir_logit0: List[np.ndarray] = []
    pair_vals: List[np.ndarray] = []

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if bidx >= max_batches:
                break
            x, y, _ = batch
            x = x.to(device)
            y = y.float().to(device)
            out_dict = model(x)
            out = out_dict['fused']
            sig = torch.sigmoid(out)
            cp_sig = sig.view([x.shape[0], -1, conn_channels, x.shape[2], x.shape[3]])
            cp_logit = out.view([x.shape[0], -1, conn_channels, x.shape[2], x.shape[3]])
            bin_sig = (cp_sig > 0.5).float()
            bin_logit = (cp_logit > 0).float()

            hori_batch = hori.repeat(x.shape[0], 1, 1, 1)
            vert_batch = vert.repeat(x.shape[0], 1, 1, 1)
            pred_sig_raw, _ = apply_connectivity_voting(bin_sig, hori_batch, vert_batch, conn_num, conn_layout)
            pred_logit_raw, _ = apply_connectivity_voting(bin_logit, hori_batch, vert_batch, conn_num, conn_layout)
            pred_sig = (pred_sig_raw > 0).float()
            pred_logit = (pred_logit_raw > 0).float()

            logits_means.append(float(out.mean().item()))
            logits_stds.append(float(out.std().item()))
            sig_gt_05.append(float((sig > 0.5).float().mean().item()))
            logit_gt_0.append(float((out > 0).float().mean().item()))
            vote_sig05.append(float(pred_sig.mean().item()))
            vote_logit0.append(float(pred_logit.mean().item()))
            gt_pos0.append(float((y > 0).float().mean().item()))
            dir_logit0.append(bin_logit[:, 0].mean(dim=(0, 2, 3)).detach().cpu().numpy())
            pair_vals.append(pair_alignment_scores(bin_logit, layout["offsets"]))

    dir_arr = np.stack(dir_logit0, axis=0).mean(axis=0)
    pair_arr = np.stack(pair_vals, axis=0).mean(axis=0)
    ckpt_name = ckpt_path.name
    epoch_key = ckpt_name.replace("_model.pth", "").replace(".pth", "")
    collapsed = int(np.isclose(np.mean(vote_logit0), 0.0, atol=1e-12))

    return EpochResult(
        checkpoint=ckpt_name,
        epoch_key=epoch_key,
        logits_mean=float(np.mean(logits_means)),
        logits_std=float(np.mean(logits_stds)),
        sig_gt_05_ratio=float(np.mean(sig_gt_05)),
        logit_gt_0_ratio=float(np.mean(logit_gt_0)),
        vote_pos_sig05=float(np.mean(vote_sig05)),
        vote_pos_logit0=float(np.mean(vote_logit0)),
        gt_pos_0_ratio=float(np.mean(gt_pos0)),
        dir_logit0_0=float(dir_arr[0]),
        dir_logit0_1=float(dir_arr[1]),
        dir_logit0_2=float(dir_arr[2]),
        dir_logit0_3=float(dir_arr[3]),
        dir_logit0_4=float(dir_arr[4]),
        dir_logit0_5=float(dir_arr[5]),
        dir_logit0_6=float(dir_arr[6]),
        dir_logit0_7=float(dir_arr[7]),
        pair_0_7=float(pair_arr[0]),
        pair_1_6=float(pair_arr[1]),
        pair_2_5=float(pair_arr[2]),
        pair_3_4=float(pair_arr[3]),
        collapsed_voting=collapsed,
    )


def summarize_tensor(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def evaluate_single_checkpoint_detailed(
    model: DconnNet,
    ckpt_path: Path,
    loader: DataLoader,
    device: torch.device,
    hori: torch.Tensor,
    vert: torch.Tensor,
    conn_num: int,
    conn_channels: int,
    conn_layout: str,
    max_batches: int,
    tau: float,
    sigma: float,
    label_mode: str,
) -> Dict[str, float]:
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    layout = resolve_connectivity_layout(conn_num, conn_layout)

    logits_mean = []
    logits_std = []
    sig_gt_05 = []
    logit_gt_0 = []
    gt_gt_0 = []
    gt_gt_05 = []
    vote_pos_sig05 = []
    vote_pos_logit0 = []
    pair_sig05 = []
    pair_logit0 = []
    score_target_mean = []
    affinity_target_mean = []

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if bidx >= max_batches:
                break
            x, y, _ = batch
            x = x.to(device)
            y = y.float().to(device)

            out_dict = model(x)
            out = out_dict['fused']
            sig = torch.sigmoid(out)
            cp_sig = sig.view([x.shape[0], -1, conn_channels, x.shape[2], x.shape[3]])
            cp_logit = out.view([x.shape[0], -1, conn_channels, x.shape[2], x.shape[3]])
            bin_sig = (cp_sig > 0.5).float()
            bin_logit = (cp_logit > 0).float()

            hori_batch = hori.repeat(x.shape[0], 1, 1, 1)
            vert_batch = vert.repeat(x.shape[0], 1, 1, 1)
            pred_sig_raw, _ = apply_connectivity_voting(bin_sig, hori_batch, vert_batch, conn_num, conn_layout)
            pred_logit_raw, _ = apply_connectivity_voting(bin_logit, hori_batch, vert_batch, conn_num, conn_layout)
            pred_sig = (pred_sig_raw > 0).float()
            pred_logit = (pred_logit_raw > 0).float()

            lstat = summarize_tensor(out)
            logits_mean.append(lstat["mean"])
            logits_std.append(lstat["std"])
            sig_gt_05.append(float((sig > 0.5).float().mean().item()))
            logit_gt_0.append(float((out > 0).float().mean().item()))
            gt_gt_0.append(float((y > 0).float().mean().item()))
            gt_gt_05.append(float((y > 0.5).float().mean().item()))
            vote_pos_sig05.append(float(pred_sig.mean().item()))
            vote_pos_logit0.append(float(pred_logit.mean().item()))
            pair_sig05.append(pair_alignment_scores(bin_sig, layout["offsets"]))
            pair_logit0.append(pair_alignment_scores(bin_logit, layout["offsets"]))

            if label_mode in ["dist", "dist_inverted"]:
                score_target = 1.0 - torch.exp(-y / tau)
                affinity_target = distance_affinity_matrix(
                    y,
                    conn_num=conn_num,
                    sigma=sigma,
                    conn_layout=conn_layout,
                )
                score_target_mean.append(float(score_target.mean().item()))
                affinity_target_mean.append(float(affinity_target.mean().item()))

    out: Dict[str, float] = {
        "logits_mean": float(np.mean(logits_mean)),
        "logits_std": float(np.mean(logits_std)),
        "sig_gt_05_ratio": float(np.mean(sig_gt_05)),
        "logit_gt_0_ratio": float(np.mean(logit_gt_0)),
        "gt_gt_0_ratio": float(np.mean(gt_gt_0)),
        "gt_gt_05_ratio": float(np.mean(gt_gt_05)),
        "vote_pos_sig05": float(np.mean(vote_pos_sig05)),
        "vote_pos_logit0": float(np.mean(vote_pos_logit0)),
    }
    ps = np.stack(pair_sig05, axis=0).mean(axis=0)
    pl = np.stack(pair_logit0, axis=0).mean(axis=0)
    out.update({
        "pair_sig05_0_7": float(ps[0]),
        "pair_sig05_1_6": float(ps[1]),
        "pair_sig05_2_5": float(ps[2]),
        "pair_sig05_3_4": float(ps[3]),
        "pair_logit0_0_7": float(pl[0]),
        "pair_logit0_1_6": float(pl[1]),
        "pair_logit0_2_5": float(pl[2]),
        "pair_logit0_3_4": float(pl[3]),
    })
    if score_target_mean:
        out["score_target_mean"] = float(np.mean(score_target_mean))
        out["affinity_target_mean"] = float(np.mean(affinity_target_mean))
    return out


def write_markdown_report(
    report_path: Path,
    output_root: Path,
    ckpt_dir: Path,
    results_csv: Path,
    epochwise_csv: Path,
    meta_path: Path,
    epochwise_rows: List[EpochResult],
    best_from_results: Optional[Tuple[int, Dict[str, float]]],
    detailed_ckpt: Path,
    detailed_stats: Dict[str, float],
):
    lines: List[str] = []
    lines.append("# Dist Debug Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- output_root: `{output_root}`")
    lines.append(f"- checkpoint_dir: `{ckpt_dir}`")
    lines.append(f"- results_csv: `{results_csv}` (exists={results_csv.exists()})")
    lines.append(f"- epochwise_csv: `{epochwise_csv}` (exists={epochwise_csv.exists()})")
    lines.append(f"- best_model_meta(from solver): `{meta_path}` (exists={meta_path.exists()})")
    lines.append("")

    lines.append("## Epochwise Snapshot")
    lines.append("| checkpoint | epoch_key | vote_pos_logit0 | collapsed | logit_gt_0_ratio | pair_1_6 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in epochwise_rows:
        lines.append(
            f"| {r.checkpoint} | {r.epoch_key} | {r.vote_pos_logit0:.6f} | {r.collapsed_voting} | "
            f"{r.logit_gt_0_ratio:.6f} | {r.pair_1_6:.6f} |"
        )
    lines.append("")

    lines.append("## Best From Results CSV")
    if best_from_results is None:
        lines.append("- not available")
    else:
        ep, m = best_from_results
        cldc = "nan" if np.isnan(m["cldice"]) else f"{m['cldice']:.6f}"
        lines.append(f"- best_epoch_from_results: `{ep}`")
        lines.append(f"- dice: `{m['dice']:.6f}`")
        lines.append(f"- train_loss: `{m['train_loss']:.6f}`")
        lines.append(f"- val_loss: `{m['val_loss']:.6f}`")
        lines.append(f"- jac: `{m['jac']:.6f}`")
        lines.append(f"- clDice: `{cldc}`")
    lines.append("")

    lines.append("## Detailed Checkpoint Analysis")
    lines.append(f"- checkpoint: `{detailed_ckpt}`")
    for k in sorted(detailed_stats.keys()):
        v = detailed_stats[k]
        lines.append(f"- {k}: `{v:.6f}`")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.num_class != 1:
        raise ValueError("num_class must be 1 for this unified debug script.")

    output_root = Path(args.output_root)
    ckpt_dir = output_root / "models" / str(args.exp_index)
    results_csv = output_root / f"results_{args.exp_index}.csv"
    epochwise_csv = ckpt_dir / args.epochwise_csv_name
    meta_txt = ckpt_dir / args.meta_name
    report_md = ckpt_dir / args.report_name

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_ids, test_ids = make_chase_split(args.exp_id)
    pat_ls = train_ids if args.split == "train" else test_ids
    mode = "train" if args.split == "train" else "test"
    dataset_args = argparse.Namespace(resize=args.resize, num_class=args.num_class)
    dataset = MyDataset_CHASE(dataset_args, train_root=args.data_root, pat_ls=pat_ls, mode=mode, label_mode=args.label_mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=args.num_workers,
    )

    ckpts = collect_checkpoints(ckpt_dir, include_best=args.include_best)
    if args.max_checkpoints > 0:
        ckpts = ckpts[:args.max_checkpoints]
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

    model = DconnNet(
        num_class=args.num_class,
        conn_num=args.conn_num,
        conn_layout=args.conn_layout,
    ).to(device)
    h, w = args.resize
    hori, vert = build_translation(args.num_class, h, w, device)

    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] checkpoints={len(ckpts)} batches={min(args.max_batches, len(loader))}")

    epochwise_rows: List[EpochResult] = []
    for ckpt in ckpts:
        row = evaluate_epoch_checkpoint(
            model=model,
            ckpt_path=ckpt,
            loader=loader,
            device=device,
            hori=hori,
            vert=vert,
            conn_num=args.conn_num,
            conn_channels=args.conn_channels,
            conn_layout=args.conn_layout,
            max_batches=args.max_batches,
        )
        epochwise_rows.append(row)
        print(
            f"[EVAL] {row.checkpoint}: vote_pos_logit0={row.vote_pos_logit0:.6f}, "
            f"collapsed={row.collapsed_voting}, logit_gt_0={row.logit_gt_0_ratio:.6f}"
        )

    write_epochwise_csv(epochwise_csv, epochwise_rows)
    print(f"[INFO] wrote {epochwise_csv}")

    results = parse_results_csv(results_csv)
    best_from_results = select_best_epoch(results)
    best_debug = next((r for r in epochwise_rows if r.checkpoint == "best_model.pth"), None)

    detailed_ckpt = ckpt_dir / args.detailed_checkpoint
    if not detailed_ckpt.exists():
        # fallback to best_model if requested path missing
        alt = ckpt_dir / "best_model.pth"
        if alt.exists():
            detailed_ckpt = alt
        else:
            detailed_ckpt = ckpts[0]
    detailed_stats = evaluate_single_checkpoint_detailed(
        model=model,
        ckpt_path=detailed_ckpt,
        loader=loader,
        device=device,
        hori=hori,
        vert=vert,
        conn_num=args.conn_num,
        conn_channels=args.conn_channels,
        conn_layout=args.conn_layout,
        max_batches=args.max_batches,
        tau=args.tau,
        sigma=args.sigma,
        label_mode=args.label_mode,
    )

    if meta_txt.exists():
        print(f"[INFO] kept existing solver-generated meta: {meta_txt}")
    else:
        print(f"[WARN] solver-generated meta not found: {meta_txt}")

    write_markdown_report(
        report_path=report_md,
        output_root=output_root,
        ckpt_dir=ckpt_dir,
        results_csv=results_csv,
        epochwise_csv=epochwise_csv,
        meta_path=meta_txt,
        epochwise_rows=epochwise_rows,
        best_from_results=best_from_results,
        detailed_ckpt=detailed_ckpt,
        detailed_stats=detailed_stats,
    )
    print(f"[INFO] wrote {report_md}")


if __name__ == "__main__":
    main()
