import csv
import os

_NAN = float("nan")

_MAIN_METRIC_KEYS = [
    ("dice",          True),
    ("jac",           True),
    ("cldice",        True),
    ("precision",     False),
    ("accuracy",      False),
    ("betti_error_0", True),
    ("betti_error_1", True),
    ("hd95",          False),
]

_STD_METRIC_KEYS = [
    "dice_std", "jac_std", "cldice_std", "precision_std",
    "accuracy_std", "betti_error_0_std", "betti_error_1_std", "hd95_std",
]


def _fmt_metric(metrics, key, required=False):
    v = metrics[key] if required else metrics.get(key, _NAN)
    return f'{float(v):0.6f}'


def _format_main_metrics(metrics):
    return [_fmt_metric(metrics, k, req) for k, req in _MAIN_METRIC_KEYS]


def _format_std_metrics(metrics):
    return [_fmt_metric(metrics, k) for k in _STD_METRIC_KEYS]


def create_exp_directory(save_dir):
    exp_model_dir = os.path.join(save_dir, "models")
    if not os.path.exists(exp_model_dir):
        os.makedirs(exp_model_dir)

    results_csv = os.path.join(save_dir, "results.csv")
    if not os.path.exists(results_csv):
        with open(results_csv, "w") as f:
            f.write(
                "epoch,train_loss,val_loss,dice,Jac,clDice,precision,accuracy,betti_error_0,betti_error_1,hd95,"
                "dice_std,Jac_std,clDice_std,precision_std,accuracy_std,betti_error_0_std,betti_error_1_std,hd95_std,elapsed_hms\n"
            )


def write_epoch_result_row(save_dir, epoch, metrics, elapsed_hms=""):
    results_csv = "results.csv"
    with open(os.path.join(save_dir, results_csv), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                f"{int(epoch):03d}",
                f'{float(metrics["train_loss"]):0.6f}',
                f'{float(metrics["val_loss"]):0.6f}',
                *_format_main_metrics(metrics),
                *_format_std_metrics(metrics),
                elapsed_hms or "",
            ]
        )


def write_eval_summary(
    save_dir,
    split_name,
    metrics,
    checkpoint_name="",
    evaluated_split="",
    eval_epoch="",
    elapsed_hms="",
):
    summary_csv = os.path.join(save_dir, f"{split_name}_results.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "result_type",
                "evaluated_split",
                "eval_epoch",
                "checkpoint",
                "train_loss",
                "eval_loss",
                "dice",
                "jac",
                "cldice",
                "precision",
                "accuracy",
                "betti_error_0",
                "betti_error_1",
                "hd95",
                "elapsed_hms",
                "dice_std",
                "jac_std",
                "cldice_std",
                "precision_std",
                "accuracy_std",
                "betti_error_0_std",
                "betti_error_1_std",
                "hd95_std",
            ]
        )
        writer.writerow(
            [
                split_name,
                evaluated_split,
                eval_epoch,
                checkpoint_name,
                f'{float(metrics["train_loss"]):0.6f}',
                f'{float(metrics["val_loss"]):0.6f}',
                *_format_main_metrics(metrics),
                elapsed_hms or "",
                *_format_std_metrics(metrics),
            ]
        )


_RESULT_TXT_ROWS = [
    ("Dice ↑",   "dice",          "dice_std",          "0.4f"),
    ("IoU ↑",    "jac",           "jac_std",           "0.4f"),
    ("clDice ↑", "cldice",        "cldice_std",        "0.4f"),
    ("β0 ↓",     "betti_error_0", "betti_error_0_std", "0.2f"),
    ("β1 ↓",     "betti_error_1", "betti_error_1_std", "0.2f"),
    ("HD95 ↓",   "hd95",          "hd95_std",          "0.4f"),
]


def write_result_txt(
    save_dir,
    dataset,
    split_name,
    metrics,
    checkpoint_name="",
    evaluated_split="",
    eval_epoch="",
    elapsed_hms="",
):
    """Write a human-readable summary of the final eval metrics, in the same
    Dice/IoU/clDice/β0/β1/HD95 layout as the paper's results table."""
    def _fmt(key, std_key, fmt):
        mean = metrics.get(key, _NAN)
        std = metrics.get(std_key, _NAN)
        return f'{float(mean):{fmt}} ± {float(std):{fmt}}'

    lines = [
        "TMAL evaluation summary",
        "=" * 40,
        f"Dataset:    {dataset}",
        f"Split:      {split_name} (evaluated_split={evaluated_split})",
        f"Checkpoint: {checkpoint_name}",
        f"Eval epoch: {eval_epoch}",
        f"Elapsed:    {elapsed_hms}",
        "",
        f"{'Metric':<10} Mean ± Std",
        "-" * 40,
    ]
    for label, key, std_key, fmt in _RESULT_TXT_ROWS:
        lines.append(f"{label:<10} {_fmt(key, std_key, fmt)}")
    lines.append("-" * 40)

    with open(os.path.join(save_dir, "result.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def save_best_checkpoint(
    save_dir,
    model_state_dict,
    optimizer_state_dict,
    current_epoch,
    stop_state,
    monitor_metric,
    epoch_metrics,
):
    best_model_dir = os.path.join(save_dir, "models")
    best_model_path = os.path.join(best_model_dir, "best_model.pth")
    checkpoint_best_path = os.path.join(best_model_dir, "checkpoint_best.pth")

    import torch

    torch.save(model_state_dict, best_model_path)
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "epoch": int(current_epoch),
            "best_val_dice": float(stop_state["best_val_dice"]),
            "best_val_loss": float(stop_state["best_val_loss"]),
            "monitor_metric": monitor_metric,
            "monitor_value": float(stop_state["best_monitor"]),
            "tie_break_with_loss": bool(stop_state["used_tie_break"]),
        },
        checkpoint_best_path,
    )
    with open(os.path.join(best_model_dir, "best_model_meta.txt"), "w") as f:
        f.write(f"best_epoch={current_epoch}\n")
        f.write(f'best_val_dice={stop_state["best_val_dice"]:.6f}\n')
        f.write(f'best_val_loss={stop_state["best_val_loss"]:.6f}\n')
        f.write(f"monitor_metric={monitor_metric}\n")
        f.write(f'best_monitor_value={stop_state["best_monitor"]:.6f}\n')
        f.write(f'used_tie_break_with_loss={bool(stop_state["used_tie_break"])}\n')
        f.write(f'best_train_loss={epoch_metrics["train_loss"]:.6f}\n')
        f.write(f'best_jac={epoch_metrics["jac"]:.6f}\n')
        f.write(f'best_clDice={epoch_metrics["cldice"]:.6f}\n')
        f.write(f'best_betti_error_0={epoch_metrics["betti_error_0"]:.6f}\n')
        f.write(f'best_betti_error_1={epoch_metrics["betti_error_1"]:.6f}\n')
        f.write(f'best_hd95={epoch_metrics.get("hd95", float("nan")):.6f}\n')


def save_resume_checkpoint(save_dir, model_state_dict, optimizer_state_dict, epoch, global_step):
    import torch
    path = os.path.join(save_dir, 'models', 'latest_checkpoint.pth')
    torch.save({
        'epoch': int(epoch),
        'global_step': int(global_step),
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
    }, path)


def load_resume_checkpoint(save_dir):
    """Return (start_epoch, global_step, ckpt_dict) or (0, 0, None) if no checkpoint found.

    Primary: latest_checkpoint.pth (full state: epoch, global_step, model + optimizer).
    Fallback: highest {N}_model.pth in models/ (model weights only, optimizer unavailable).
    """
    import re
    import torch

    primary_path = os.path.join(save_dir, 'models', 'latest_checkpoint.pth')
    if os.path.exists(primary_path):
        ckpt = torch.load(primary_path, map_location='cpu')
        return int(ckpt['epoch']), int(ckpt.get('global_step', 0)), ckpt

    models_dir = os.path.join(save_dir, 'models')
    if not os.path.isdir(models_dir):
        return 0, 0, None

    pattern = re.compile(r'^(\d+)_model\.pth$')
    candidates = [
        (int(m.group(1)), fname)
        for fname in os.listdir(models_dir)
        if (m := pattern.match(fname))
    ]
    if not candidates:
        return 0, 0, None

    best_epoch, best_fname = max(candidates, key=lambda x: x[0])
    best_path = os.path.join(models_dir, best_fname)
    print(f'[RESUME] No latest_checkpoint.pth found. Falling back to {best_fname} (epoch {best_epoch}).')

    raw_state_dict = torch.load(best_path, map_location='cpu')
    synthetic_ckpt = {
        'epoch': best_epoch,
        'global_step': 0,
        'model_state_dict': raw_state_dict,
        'optimizer_state_dict': None,
    }
    return best_epoch, 0, synthetic_ckpt


def load_best_checkpoint(save_dir, net, val_loader, final_eval_split):
    """Load best_model.pth if it exists; otherwise keep last-epoch weights.
    Returns (net, checkpoint_name)."""
    import torch
    best_model_path = os.path.join(save_dir, 'models', 'best_model.pth')
    if val_loader is not None and os.path.exists(best_model_path):
        print(f'LOAD BEST MODEL FOR FINAL {final_eval_split.upper()} EVAL: {best_model_path}')
        net.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        net = net.cuda()
        return net, 'best_model.pth'
    if val_loader is None:
        print(f'VAL LOADER IS NONE; SKIP BEST MODEL SELECTION. USE LAST EPOCH MODEL FOR FINAL {final_eval_split.upper()} EVAL.')
    else:
        print(f'BEST MODEL NOT FOUND; USE LAST EPOCH MODEL FOR FINAL {final_eval_split.upper()} EVAL.')
    return net, 'last_epoch_model'


def build_sample_metric_row(
    epoch, batch_idx, sample_in_batch, sample_name,
    val_loss, dice, jac, cldice, precision, accuracy,
    betti_error_0, betti_error_1, hd95=float('nan'),
):
    return {
        'epoch': int(epoch),
        'batch': int(batch_idx),
        'sample_in_batch': int(sample_in_batch),
        'sample_name': sample_name,
        'val_loss': float(val_loss),
        'dice': float(dice),
        'jac': float(jac),
        'cldice': float(cldice),
        'precision': float(precision),
        'accuracy': float(accuracy),
        'betti_error_0': float(betti_error_0),
        'betti_error_1': float(betti_error_1),
        'hd95': float(hd95),
    }


def write_sample_metrics_csv(sample_csv, sample_metric_rows):
    """Append per-sample metric rows to the split's sample metrics CSV."""
    write_header = not os.path.exists(sample_csv)
    fieldnames = [
        'epoch', 'batch', 'sample_in_batch', 'sample_name', 'val_loss',
        'dice', 'jac', 'cldice', 'precision', 'accuracy',
        'betti_error_0', 'betti_error_1', 'hd95',
    ]
    with open(sample_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(sample_metric_rows)
