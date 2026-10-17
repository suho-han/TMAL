import math
import os
import tempfile


class SafeSummaryWriter:
    """Best-effort TensorBoard writer that disables itself on write failures."""

    def __init__(self, writer=None, log_dir=None, disabled_reason=None):
        self.writer = writer
        self.log_dir = log_dir
        self.disabled = writer is None
        self._warned = False
        if disabled_reason is not None:
            self._disable(disabled_reason)

    def _disable(self, exc):
        self.disabled = True
        if not self._warned:
            print(f"[WARN] TensorBoard logging disabled for {self.log_dir}: {exc}")
            self._warned = True
        writer = self.writer
        self.writer = None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    def add_scalar(self, *args, **kwargs):
        if self.disabled or self.writer is None:
            return
        try:
            self.writer.add_scalar(*args, **kwargs)
        except Exception as exc:
            self._disable(exc)

    def close(self):
        if self.writer is None:
            return
        try:
            self.writer.close()
        except Exception as exc:
            self._disable(exc)


def create_safe_summary_writer(summary_writer_cls, log_dir):
    try:
        os.makedirs(log_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".tb_write_test_", dir=log_dir, delete=True):
            pass
        writer = summary_writer_cls(log_dir=log_dir)
        return SafeSummaryWriter(writer=writer, log_dir=log_dir)
    except Exception as exc:
        return SafeSummaryWriter(log_dir=log_dir, disabled_reason=exc)


def is_nan_metric(value):
    return math.isnan(float(value))


def format_elapsed_hms(elapsed_seconds):
    if elapsed_seconds is None or is_nan_metric(elapsed_seconds):
        return ""

    total_seconds = max(0, int(round(float(elapsed_seconds))))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_epoch_postfix(mean_train_loss, val_loader, epoch_metrics, early_stopper, dist_edge_stats):
    epoch_postfix = {"train_loss": f"{mean_train_loss:.6f}"}
    if val_loader is not None:
        epoch_postfix["val_loss"] = f'{float(epoch_metrics["val_loss"]):.6f}'
        epoch_postfix["val_dice"] = f'{float(epoch_metrics["dice"]):.6f}'
        if early_stopper is not None and not is_nan_metric(early_stopper.best_val_dice):
            epoch_postfix["best_val_dice"] = f"{float(early_stopper.best_val_dice):.6f}"
        else:
            epoch_postfix["best_val_dice"] = "nan"
        if early_stopper is not None and early_stopper.enabled:
            epoch_postfix["es"] = f"{int(early_stopper.counter)}/{int(early_stopper.patience)}"
        else:
            epoch_postfix["es"] = "disabled"
    elif dist_edge_stats is not None:
        epoch_postfix["edge_mean"] = f'{float(dist_edge_stats["edge_mean"]):.6f}'
        epoch_postfix["edge_nonzero"] = f'{float(dist_edge_stats["edge_nonzero_ratio"]):.6f}'
    return epoch_postfix


def log_epoch_scalars(writer, current_epoch, mean_train_loss, epoch_metrics,
                      early_stopper=None, val_loader=None, dist_edge_stats=None):
    """Write per-epoch scalar summaries to TensorBoard."""
    writer.add_scalar('epoch/train_loss', float(mean_train_loss), current_epoch)
    for metric in ['val_loss', 'dice', 'jac', 'cldice', 'precision', 'accuracy',
                   'betti_error_0', 'betti_error_1', 'hd95']:
        v = epoch_metrics.get(metric, float('nan'))
        if not is_nan_metric(v):
            writer.add_scalar(f'epoch/{metric}', float(v), current_epoch)
    for key, value in epoch_metrics.get('val_loss_terms', {}).items():
        writer.add_scalar(f'val/{key}', float(value), current_epoch)
    if early_stopper is not None and val_loader is not None:
        if not is_nan_metric(early_stopper.best_val_dice):
            writer.add_scalar('epoch/best_val_dice', float(early_stopper.best_val_dice), current_epoch)
        writer.add_scalar('epoch/early_stopping_counter', float(early_stopper.counter), current_epoch)
    if dist_edge_stats is not None:
        writer.add_scalar('train/edge_mean', float(dist_edge_stats['edge_mean']), current_epoch)
        writer.add_scalar('train/edge_nonzero_ratio', float(dist_edge_stats['edge_nonzero_ratio']), current_epoch)


def build_progress_prefix(args):
    dataset_name = str(getattr(args, 'dataset', 'unknown'))
    output_dir_name = str(getattr(args, 'output_dir', '') or getattr(args, 'save', 'output'))
    return f'({dataset_name}) *({output_dir_name})'


def resolve_training_runtime_configs(args):
    monitoring_config = getattr(args, 'monitoring_config', {}) or {}
    early_stopping_config = getattr(args, 'early_stopping_config', {}) or {}
    checkpoint_config = getattr(args, 'checkpoint_config', {}) or {}

    monitor_metric = monitoring_config.get('monitor_metric', getattr(args, 'monitor_metric', 'val_dice'))
    if monitor_metric not in ('val_dice', 'val_loss'):
        raise ValueError(f"Unsupported monitor_metric={monitor_metric!r}")

    save_best_only = bool(checkpoint_config.get('save_best_only', getattr(args, 'save_best_only', False)))
    save_per_epochs = int(checkpoint_config.get('save_per_epochs', getattr(args, 'save_per_epochs', 50)))
    if save_per_epochs < 1:
        raise ValueError('save_per_epochs must be >= 1')

    early_stopping_kwargs = {
        'patience': int(early_stopping_config.get('patience', getattr(args, 'early_stopping_patience', 20))),
        'min_delta': float(early_stopping_config.get('min_delta', getattr(args, 'early_stopping_min_delta', 0.001))),
        'tie_break_with_loss': bool(early_stopping_config.get('tie_break_with_loss', getattr(args, 'tie_break_with_loss', True))),
        'tie_eps': float(early_stopping_config.get('tie_eps', getattr(args, 'early_stopping_tie_eps', 1e-4))),
        'stop_interval': int(early_stopping_config.get('stop_interval', getattr(args, 'early_stopping_stop_interval', 10))),
    }
    return monitor_metric, save_best_only, save_per_epochs, early_stopping_kwargs


class EarlyStopping:
    def __init__(
        self,
        monitor_metric="val_dice",
        mode="max",
        patience=20,
        min_delta=0.001,
        tie_break_with_loss=True,
        tie_eps=1e-4,
        stop_interval=10,
    ):
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.tie_break_with_loss = bool(tie_break_with_loss)
        self.tie_eps = float(tie_eps)
        self.stop_interval = max(1, int(stop_interval))

        self.enabled = self.patience > 0
        self.best_monitor = float("nan")
        self.best_val_dice = float("nan")
        self.best_val_loss = float("nan")
        self.best_epoch = 0
        self.counter = 0
        self.pending_stop = False

    def _is_improved(self, monitor_value):
        if self.best_epoch == 0:
            return not is_nan_metric(monitor_value)
        if is_nan_metric(monitor_value):
            return False
        if self.mode == "max":
            return monitor_value > (self.best_monitor + self.min_delta)
        return monitor_value < (self.best_monitor - self.min_delta)

    def _is_tie_break_improved(self, val_dice, val_loss):
        if self.monitor_metric != "val_dice":
            return False
        if not self.tie_break_with_loss:
            return False
        if self.best_epoch == 0:
            return False
        if is_nan_metric(val_dice) or is_nan_metric(val_loss) or is_nan_metric(self.best_val_loss):
            return False
        if abs(float(val_dice) - float(self.best_val_dice)) > self.tie_eps:
            return False
        return float(val_loss) < float(self.best_val_loss)

    def step(self, monitor_value, val_dice, val_loss, epoch):
        improved = self._is_improved(monitor_value)
        used_tie_break = False
        if not improved and self._is_tie_break_improved(val_dice, val_loss):
            improved = True
            used_tie_break = True

        if improved:
            self.best_monitor = float(monitor_value)
            self.best_val_dice = float(val_dice)
            self.best_val_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.counter = 0
            self.pending_stop = False
        else:
            if self.enabled:
                self.counter += 1
                if self.counter >= self.patience:
                    self.pending_stop = True

        should_stop = bool(self.enabled and self.pending_stop and (int(epoch) % self.stop_interval == 0))
        waiting_for_boundary = bool(self.enabled and self.pending_stop and not should_stop)

        return {
            "improved": improved,
            "used_tie_break": used_tie_break,
            "counter": int(self.counter),
            "best_monitor": float(self.best_monitor),
            "best_val_dice": float(self.best_val_dice),
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": int(self.best_epoch),
            "should_stop": should_stop,
            "waiting_for_boundary": waiting_for_boundary,
        }
