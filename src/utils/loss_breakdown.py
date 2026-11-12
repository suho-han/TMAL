import csv
import os
from collections import OrderedDict


class LossBreakdownLogger:
    """Per-epoch averaged loss-term writer.

    Fork-specific diagnostic. Default disabled. When enabled by `--debug_loss_breakdown`,
    accumulates per-batch loss term values across an epoch and writes a single row per epoch
    to `<save_dir>/loss_breakdown.csv`. The schema grows monotonically; previously
    written rows are not rewritten when new keys appear, so a single CSV may contain
    rows with different column counts. Reader code should treat missing values as blank.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.csv_path = os.path.join(save_dir, "loss_breakdown.csv")
        self._sums = OrderedDict()
        self._count = 0
        self._known_keys = []
        self._header_written = os.path.exists(self.csv_path)

    def reset(self):
        self._sums = OrderedDict()
        self._count = 0

    def accumulate(self, terms):
        if not terms:
            return
        for key, value in terms.items():
            try:
                fval = float(value)
            except Exception:
                continue
            if key not in self._sums:
                self._sums[key] = 0.0
                if key not in self._known_keys:
                    self._known_keys.append(key)
            self._sums[key] += fval
        self._count += 1

    def flush(self, epoch):
        if self._count == 0:
            return
        averages = OrderedDict()
        for key in self._known_keys:
            total = self._sums.get(key, 0.0)
            averages[key] = total / self._count if self._count > 0 else 0.0

        os.makedirs(self.save_dir, exist_ok=True)
        write_header = not self._header_written
        fieldnames = ["epoch"] + list(averages.keys())
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._header_written = True
            row = {"epoch": int(epoch)}
            for key, val in averages.items():
                row[key] = f"{val:.6f}"
            writer.writerow(row)

        self.reset()
