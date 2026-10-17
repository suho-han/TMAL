import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_paper_table import (  # noqa: E402
    DATASET_CONFIGS,
    resolve_dataset_roots,
)


def test_resolve_dataset_roots_joins_input_roots(tmp_path):
    base_root = (tmp_path / "output").resolve()
    alt_root = (tmp_path / "output_dropone_svariation").resolve()
    dataset_root = (base_root / "drive").resolve()
    roots = resolve_dataset_roots(str(dataset_root), [str(base_root), str(alt_root), str(dataset_root)])
    assert roots == [str(dataset_root), str((alt_root / "drive").resolve())]


def test_drive_proposed_query_is_fixed_experiment_source():
    drive_cfg = next(cfg for cfg in DATASET_CONFIGS if cfg["name"] == "DRIVE")
    proposed_query = next(query for query, display_name in drive_cfg["experiments"] if display_name == "Proposed")
    assert proposed_query == {
        "label_mode": "dist_inverted",
        "conn_num": 8,
        "conn_fusion": "dg_dsf",
        "fusion_branches": "io",
        "loss": "smooth_l1",
        "seg_aux_weight": 0.5,
        "conn_aux_c3_weight": 0.5,
        "conn_aux_c5_weight": 0.3,
        "final_decision": "max",
    }
