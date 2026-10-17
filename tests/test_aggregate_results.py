import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_results import (
    _experiment_mean_unique_key,
    _format_fusion_display_name,
    _format_segaux_column_value,
    _load_external_baselines_for_dataset,
    aggregate_output_stem,
    build_experiment_name_from_runtime_config,
    discover_target_roots,
    experiment_sort_key,
    filter_rows_for_aggregate_scope,
    normalize_proposed_suffix,
    parse_experiment_metadata,
    parse_input_roots,
    should_exclude_ambiguous_scaled_sum_row,
    should_include_dataset_in_default_aggregate,
    write_ablation_latex,
    write_experiment_mean_dataset_tables_latex,
    write_experiment_mean_latex,
)

def _row(
    dataset,
    experiment,
    conn_layout="default",
    conn_fusion="none",
    seg_aux_weight=None,
    seg_aux_variant="none",
):
    return {
        "dataset": dataset,
        "experiment": experiment,
        "conn_num": 8,
        "conn_layout": conn_layout,
        "conn_fusion": conn_fusion,
        "seg_aux_weight": seg_aux_weight,
        "seg_aux_variant": seg_aux_variant,
        "loss": "bce",
        "num_folds": 5,
        "best_dice": 0.9,
        "best_dice_std": 0.01,
        "best_jac": 0.8,
        "best_jac_std": 0.01,
        "best_accuracy": 0.95,
        "best_accuracy_std": 0.01,
        "best_precision": 0.96,
        "best_precision_std": 0.01,
        "best_cldice": 0.7,
        "best_cldice_std": 0.01,
        "best_betti_error_0": 0.1,
        "best_betti_error_0_std": 0.01,
        "best_betti_error_1": 0.2,
        "best_betti_error_1_std": 0.01,
        "best_hd95": 1.5,
        "best_hd95_std": 0.1,
    }


def test_experiment_mean_dataset_tables_are_grouped_by_dataset_family(tmp_path):
    rows_by_dataset = {
        "isic": [_row("isic", "isic_exp")],
        "chase": [_row("chase", "chase_exp")],
        "drive": [_row("drive", "drive_exp")],
        "cremi": [_row("cremi", "cremi_exp")],
        "octa500-6M": [_row("octa500-6M", "octa6m_exp")],
        "octa500-3M": [_row("octa500-3M", "octa3m_exp")],
    }

    out_path = Path(tmp_path) / "grouped.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()

    cremi_idx = text.index(r"\section*{CREMI}")
    drive_idx = text.index(r"\section*{DRIVE}")
    chase_idx = text.index(r"\section*{CHASE}")
    isic_idx = text.index(r"\section*{ISIC}")
    octa_idx = text.index(r"\section*{OCTA3M\&6M}")

    assert cremi_idx < drive_idx < chase_idx < isic_idx < octa_idx
    assert text.count(r"\clearpage") >= 5


def test_default_aggregate_scope_includes_only_drive_chase_and_octa_family():
    assert should_include_dataset_in_default_aggregate("drive")
    assert should_include_dataset_in_default_aggregate("CHASEDB1")
    assert should_include_dataset_in_default_aggregate("octa500-3M")
    assert should_include_dataset_in_default_aggregate("octa500-6M")
    assert not should_include_dataset_in_default_aggregate("cremi")
    assert not should_include_dataset_in_default_aggregate("isic2018")


def test_filter_rows_for_aggregate_scope_defaults_to_drive_chase_octa():
    rows = [
        _row("drive", "drive_exp"),
        _row("chase", "chase_exp"),
        _row("octa500-3M", "octa_exp"),
        _row("cremi", "cremi_exp"),
        _row("isic", "isic_exp"),
    ]

    filtered = filter_rows_for_aggregate_scope(rows, include_all=False)

    assert [row["dataset"] for row in filtered] == ["drive", "chase", "octa500-3M"]


def test_filter_rows_for_aggregate_scope_with_all_keeps_all_counted_datasets():
    rows = [
        _row("drive", "drive_exp"),
        _row("chase", "chase_exp"),
        _row("octa500-3M", "octa_exp"),
        _row("cremi", "cremi_exp"),
        _row("isic", "isic_exp"),
        _row("trash", "trash_exp"),
    ]

    filtered = filter_rows_for_aggregate_scope(rows, include_all=True)

    assert [row["dataset"] for row in filtered] == [
        "drive",
        "chase",
        "octa500-3M",
        "cremi",
        "isic",
    ]


def test_aggregate_output_stem_appends_all_suffix_only_for_all_mode():
    assert aggregate_output_stem("summary", include_all=False) == "summary_experiment_means"
    assert aggregate_output_stem("summary", include_all=True) == "summary_experiment_means_all"
    assert (
        aggregate_output_stem("summary_experiment_means", include_all=True)
        == "summary_experiment_means_all"
    )


def test_parse_input_roots_splits_and_dedupes(tmp_path):
    root_a = (tmp_path / "output").resolve()
    root_b = (tmp_path / "output_dropone_svariation").resolve()
    roots = parse_input_roots(f"{root_a}, {root_b}, {root_a}")
    assert roots == [str(root_a), str(root_b)]


def test_discover_target_roots_excludes_smoke_directory(tmp_path):
    real_root = Path(tmp_path) / "drive" / "dist_8_smooth_l1"
    smoke_root = Path(tmp_path) / "_smoke" / "drive" / "dist_8_smooth_l1"
    real_root.mkdir(parents=True, exist_ok=True)
    smoke_root.mkdir(parents=True, exist_ok=True)
    (real_root / "final_results_1.csv").write_text("best_epoch,best_dice\n1,0.9\n")
    (smoke_root / "final_results_1.csv").write_text("best_epoch,best_dice\n1,0.1\n")

    roots = discover_target_roots(str(tmp_path), ["1"], "final_results_{fold}.csv")
    roots_set = {Path(p).resolve() for p in roots}

    assert real_root.resolve() in roots_set
    assert smoke_root.resolve() not in roots_set


def test_out8_conn_label_is_rendered_as_prime(tmp_path):
    rows_by_dataset = {
        "drive": [_row("drive", "binary", conn_layout="out8")],
    }

    out_path = Path(tmp_path) / "out8.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"binary & 8' & none & none & -- & -- & \textbf{0.9000}" in text


def test_external_baselines_include_cape_for_drive_and_deepcrack(tmp_path):
    for dataset_name in ("drive", "deepcrack"):
        cape_dir = Path(tmp_path) / "CAPE" / dataset_name
        cape_dir.mkdir(parents=True, exist_ok=True)
        (cape_dir / "final_results.csv").write_text(
            "dice,jac,cldice,betti_error_0,betti_error_1,hd95\n"
            "0.91,0.82,0.73,0.11,0.22,1.4\n"
        )

    drive_baselines = _load_external_baselines_for_dataset("DRIVE", baselines_root=str(tmp_path))
    deepcrack_baselines = _load_external_baselines_for_dataset("DeepCrack", baselines_root=str(tmp_path))

    assert any(label == "CAPE" for label, _ in drive_baselines)
    assert any(label == "CAPE" for label, _ in deepcrack_baselines)


def test_fusion_experiment_metadata_is_parsed_and_sorted_by_mode_prefix():
    meta = parse_experiment_metadata("dist_scaled_sum_rs0.3_8_smooth_l1")
    assert meta["experiment"] == "dist_scaled_sum_rs0.3"
    assert meta["label_mode"] == "dist"
    assert meta["conn_num"] == 8
    assert meta["conn_layout"] == "default"
    assert meta["loss"] == "smooth_l1"
    assert meta["conn_fusion"] == "scaled_sum"

    key = experiment_sort_key(
        {
            "experiment": meta["experiment"],
            "conn_num": meta["conn_num"],
            "conn_layout": meta["conn_layout"],
            "loss": meta["loss"],
            "fold_scope": "direct",
            "fold_scope_count": "NA",
        }
    )
    assert key[0] == 1  # dist family ordering


def test_scaled_sum_loss_profile_tag_does_not_block_rs_parsing():
    # Folder names like binary_scaled_sum_A_rs0.1_8_bce have a single-letter
    # loss-profile tag (A) between scaled_sum_ and rs…; the parser must skip it.
    cases = [
        ("binary_scaled_sum_A_rs0.1_8_bce", 0.1),
        ("binary_scaled_sum_A_rs0.2_8_bce", 0.2),
        ("dist_scaled_sum_A_rs0.3_8_smooth_l1", 0.3),
        ("dist_inverted_scaled_sum_A_rs0.5_8_gjml_sf_l1", 0.5),
    ]
    for name, expected_rs in cases:
        meta = parse_experiment_metadata(name)
        assert meta["conn_fusion"] == "scaled_sum", name
        assert meta["fusion_residual_scale"] == expected_rs, name
        assert not should_exclude_ambiguous_scaled_sum_row(meta), name


def test_decoder_guided_is_parsed_as_fusion_objective():
    meta = parse_experiment_metadata("dist_inverted_decoder_guided_8_gjml_sf_l1")
    assert meta["experiment"] == "dist_inverted_decoder_guided"
    assert meta["label_mode"] == "dist_inverted"
    assert meta["conn_num"] == 8
    assert meta["conn_layout"] == "default"
    assert meta["loss"] == "gjml_sf_l1"
    assert meta["conn_fusion"] == "none"


def test_ordered_name_with_segaux_suffix_keeps_conn_and_loss_metadata():
    meta = parse_experiment_metadata("dist_inverted_decoder_guided_8_gjml_sf_l1_segaux")
    assert meta["experiment"] == "dist_inverted_decoder_guided"
    assert meta["label_mode"] == "dist_inverted"
    assert meta["conn_num"] == 8
    assert meta["conn_layout"] == "default"
    assert meta["loss"] == "gjml_sf_l1"
    assert meta["conn_fusion"] == "none"
    assert meta["seg_aux_weight"] is None
    assert meta["seg_aux_variant"] == "segaux"


def test_ordered_name_with_segaux_weight_suffix_parses_numeric_weight():
    meta = parse_experiment_metadata("dist_inverted_decoder_guided_8_gjml_sf_l1_segaux_w0.1")
    assert meta["seg_aux_weight"] == 0.1
    assert meta["seg_aux_variant"] == "w0.1"


def test_experiment_name_suffix_after_segaux_is_ignored_for_metadata():
    meta = parse_experiment_metadata("dist_dg_direct_8_gjml_sf_l1_segaux_w5.0_im")
    assert meta["label_mode"] == "dist"
    assert meta["conn_fusion"] == "dg"
    assert meta["conn_num"] == 8
    assert meta["loss"] == "gjml_sf_l1"
    assert meta["seg_aux_weight"] == 5.0
    assert meta["seg_aux_variant"] == "w5.0(im)"
    assert meta["fusion_branches"] == "im"


def test_fusion_branches_ip_is_parsed_from_ordered_name():
    meta = parse_experiment_metadata("dist_inverted_dg_direct_ip_8_smooth_l1_segaux_w0.5")
    assert meta["conn_fusion"] == "dg_direct"
    assert meta["fusion_branches"] == "ip"
    assert meta["loss"] == "smooth_l1"
    assert meta["seg_aux_weight"] == 0.5


def test_scale3_layout_is_parsed_from_ordered_name():
    meta = parse_experiment_metadata("dist_inverted_8_scale3_8_smooth_l1")
    assert meta["label_mode"] == "dist_inverted"
    assert meta["conn_num"] == 8
    assert meta["conn_layout"] == "scale3_8"
    assert meta["loss"] == "smooth_l1"


def test_dg_direct_is_parsed_as_distinct_fusion_objective():
    meta = parse_experiment_metadata("binary_dg_direct_8_bce_segaux_w0.5")
    assert meta["experiment"] == "binary_dg_direct"
    assert meta["label_mode"] == "binary"
    assert meta["conn_num"] == 8
    assert meta["loss"] == "bce"
    assert meta["conn_fusion"] == "dg"
    assert meta["seg_aux_weight"] == 0.5
    assert meta["seg_aux_variant"] == "w0.5"


def test_dataset_table_keeps_dg_in_fusion_column(tmp_path):
    rows_by_dataset = {
        "cremi": [
            _row(
                "cremi",
                "dist_inverted_decoder_guided_A",
                conn_fusion="dg",
            )
        ],
    }

    out_path = Path(tmp_path) / "dgrf.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"dist\_inverted & 8 & dg & none & -- & -- & \textbf{0.9000}" in text


def test_dataset_table_shows_segaux_weight_in_dec_column(tmp_path):
    rows_by_dataset = {
        "drive": [
            _row(
                "drive",
                "dist_inverted_decoder_guided_A",
                conn_fusion="dg",
                seg_aux_weight=0.1,
                seg_aux_variant="w0.1",
            )
        ],
    }

    out_path = Path(tmp_path) / "segaux_dec.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"dist\_inverted & 8 & dg & w0.1 & -- & -- & \textbf{0.9000}" in text


def test_dataset_table_shows_default_segaux_in_dec_column(tmp_path):
    rows_by_dataset = {
        "drive": [
            _row(
                "drive",
                "dist_inverted_decoder_guided_A",
                conn_fusion="dg",
                seg_aux_weight=None,
                seg_aux_variant="segaux",
            )
        ],
    }

    out_path = Path(tmp_path) / "segaux_default_dec.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"dist\_inverted & 8 & dg & segaux & -- & -- & \textbf{0.9000}" in text


def test_ablation_table_hides_dg_profile_a(tmp_path):
    rows = [
        _row(
            "drive",
            "dist_inverted_decoder_guided_A",
            conn_fusion="dg",
        )
    ]

    out_path = Path(tmp_path) / "ablation_decoder_guided.tex"
    write_ablation_latex(str(out_path), rows)

    text = out_path.read_text()
    assert r"\multirow{2}{*}{No.}" in text
    assert r"dg &  & none" in text
    assert r"dg & A & none" not in text


def test_dataset_table_hides_scaled_sum_profile_a_but_keeps_rs(tmp_path):
    rows_by_dataset = {
        "drive": [
            _row(
                "drive",
                "dist_scaled_sum_rs0.1",
                conn_fusion="scaled_sum",
            )
        ],
    }
    rows_by_dataset["drive"][0]["fusion_residual_scale"] = 0.1

    out_path = Path(tmp_path) / "scaled_sum_summary.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"dist & 8 & scaled\_sum/rs0.1 & none & -- & -- & \textbf{0.9000}" in text
    assert r"scaled\_sum/A/rs0.1" not in text


def test_ablation_table_hides_scaled_sum_profile_a_but_keeps_rs(tmp_path):
    rows = [
        _row(
            "drive",
            "dist_scaled_sum_rs0.1",
            conn_fusion="scaled_sum",
        )
    ]
    rows[0]["fusion_residual_scale"] = 0.1

    out_path = Path(tmp_path) / "ablation_scaled_sum.tex"
    write_ablation_latex(str(out_path), rows)

    text = out_path.read_text()
    assert r"scaled\_sum & rs0.1 & none" in text
    assert r"scaled\_sum & A/rs0.1 & none" not in text


def test_experiment_sort_key_follows_label_conn_fusion_dec_loss_order():
    row_a = _row(
        "drive",
        "dist_exp",
        conn_fusion="decoder_guided",
    )
    row_a["loss"] = "gjml_sf_l1"

    row_b = _row(
        "drive",
        "dist_exp",
        conn_fusion="none",
    )
    row_b["loss"] = "smooth_l1"

    # With requested order (label_mode, conn, fusion, SegAux, loss),
    # "fusion=none" should come before decoder-guided entry.
    assert experiment_sort_key(row_b) < experiment_sort_key(row_a)


def test_experiment_mean_unique_key_keeps_suffix_variants_distinct():
    base = _row(
        "cremi",
        "dist_inverted_decoder_guided",
        conn_fusion="decoder_guided",
    )
    base["loss"] = "gjml_sf_l1"
    base["fold_scope"] = "direct"
    base["experiment_source"] = "dist_inverted_decoder_guided_8_gjml_sf_l1"

    segaux = dict(base)
    segaux["experiment_source"] = "dist_inverted_decoder_guided_8_gjml_sf_l1_segaux"

    assert _experiment_mean_unique_key(base) != _experiment_mean_unique_key(segaux)


def test_ablation_latex_includes_conn_fusion_and_fusion_loss_table(tmp_path):
    rows = [
        _row(
            "drive",
            "binary_gate",
            conn_fusion="gate",
        ),
        _row(
            "drive",
            "binary_gate",
            conn_fusion="gate",
        ),
        _row(
            "drive",
            "binary_conv_residual",
            conn_fusion="conv_residual",
        ),
    ]

    out_path = Path(tmp_path) / "ablation.tex"
    write_ablation_latex(str(out_path), rows)

    text = out_path.read_text()
    assert r"\caption{Ablation on Fusion + Decoder/SegAux (DRIVE)}" in text
    assert r"\multirow{2}{*}{Conn. Fusion}" in text
    assert r"\multirow{2}{*}{Fusion Spec}" in text
    assert r"gate &  & none" in text
    assert r"conv\_residual &  & none" in text


def test_ablation_label_mode_table_keeps_only_three_modes(tmp_path):
    rows = [
        _row("drive", "binary_gate", conn_fusion="gate"),
        _row("drive", "dist_gate", conn_fusion="gate"),
        _row(
            "drive",
            "dist_inverted_gate",
            conn_fusion="gate",
        ),
    ]

    out_path = Path(tmp_path) / "ablation_modes.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    assert r"\caption{Ablation on Label Mode (DRIVE)}" in text
    assert r"binary &" in text
    assert r"dist &" in text
    assert r"dist\_inverted &" in text
    assert r"binary\_gate\_A" not in text
    assert r"dist\_inverted\_gate\_dec\_residual\_gate" not in text
    assert r"\caption{Ablation on Fusion}" not in text
    assert r"\caption{Ablation on Fusion + Decoder/SegAux (DRIVE)}" in text


def test_ablation_fusion_objective_table_order_is_fixed(tmp_path):
    rows = [
        _row("drive", "binary_none", conn_fusion="none"),
        _row(
            "drive",
            "binary_scaled_sum",
            conn_fusion="scaled_sum",
        ),
        _row(
            "drive",
            "binary_gate",
            conn_fusion="gate",
        ),
        _row(
            "drive",
            "binary_conv_residual",
            conn_fusion="conv_residual",
        ),
    ]

    out_path = Path(tmp_path) / "ablation_fusion_order.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    assert r"\caption{Ablation on Fusion}" not in text
    assert r"\caption{Ablation on Fusion + Decoder/SegAux (DRIVE)}" in text
    none_idx = text.index(r"none &  & none")
    conv_idx = text.index(r"conv\_residual &  & none")
    gate_idx = text.index(r"gate &  & none")
    scaled_idx = text.index(r"scaled\_sum &  & none")
    assert none_idx < conv_idx < gate_idx < scaled_idx


def test_ablation_latex_splits_tables_into_requested_dataset_order(tmp_path):
    rows = [
        _row("isic", "dist", conn_fusion="none"),
        _row("chasedb1", "dist", conn_fusion="none"),
        _row("drive", "dist", conn_fusion="none"),
        _row("octa500-6M", "dist", conn_fusion="none"),
        _row("octa500-3M", "dist", conn_fusion="none"),
        _row("cremi", "dist", conn_fusion="none"),
    ]
    for row in rows:
        row["loss"] = "smooth_l1"

    out_path = Path(tmp_path) / "ablation_split.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    chase_caption = r"\caption{Ablation on Loss (CHASE)}"
    drive_caption = r"\caption{Ablation on Loss (DRIVE)}"
    octa6m_caption = r"\caption{Ablation on Loss (OCTA6M)}"
    octa3m_caption = r"\caption{Ablation on Loss (OCTA3M)}"
    others_caption = r"\caption{Ablation on Loss (Others)}"

    chase_idx = text.index(chase_caption)
    drive_idx = text.index(drive_caption)
    octa6m_idx = text.index(octa6m_caption)
    octa3m_idx = text.index(octa3m_caption)
    others_idx = text.index(others_caption)

    assert drive_idx < chase_idx < octa6m_idx < octa3m_idx < others_idx
    assert r"\caption{Ablation on Label Mode (CHASE)}" in text
    assert r"\caption{Ablation on Label Mode (DRIVE)}" in text
    assert r"\caption{Ablation on Label Mode (OCTA6M)}" in text
    assert r"\caption{Ablation on Label Mode (OCTA3M)}" in text
    assert r"\caption{Ablation on Label Mode (Others)}" in text

    others_table_start = text.rfind(r"\begin{tabular}", 0, others_idx)
    others_table_text = text[others_table_start:others_idx]
    assert r"\multicolumn{6}{c}{isic}" in others_table_text
    assert r"\multicolumn{6}{c}{cremi}" in others_table_text
    assert r"\multicolumn{6}{c}{drive}" not in others_table_text


def test_ablation_drop_one_best_and_loss_toggle_use_only_smooth_or_gjml(tmp_path):
    rows = []

    bce_row = _row("drive", "binary_base", conn_fusion="dg_dsf", seg_aux_weight=0.5, seg_aux_variant="w0.5")
    bce_row["loss"] = "bce"
    bce_row["best_dice"] = 0.99
    bce_row["best_jac"] = 0.89
    rows.append(bce_row)

    smooth_row = _row("drive", "dist_smooth", conn_fusion="dg_dsf", seg_aux_weight=0.5, seg_aux_variant="w0.5")
    smooth_row["loss"] = "smooth_l1"
    smooth_row["best_dice"] = 0.95
    smooth_row["best_jac"] = 0.85
    rows.append(smooth_row)

    gjml_row = _row("drive", "dist_gjml", conn_fusion="dg_dsf", seg_aux_weight=0.5, seg_aux_variant="w0.5")
    gjml_row["loss"] = "gjml_sf_l1"
    gjml_row["best_dice"] = 0.93
    gjml_row["best_jac"] = 0.83
    rows.append(gjml_row)

    out_path = Path(tmp_path) / "ablation_drop_one_loss.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    assert r"\caption{Ablation study on drive dataset." in text


def test_ablation_drop_one_fusion_uses_same_conn_plain_none_baseline(tmp_path):
    best_row = _row(
        "drive",
        "dist_inverted_best",
        conn_fusion="decoder_guided",
        seg_aux_weight=0.5,
        seg_aux_variant="w0.5",
    )
    best_row["loss"] = "smooth_l1"
    best_row["best_dice"] = 0.95
    best_row["best_jac"] = 0.85

    same_conn_plain = _row("drive", "dist_inverted_plain")
    same_conn_plain["loss"] = "smooth_l1"
    same_conn_plain["best_dice"] = 0.93
    same_conn_plain["best_jac"] = 0.83

    better_wrong_conn = _row("drive", "dist_inverted_plain_24")
    better_wrong_conn["conn_num"] = 24
    better_wrong_conn["loss"] = "smooth_l1"
    better_wrong_conn["best_dice"] = 0.94
    better_wrong_conn["best_jac"] = 0.84

    out_path = Path(tmp_path) / "ablation_drop_one_fusion.tex"
    write_ablation_latex(str(out_path), [best_row, same_conn_plain, better_wrong_conn])
    text = out_path.read_text()

    assert r"\caption{Ablation study on drive dataset." in text


def test_ablation_drop_one_fusion_is_blank_when_best_is_already_none(tmp_path):
    best_row = _row("isic", "dist_inverted_best")
    best_row["loss"] = "smooth_l1"
    best_row["best_dice"] = 0.95
    best_row["best_jac"] = 0.85

    alt_row = _row(
        "isic",
        "dist_inverted_gate",
        conn_fusion="gate",
    )
    alt_row["loss"] = "smooth_l1"
    alt_row["best_dice"] = 0.94
    alt_row["best_jac"] = 0.84

    out_path = Path(tmp_path) / "ablation_drop_one_fusion_none.tex"
    write_ablation_latex(str(out_path), [best_row, alt_row])
    text = out_path.read_text()

    assert r"\caption{Ablation study on isic dataset." in text


def test_ablation_drop_one_splits_other_datasets_per_dataset(tmp_path):
    isic_row = _row("isic", "dist_isic", conn_fusion="dg_dsf", seg_aux_weight=0.5, seg_aux_variant="w0.5")
    isic_row["loss"] = "smooth_l1"
    chase_row = _row("chase", "dist_chase", conn_fusion="dg_dsf", seg_aux_weight=0.5, seg_aux_variant="w0.5")
    chase_row["loss"] = "smooth_l1"

    out_path = Path(tmp_path) / "ablation_drop_one_other_split.tex"
    write_ablation_latex(str(out_path), [isic_row, chase_row])
    text = out_path.read_text()

    assert r"\caption{Ablation study on isic dataset." in text
    assert r"\caption{Ablation study on chase dataset." in text


def test_ablation_drop_one_label_mode_is_blank_when_best_is_already_binary(tmp_path):
    best_row = _row("cremi", "binary_best", conn_fusion="dg_dsf", seg_aux_weight=0.5, seg_aux_variant="w0.5")
    best_row["loss"] = "smooth_l1"
    best_row["best_dice"] = 0.95
    best_row["best_jac"] = 0.85

    alt_row = _row("cremi", "dist_alt", conn_fusion="dg_dsf", seg_aux_weight=0.5, seg_aux_variant="w0.5")
    alt_row["loss"] = "smooth_l1"
    alt_row["best_dice"] = 0.94
    alt_row["best_jac"] = 0.84

    out_path = Path(tmp_path) / "ablation_drop_one_binary_label.tex"
    write_ablation_latex(str(out_path), [best_row, alt_row])
    text = out_path.read_text()

    assert r"\caption{Ablation study on cremi dataset." in text


def test_ablation_drop_one_dist_to_binary_prefers_fixed_source_row(tmp_path):
    best_row = _row(
        "drive",
        "dist_inverted_dg_direct_8_smooth_l1_segaux_w0.5_s13_proposed",
        conn_fusion="dg_direct",
        seg_aux_weight=0.5,
        seg_aux_variant="w0.5",
    )
    best_row["label_mode"] = "dist_inverted"
    best_row["fusion_branches"] = "ip"
    best_row["loss"] = "smooth_l1"
    best_row["best_dice"] = 0.83
    best_row["best_jac"] = 0.71

    drop_labelmode_row = _row(
        "drive",
        "binary_dg_direct_8_bce_segaux_w0.5_s13_drop_labelmode",
        conn_fusion="dg_direct",
        seg_aux_weight=0.5,
        seg_aux_variant="w0.5(s13_drop_labelmode)",
    )
    drop_labelmode_row["label_mode"] = "binary"
    drop_labelmode_row["fusion_branches"] = "ip"
    drop_labelmode_row["loss"] = "bce"
    drop_labelmode_row["best_dice"] = 0.75
    drop_labelmode_row["best_jac"] = 0.60
    drop_labelmode_row["experiment_source"] = "binary_dg_direct_8_bce_segaux_w0.5_s13_drop_labelmode"

    fixed_binary_row = _row(
        "drive",
        "binary_dg_direct_8_bce_segaux_w0.5",
        conn_fusion="dg_direct",
        seg_aux_weight=0.5,
        seg_aux_variant="w0.5",
    )
    fixed_binary_row["label_mode"] = "binary"
    fixed_binary_row["fusion_branches"] = "none"
    fixed_binary_row["loss"] = "bce"
    fixed_binary_row["best_dice"] = 0.78
    fixed_binary_row["best_jac"] = 0.64
    fixed_binary_row["experiment_source"] = "binary_dg_direct_8_bce_segaux_w0.5"

    out_path = Path(tmp_path) / "ablation_drop_one_dist_to_binary_source.tex"
    write_ablation_latex(str(out_path), [best_row, drop_labelmode_row, fixed_binary_row])
    text = out_path.read_text()

    # The 2x2 ablation table must use fixed_binary_row (dice=0.78) over drop_labelmode_row (dice=0.75)
    assert "0.7800" in text
    assert "0.7500" not in text


def test_ablation_drop_one_appends_fixed_binary8_plain_base_row(tmp_path):
    best_row = _row(
        "drive",
        "dist_inverted_best",
        conn_fusion="decoder_guided",
        seg_aux_weight=0.5,
        seg_aux_variant="w0.5",
    )
    best_row["label_mode"] = "dist_inverted"
    best_row["loss"] = "smooth_l1"
    best_row["best_dice"] = 0.95
    best_row["best_jac"] = 0.85

    baseline_row = _row("drive", "binary_8_base")
    baseline_row["label_mode"] = "binary"
    baseline_row["loss"] = "bce"
    baseline_row["best_dice"] = 0.90
    baseline_row["best_jac"] = 0.80

    out_path = Path(tmp_path) / "ablation_drop_one_base.tex"
    write_ablation_latex(str(out_path), [best_row, baseline_row])
    text = out_path.read_text()

    assert r"\caption{Ablation study on drive dataset." in text


def test_ablation_drop_one_includes_cldice_and_betti_with_error_direction(tmp_path):
    best_row = _row("drive", "dist_best", conn_fusion="dg_dsf", seg_aux_weight=0.5, seg_aux_variant="w0.5")
    best_row["loss"] = "smooth_l1"
    best_row["best_dice"] = 0.95
    best_row["best_jac"] = 0.85
    best_row["best_cldice"] = 0.80
    best_row["best_betti_error_0"] = 0.30
    best_row["best_betti_error_1"] = 0.40

    alt_row = _row("drive", "dist_gjml", conn_fusion="dg_dsf", seg_aux_weight=0.5, seg_aux_variant="w0.5")
    alt_row["loss"] = "gjml_sf_l1"
    alt_row["best_dice"] = 0.93
    alt_row["best_jac"] = 0.83
    alt_row["best_cldice"] = 0.78
    alt_row["best_betti_error_0"] = 0.20
    alt_row["best_betti_error_1"] = 0.55

    out_path = Path(tmp_path) / "ablation_drop_one_metrics.tex"
    write_ablation_latex(str(out_path), [best_row, alt_row])
    text = out_path.read_text()

    std_col = r"\textcolor{gray}{(std)}"
    assert (
        rf"clDice $\uparrow$ & {std_col} & Err $(\beta_0)$ $\downarrow$ & {std_col}"
        rf" & Err $(\beta_1)$ $\downarrow$ & {std_col} & HD95 $\downarrow$ & {std_col}"
    ) in text
    assert r"\multirow{2}{*}{-Loss}" not in text


def test_dataset_tables_ignore_trash_dataset(tmp_path):
    rows_by_dataset = {
        "drive": [_row("drive", "drive_exp")],
        "trash": [_row("trash", "trash_exp")],
    }

    out_path = Path(tmp_path) / "datasets_no_trash.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert "trash" not in text
    assert r"\caption{Cross-experiment mean summary (drive, Loss=bce)}" in text


def test_default_scope_dataset_summary_excludes_cremi_and_isic_sections(tmp_path):
    rows = [
        _row("drive", "drive_exp"),
        _row("chase", "chase_exp"),
        _row("octa500-3M", "octa_exp"),
        _row("cremi", "cremi_exp"),
        _row("isic", "isic_exp"),
    ]
    filtered = filter_rows_for_aggregate_scope(rows, include_all=False)
    rows_by_dataset = {}
    for row in filtered:
        rows_by_dataset.setdefault(row["dataset"], []).append(row)

    out_path = Path(tmp_path) / "default_scope_datasets.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"\section*{DRIVE}" in text
    assert r"\section*{CHASE}" in text
    assert r"\section*{OCTA3M\&6M}" in text
    assert r"\section*{CREMI}" not in text
    assert r"\section*{ISIC}" not in text
    assert text.count(r"\clearpage") >= 2


def test_ablation_tables_ignore_trash_dataset(tmp_path):
    rows = [
        _row("drive", "dist_drive"),
        _row("trash", "dist_trash"),
    ]
    for row in rows:
        row["loss"] = "smooth_l1"

    out_path = Path(tmp_path) / "ablation_no_trash.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    assert "trash" not in text
    assert r"\caption{Ablation on Loss (DRIVE)}" in text


def test_ablation_includes_s_variation_table_for_new_fusion_branches(tmp_path):
    rows = [
        {
            **_row("drive", "dist_inverted_8_smooth_l1_segaux_w0.5"),
            "label_mode": "dist_inverted",
            "conn_num": 8,
            "conn_layout": "default",
            "conn_fusion": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
        },
        {
            **_row("drive", "dist_inverted_dg_dsf_io_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3"),
            "label_mode": "dist_inverted",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
        },
        {
            **_row("drive", "dist_inverted_dg_dsf_ip_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3"),
            "label_mode": "dist_inverted",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "ip",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
        },
        {
            **_row("drive", "dist_inverted_dg_dsf_iopf_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3"),
            "label_mode": "dist_inverted",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "iopf",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
        },
        {
            **_row("drive", "dist_inverted_8_out8_smooth_l1"),
            "label_mode": "dist_inverted",
            "conn_num": 8,
            "conn_layout": "out8",
            "conn_fusion": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": None,
            "seg_aux_variant": "none",
        },
    ]
    out_path = Path(tmp_path) / "ablation_s_variation.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()
    assert r"\caption{Ablation study on $\mathcal{S}$ (Drive dataset)." in text
    assert r"$\mathcal{S}$ & Dice $\uparrow$" in text
    # 1 and 2 are non-fusion baselines, so they render without braces.
    lines = text.splitlines()
    assert any(line.startswith("1 & ") for line in lines)
    assert any(line.startswith("2 & ") for line in lines)
    assert r"\textbf{\{1,2\}} & " in text
    # {1,3} (partial-branch combo) is intentionally excluded from this table.
    assert r"\{1,3\} & " not in text
    assert r"\{1,2,3,4\} & " in text


def test_ablation_include_buckets_filters_drive_deepcrack_only(tmp_path):
    rows = [
        _row("drive", "dist_drive"),
        _row("deepcrack", "dist_deepcrack"),
        _row("chase", "dist_chase"),
    ]
    for row in rows:
        row["loss"] = "smooth_l1"

    out_path = Path(tmp_path) / "ablation_drive_deepcrack_only.tex"
    write_ablation_latex(
        str(out_path),
        rows,
        include_buckets={"DRIVE", "DeepCrack"},
    )
    text = out_path.read_text()

    assert r"\caption{Ablation on Loss (DRIVE)}" in text
    assert r"\caption{Ablation on Loss (DeepCrack)}" in text
    assert r"\caption{Ablation on Loss (CHASE)}" not in text


def test_ablation_s_variation_comes_right_after_drop_one(tmp_path):
    rows = [
        {
            **_row("drive", "dist_inverted_dg_dsf_io_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3"),
            "label_mode": "dist_inverted",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
        },
        {
            **_row("drive", "dist_inverted_dg_dsf_ip_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3"),
            "label_mode": "dist_inverted",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "ip",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
        },
        {
            **_row("drive", "binary_8_bce"),
            "label_mode": "binary",
            "conn_fusion": "none",
            "fusion_branches": "none",
            "loss": "bce",
            "seg_aux_weight": None,
            "seg_aux_variant": "none",
        },
    ]

    out_path = Path(tmp_path) / "ablation_s_after_drop_one.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    drop_one_idx = text.index(r"\caption{Ablation study on drive dataset.")
    s_idx = text.index(r"\caption{Ablation study on $\mathcal{S}$ (Drive dataset).")
    label_mode_idx = text.index(r"\caption{Ablation on Label Mode (DRIVE)}")

    assert drop_one_idx < s_idx < label_mode_idx


def test_ablation_s_variation_for_deepcrack_comes_after_drive_s(tmp_path):
    rows = [
        {
            **_row("drive", "drive_proposed"),
            "label_mode": "dist_inverted",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
        },
        {
            **_row("drive", "drive_s13_proposed"),
            "label_mode": "dist_inverted",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "ip",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
        },
        {
            **_row("deepcrack", "deepcrack_proposed"),
            "label_mode": "dist_inverted",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
        },
        {
            **_row("deepcrack", "deepcrack_s13_proposed"),
            "label_mode": "dist_inverted",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "ip",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
        },
    ]
    out_path = Path(tmp_path) / "ablation_s_variation_drive_deepcrack_order.tex"
    write_ablation_latex(
        str(out_path),
        rows,
        include_buckets={"DRIVE", "DeepCrack"},
    )
    text = out_path.read_text()
    drive_s_idx = text.index(r"\caption{Ablation study on $\mathcal{S}$ (Drive dataset).")
    deepcrack_s_idx = text.index(r"\caption{Ablation study on $\mathcal{S}$ (DeepCrack dataset).")
    drive_label_idx = text.index(r"\caption{Ablation on Label Mode (DRIVE)}")
    assert drive_s_idx < deepcrack_s_idx < drive_label_idx


def test_ablation_s_variation_includes_out8_plain_row_only(tmp_path):
    rows = [
        {
            **_row("drive", "plain_w05"),
            "label_mode": "dist_inverted",
            "conn_num": 8,
            "conn_layout": "default",
            "conn_fusion": "none",
            "fusion_branches": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "best_dice": 0.88,
        },
        {
            **_row("drive", "proposed"),
            "label_mode": "dist_inverted",
            "conn_num": 8,
            "conn_layout": "default",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
            "best_dice": 0.90,
        },
        {
            **_row("drive", "out8_plain"),
            "label_mode": "dist_inverted",
            "conn_num": 8,
            "conn_layout": "out8",
            "conn_fusion": "none",
            "fusion_branches": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": None,
            "seg_aux_variant": "none",
            "best_dice": 0.89,
        },
    ]
    out_path = Path(tmp_path) / "ablation_s_variation_out8_conn24.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()
    lines = text.splitlines()
    # 1 and 2 are non-fusion baselines, so they render without braces.
    assert any(line.startswith("1 & ") for line in lines)
    assert r"\textbf{\{1,2\}} & " in text
    assert any(line.startswith("2 & ") for line in lines)
    assert r"\{1+2\}" not in text


def test_ablation_s_variation_uses_non_s_w0_5_for_1_2(tmp_path):
    rows = [
        {
            **_row("drive", "dist_inverted_dg_dsf_io_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3"),
            "label_mode": "dist_inverted",
            "conn_num": 8,
            "conn_layout": "default",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
            "best_dice": 0.80,
        },
        {
            **_row("drive", "dist_inverted_dg_dsf_ip_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3"),
            "label_mode": "dist_inverted",
            "conn_num": 8,
            "conn_layout": "default",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "ip",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
            "best_dice": 0.99,
        },
    ]
    out_path = Path(tmp_path) / "ablation_s_variation_1_2_non_s.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    # {1,2} must come from non-s proposed row (dice=0.80), not from s13 row (dice=0.99).
    row_12_line = next(line for line in text.splitlines() if r"\textbf{\{1,2\}} & " in line)
    assert "0.8000" in row_12_line
    assert "0.9900" not in row_12_line


def test_ablation_s_variation_branch_field_alone_determines_bucket(tmp_path):
    # Unlike the legacy dg_direct anchor-by-name workaround, dg_dsf rows are
    # bucketed purely by their fusion_branches field; experiment_source naming
    # has no special-cased effect.
    rows = [
        {
            **_row("drive", "dist_inverted_dg_dsf_io_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3"),
            "label_mode": "dist_inverted",
            "conn_num": 8,
            "conn_layout": "default",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
            "best_dice": 0.91,
        },
        {
            **_row("drive", "dist_inverted_dg_dsf_iop_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3"),
            "label_mode": "dist_inverted",
            "conn_num": 8,
            "conn_layout": "default",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "iop",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
            "best_dice": 0.82,
        },
    ]
    out_path = Path(tmp_path) / "ablation_s_variation_branch_field.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    # Presence alone confirms branch_field-only bucketing: the "io" row lands
    # in {1,2} and the "iop" row lands in {1,2,3}, not merged into one bucket.
    next(line for line in text.splitlines() if r"\textbf{\{1,2\}} & " in line)
    next(line for line in text.splitlines() if r"\{1,2,3\} & " in line)


def test_imo_variation_includes_conn24_plain_as_d(tmp_path):
    rows = [
        {
            **_row("drive", "dist_inverted_dg_dsf_io_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3"),
            "label_mode": "dist_inverted",
            "conn_num": 8,
            "conn_layout": "default",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
            "conn_aux_c5_weight": 0.3,
        },
        {
            **_row("drive", "dist_inverted_dg_dsf_im_8_smooth_l1_segaux_w0.5_c3-0.5_c4-0.2"),
            "label_mode": "dist_inverted",
            "conn_num": 8,
            "conn_layout": "default",
            "conn_fusion": "dg_dsf",
            "fusion_branches": "im",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
            "conn_aux_c3_weight": 0.5,
        },
        {
            **_row("drive", "dist_inverted_24_smooth_l1_segaux_w0.5"),
            "label_mode": "dist_inverted",
            "conn_num": 24,
            "conn_layout": "default",
            "conn_fusion": "none",
            "fusion_branches": "none",
            "loss": "smooth_l1",
            "seg_aux_weight": 0.5,
            "seg_aux_variant": "w0.5",
        },
        {
            # gjml_sf_l1 loss should not qualify for D (only smooth_l1 does).
            **_row("drive", "dist_inverted_24_gjml_sf_l1"),
            "label_mode": "dist_inverted",
            "conn_num": 24,
            "conn_layout": "default",
            "conn_fusion": "none",
            "fusion_branches": "none",
            "loss": "gjml_sf_l1",
            "seg_aux_weight": None,
            "seg_aux_variant": "none",
        },
    ]
    out_path = Path(tmp_path) / "ablation_imo_conn24_d.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    assert r"\{A,C\} & " in text
    d_lines = [line for line in text.splitlines() if line.startswith("D & ")]
    assert len(d_lines) == 1
    assert "D=conn24 (no fusion)" in text


def test_default_scope_ablation_excludes_cremi_and_isic(tmp_path):
    rows = [
        _row("drive", "dist_drive"),
        _row("chase", "dist_chase"),
        _row("octa500-3M", "dist_octa"),
        _row("cremi", "dist_cremi"),
        _row("isic", "dist_isic"),
    ]
    for row in rows:
        row["loss"] = "smooth_l1"

    filtered = filter_rows_for_aggregate_scope(rows, include_all=False)
    out_path = Path(tmp_path) / "default_scope_ablation.tex"
    write_ablation_latex(str(out_path), filtered)
    text = out_path.read_text()

    assert r"\multicolumn{6}{c}{drive}" in text
    assert r"\multicolumn{6}{c}{chase}" in text
    assert r"\multicolumn{6}{c}{octa500--3M}" in text
    assert r"\multicolumn{6}{c}{cremi}" not in text
    assert r"\multicolumn{6}{c}{isic}" not in text
    assert r"\clearpage" not in text


def test_cross_experiment_latex_ignores_trash_dataset(tmp_path):
    rows = [
        _row("drive", "drive_exp"),
        _row("trash", "trash_exp"),
    ]

    out_path = Path(tmp_path) / "all_no_trash.tex"
    write_experiment_mean_latex(
        str(out_path),
        "Cross-experiment Mean Summary",
        rows,
    )

    text = out_path.read_text()
    assert "trash" not in text
    assert r"\caption{Cross-experiment mean summary (drive)}" in text


# ---------------------------------------------------------------------------
# Tests for explicit io/im/imo branch display and proposed normalization
# ---------------------------------------------------------------------------

def test_normalize_proposed_suffix_s123():
    assert normalize_proposed_suffix("dist_inverted_dg_direct_8_smooth_l1_segaux_w0.3_s123_proposed") == \
        "dist_inverted_dg_direct_8_smooth_l1_segaux_w0.3_imp"


def test_normalize_proposed_suffix_s1234():
    assert normalize_proposed_suffix("dist_inverted_dg_direct_8_smooth_l1_segaux_w0.5_s1234_proposed") == \
        "dist_inverted_dg_direct_8_smooth_l1_segaux_w0.5_impf"


def test_normalize_proposed_suffix_s14():
    assert normalize_proposed_suffix("dist_inverted_dg_direct_8_smooth_l1_segaux_w0.3_s14_proposed") == \
        "dist_inverted_dg_direct_8_smooth_l1_segaux_w0.3_if"


def test_normalize_proposed_suffix_plain():
    assert normalize_proposed_suffix("exp_proposed") == "exp_imp"


def test_normalize_proposed_suffix_no_change():
    assert normalize_proposed_suffix("dist_inverted_dg_direct_A_8_smooth_l1_segaux_w0.5") == \
        "dist_inverted_dg_direct_A_8_smooth_l1_segaux_w0.5"


def test_proposed_normalized_to_imp_parses_correctly():
    meta = parse_experiment_metadata("dist_inverted_dg_direct_A_8_smooth_l1_segaux_w0.3_s123_proposed")
    assert meta["conn_fusion"] == "dg_direct"
    assert meta["fusion_branches"] == "imp"
    assert meta["seg_aux_weight"] == 0.3
    assert meta["seg_aux_variant"] == "w0.3(imp)"
    assert meta["loss"] == "smooth_l1"


def test_io_suffix_after_segaux_propagates_to_fusion_branches():
    meta = parse_experiment_metadata("dist_inverted_dg_direct_A_8_smooth_l1_segaux_w0.2_io")
    assert meta["conn_fusion"] == "dg_direct"
    assert meta["fusion_branches"] == "io"
    assert meta["seg_aux_weight"] == 0.2
    assert meta["seg_aux_variant"] == "w0.2(io)"


def test_io_c5_annotation_captured_in_segaux_variant():
    meta = parse_experiment_metadata("dist_inverted_dg_direct_A_8_smooth_l1_segaux_w0.3_io_c5-0.3")
    assert meta["conn_fusion"] == "dg_direct"
    assert meta["fusion_branches"] == "io"
    assert meta["seg_aux_weight"] == 0.3
    assert meta["seg_aux_variant"] == "w0.3(io_c5-0.3)"
    assert meta["conn_aux_c5_weight"] == 0.3


def test_c3c5_segaux_captured_from_directory_name():
    # Note: parse_experiment_metadata on raw dir name without profile letter gives conn_fusion="dg"
    # (dg_direct requires profile letter or io suffix in ordered_body to be detected).
    # The important invariant is that c3/c5 weights and seg_aux_weight are correctly extracted.
    meta = parse_experiment_metadata(
        "dist_inverted_dg_direct_8_smooth_l1_segaux_w0.3_c3-0.2_c5-0.2_io"
    )
    assert meta["seg_aux_weight"] == 0.3
    assert meta["conn_aux_c3_weight"] == 0.2
    assert meta["conn_aux_c5_weight"] == 0.2
    assert meta["seg_aux_variant"] == "w0.3(io_c3-0.2_c5-0.2)"


def test_c3c5_segaux_via_config_canonical_name():
    # The canonical name generated by build_experiment_name_from_runtime_config now embeds
    # fusion_branches in the fusion tag and strips it from experiment_name_suffix.
    # New canonical format: dg_direct_io before conn_num, no trailing _io.
    meta = parse_experiment_metadata(
        "dist_inverted_dg_direct_io_8_smooth_l1_segaux_w0.3_c3-0.2_c5-0.2"
    )
    assert meta["conn_fusion"] == "dg_direct"
    assert meta["fusion_branches"] == "io"
    assert meta["seg_aux_weight"] == 0.3
    assert meta["conn_aux_c3_weight"] == 0.2
    assert meta["conn_aux_c5_weight"] == 0.2
    assert meta["seg_aux_variant"] == "w0.3(io_c3-0.2_c5-0.2)"


def test_c3c5_segaux_old_canonical_name_still_parses():
    # Backward-compat: old folder names (pre-rename) still parse correctly even though
    # build_experiment_name_from_runtime_config no longer generates this format.
    meta = parse_experiment_metadata(
        "dist_inverted_dg_direct_io_8_smooth_l1_segaux_w0.3_c3-0.2_c5-0.2_io"
    )
    assert meta["conn_fusion"] == "dg_direct"
    assert meta["fusion_branches"] == "io"
    assert meta["conn_aux_c3_weight"] == 0.2
    assert meta["conn_aux_c5_weight"] == 0.2


def test_c3c5_segaux_variant_no_branches():
    meta = parse_experiment_metadata(
        "dist_inverted_dg_direct_8_smooth_l1_segaux_w0.3_c3-0.5_c5-1.0"
    )
    assert meta["seg_aux_weight"] == 0.3
    assert meta["conn_aux_c3_weight"] == 0.5
    assert meta["conn_aux_c5_weight"] == 1.0
    assert meta["seg_aux_variant"] == "w0.3(c3-0.5_c5-1.0)"


def test_c5_standalone_captured_from_directory_name():
    meta = parse_experiment_metadata("binary_dg_direct_8_bce_c5-0.2_io")
    assert meta["conn_aux_c5_weight"] == 0.2
    assert meta["conn_aux_c3_weight"] is None
    assert meta["seg_aux_weight"] is None


def test_c5_with_segaux_io_format_also_sets_conn_aux_c5():
    # _io_c5-0.3 format: existing segaux regex handles weight, c5-nosegaux sets conn_aux_c5
    meta = parse_experiment_metadata(
        "dist_inverted_dg_direct_A_8_smooth_l1_segaux_w0.3_io_c5-0.3"
    )
    assert meta["conn_aux_c5_weight"] == 0.3


def test_segaux_column_shows_c3c5_annotation():
    result = _format_segaux_column_value(0.3, "w0.3(io_c3-0.2_c5-0.2)", fusion_branches="io")
    assert result == "w0.3/c3-0.2_c5-0.2"


def test_no_c3c5_on_plain_experiment():
    meta = parse_experiment_metadata("dist_inverted_dg_direct_8_smooth_l1_segaux_w0.5_io")
    assert meta["conn_aux_c3_weight"] is None
    assert meta["conn_aux_c5_weight"] is None


def test_fusion_display_shows_io_when_fusion_branches_io():
    assert _format_fusion_display_name("dg_direct", "io") == "dg_direct/io"


def test_fusion_display_shows_im_when_fusion_branches_im():
    assert _format_fusion_display_name("dg_direct", "im") == "dg_direct/im"


def test_fusion_display_hides_none_branch():
    assert _format_fusion_display_name("dg_direct", "none") == "dg_direct"


def test_segaux_column_strips_io_shows_weight_only():
    result = _format_segaux_column_value(0.2, "w0.2(io)", fusion_branches="io")
    assert result == "0.2"


def test_segaux_column_strips_io_prefix_shows_c5_annotation():
    result = _format_segaux_column_value(0.3, "w0.3(io_c5-0.3)", fusion_branches="io")
    assert result == "w0.3/c5-0.3"


def test_segaux_column_strips_imp_shows_weight_only():
    result = _format_segaux_column_value(0.3, "w0.3(imp)", fusion_branches="imp")
    assert result == "0.3"


def test_io_in_fusion_tag_from_config():
    config = {
        "label_mode": "dist_inverted",
        "conn_num": 8,
        "conn_fusion": "dg_direct",
        "fusion_branches": "io",
        "fusion_loss_profile": "A",
        "dist_aux_loss": "smooth_l1",
        "use_seg_aux": True,
        "seg_aux_weight": 0.5,
    }
    name = build_experiment_name_from_runtime_config(config)
    assert name is not None
    assert "_io_" in name or name.endswith("_io") or "_io_8_" in name


def test_io_in_fusion_tag_no_double_suffix():
    """When experiment_name_suffix == fusion_branches, it should not be added twice."""
    config = {
        "label_mode": "dist_inverted",
        "conn_num": 8,
        "conn_fusion": "dg_direct",
        "fusion_branches": "io",
        "fusion_loss_profile": "A",
        "dist_aux_loss": "smooth_l1",
        "use_seg_aux": True,
        "seg_aux_weight": 0.5,
        "experiment_name_suffix": "io",
    }
    name = build_experiment_name_from_runtime_config(config)
    assert name is not None
    # Should appear exactly once, not twice
    assert name.count("_io") == 1
