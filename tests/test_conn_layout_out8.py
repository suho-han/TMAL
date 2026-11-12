import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

import train
from connect_loss import (
    Bilateral_voting_kxk,
    connect_loss,
    distance_affinity_matrix,
    resolve_connectivity_layout,
    shift_n_directions,
)
from scripts import train_launcher_from_config as launcher
from scripts.rebuild_dist_signed_artifacts import pair_alignment_scores


def _patch_cuda_to_noop(monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self, raising=False)


def _build_shift_matrix(size):
    matrix = torch.zeros((1, 1, size, size), dtype=torch.float32)
    for i in range(size - 1):
        matrix[0, 0, i, i + 1] = 1.0
    return matrix


def _reference_shift(x, dy, dx):
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


def _make_args(label_mode='binary'):
    return SimpleNamespace(
        num_class=1,
        dataset='chase',
        label_mode=label_mode,
        dist_aux_loss='smooth_l1',
        dist_sf_l1_gamma=1.0,
        conn_layout='out8',
    )


def test_resolve_connectivity_layout_out8_matches_expected_offsets():
    layout = resolve_connectivity_layout(8, 'out8')
    assert layout['name'] == 'out8'
    assert layout['channel_count'] == 8
    assert layout['kernel_size'] == 5
    assert layout['offsets'] == [
        (-2, -2),
        (-2, 0),
        (-2, 2),
        (0, -2),
        (0, 2),
        (2, -2),
        (2, 0),
        (2, 2),
    ]


def test_shift_n_directions_out8_matches_expected_offsets():
    x = torch.arange(1, 1 + 7 * 7, dtype=torch.float32).view(1, 1, 7, 7)
    layout = resolve_connectivity_layout(8, 'out8')
    shifted = shift_n_directions(x, 8, conn_layout='out8')
    assert len(shifted) == 8
    for tensor, (dy, dx) in zip(shifted, layout['offsets']):
        expected = _reference_shift(x, dy, dx)
        assert torch.equal(tensor, expected)


def test_distance_affinity_matrix_out8_shape():
    dist_map = torch.rand((2, 1, 6, 7), dtype=torch.float32)
    affinity = distance_affinity_matrix(dist_map, conn_num=8, sigma=2.0, conn_layout='out8')
    assert affinity.shape == (2, 8, 6, 7)


def test_bilateral_voting_kxk_out8_uses_reverse_offsets(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)

    batch = 2
    height = 7
    width = 7
    layout = resolve_connectivity_layout(8, 'out8')
    hori = _build_shift_matrix(width).repeat(batch, 1, 1, 1)
    vert = _build_shift_matrix(height).repeat(batch, 1, 1, 1)
    affinity = torch.rand((batch, 1, 8, height, width), dtype=torch.float32)

    _, vote = Bilateral_voting_kxk(
        affinity,
        hori,
        vert,
        conn_num=layout['kernel_size'],
        offsets=layout['offsets'],
    )

    offset_to_idx = {offset: idx for idx, offset in enumerate(layout['offsets'])}
    for idx, (dy, dx) in enumerate(layout['offsets']):
        rev_idx = offset_to_idx[(-dy, -dx)]
        expected = affinity[:, :, idx] * _reference_shift(affinity[:, :, rev_idx], dy, dx)
        assert torch.allclose(vote[:, :, idx], expected, atol=1e-6, rtol=1e-5)


def test_single_class_forward_supports_out8_binary_and_dist(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)

    size = 7
    hori = _build_shift_matrix(size)
    vert = _build_shift_matrix(size)
    affinity_map = torch.zeros((1, 8, size, size), dtype=torch.float32)

    binary_target = torch.zeros((1, 1, size, size), dtype=torch.float32)
    binary_target[:, :, 2:5, 2:5] = 1.0
    binary_module = connect_loss(
        _make_args(label_mode='binary'),
        hori,
        vert,
        label_mode='binary',
        conn_num=8,
        sigma=2.0,
        conn_layout='out8',
    )
    binary_loss, binary_terms = binary_module(affinity_map, binary_target, return_details=True)
    assert torch.isfinite(binary_loss)
    assert torch.isfinite(binary_terms['edge'])

    dist_target = torch.zeros((1, 1, size, size), dtype=torch.float32)
    dist_target[:, :, 2:5, 2:5] = 0.8
    dist_module = connect_loss(
        _make_args(label_mode='dist'),
        hori,
        vert,
        label_mode='dist',
        conn_num=8,
        sigma=2.0,
        conn_layout='out8',
    )
    dist_loss, dist_terms = dist_module(affinity_map, dist_target, return_details=True)
    assert torch.isfinite(dist_loss)
    assert torch.isfinite(dist_terms['affinity'])


def test_train_parse_args_accepts_out8(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--conn_layout', 'out8',
            '--output_dir', str(tmp_path),
        ],
    )
    args = train.parse_args()
    assert args.conn_num == 8
    assert args.conn_layout == 'out8'
    assert args.connectivity_layout['kernel_size'] == 5


def test_train_parse_args_appends_experiment_name_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--output_dir', str(tmp_path),
            '--experiment_name_suffix', 'connaux_off',
        ],
    )
    args = train.parse_args()
    assert args.experiment_name_suffix == 'connaux_off'
    assert args.save.endswith('binary_8_bce_connaux_off')


def test_train_parse_args_accepts_max_batch_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--output_dir', str(tmp_path),
            '--max_train_batches', '1',
            '--max_eval_batches', '2',
        ],
    )
    args = train.parse_args()
    assert args.max_train_batches == 1
    assert args.max_eval_batches == 2


def test_train_parse_args_rejects_negative_max_batch_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--output_dir', str(tmp_path),
            '--max_train_batches', '-1',
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_train_parse_args_rejects_invalid_conn_layout_combo(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '24',
            '--conn_layout', 'out8',
            '--output_dir', str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_train_parse_args_rejects_multiclass_out8(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'retouch-Spectrailis',
            '--num-class', '4',
            '--conn_num', '8',
            '--conn_layout', 'out8',
            '--output_dir', str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_launcher_build_single_schedule_supports_out8():
    config = {
        'mode': 'single',
        'dataset': 'chase',
        'device': 0,
        'single': {
            'conn_num': 8,
            'conn_layout': 'out8',
            'label_mode': 'binary',
            'output_dir': 'output',
        },
    }
    runs = launcher.build_single_schedule(config, device=0)
    assert len(runs) == 1
    assert runs[0]['conn_layout'] == 'out8'

    cmd = launcher.build_train_cmd(
        pybin='python',
        repo_root=launcher.Path('.'),
        preset=runs[0]['preset'],
        conn_num=runs[0]['conn_num'],
        conn_layout=runs[0]['conn_layout'],
        label_mode=runs[0]['label_mode'],
        dist_aux_loss=runs[0]['dist_aux_loss'],
        dist_sf_l1_gamma=1.0,
        device=runs[0]['device'],
        epochs=runs[0]['epochs'],
        folds=runs[0]['folds'],
        target_fold=runs[0]['target_fold'],
    )
    assert '--conn_layout' in cmd


def test_launcher_module_imports_when_sys_argv_points_to_stdin(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['-'], raising=False)

    module = importlib.reload(launcher)

    assert hasattr(module, 'build_multi_schedule')
    assert hasattr(module, 'build_single_schedule')


def test_launcher_rejects_invalid_out8_with_conn24():
    config = {
        'mode': 'single',
        'dataset': 'chase',
        'device': 0,
        'single': {
            'conn_num': 24,
            'conn_layout': 'out8',
            'label_mode': 'binary',
        },
    }
    with pytest.raises(ValueError):
        launcher.build_single_schedule(config, device=0)


def test_launcher_filters_completed_run_and_appends_retry_log(tmp_path, capsys):
    repo_root = tmp_path
    output_root = tmp_path / 'custom_output'
    config_path = repo_root / 'scripts' / 'configs' / 'test.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('mode: single\n', encoding='utf-8')

    config = {
        'mode': 'single',
        'dataset': 'chase',
        'device': 0,
        'single': {
            'conn_num': 8,
            'label_mode': 'binary',
            'output_dir': str(output_root),
        },
    }
    runs = launcher.build_single_schedule(config, device=0)
    run = runs[0]
    exp_dir = launcher.get_run_output_dir(run)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'final_results_1.csv').write_text('dice\n0.9\n', encoding='utf-8')

    kept_runs, skipped_runs, log_path = launcher.filter_completed_runs(runs, repo_root, config_path)

    assert kept_runs == []
    assert len(skipped_runs) == 1
    assert skipped_runs[0]['reason'].endswith('final_results_1.csv exists')
    assert log_path == repo_root / 'output' / 'retried_exp.md'
    log_text = log_path.read_text(encoding='utf-8')
    assert 'dataset=`chase`' in log_text
    assert 'experiment=`binary_8_bce`' in log_text
    assert f'output_dir=`{output_root}`' in log_text
    assert 'target_fold=`-`' in log_text
    assert 'final_results_1.csv' in log_text

    captured = capsys.readouterr()
    assert 'Completed experiment detected; skipping retried run' in captured.err


def test_launcher_completion_status_accepts_expected_epoch_model(tmp_path):
    config = {
        'mode': 'single',
        'dataset': 'chase',
        'device': 0,
        'single': {
            'conn_num': 8,
            'label_mode': 'binary',
            'epochs': 7,
            'output_dir': str(tmp_path / 'output'),
        },
    }
    run = launcher.build_single_schedule(config, device=0)[0]
    models_dir = launcher.get_run_output_dir(run) / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / '7_model.pth').write_text('', encoding='utf-8')

    completed, reason = launcher.get_run_completion_status(run)

    assert completed is True
    assert reason is not None
    assert reason.endswith('models/7_model.pth exists')


def test_launcher_completion_status_accepts_single_fold_final_results_csv(tmp_path):
    config = {
        'mode': 'single',
        'dataset': 'chase',
        'device': 0,
        'single': {
            'conn_num': 8,
            'label_mode': 'binary',
            'output_dir': str(tmp_path / 'output'),
        },
    }
    run = launcher.build_single_schedule(config, device=0)[0]
    exp_dir = launcher.get_run_output_dir(run)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'final_results.csv').write_text('dice\n0.9\n', encoding='utf-8')

    completed, reason = launcher.get_run_completion_status(run)

    assert completed is True
    assert reason is not None
    assert reason.endswith('final_results.csv exists')


def test_launcher_warns_skip_completed_keys_are_deprecated(capsys):
    launcher.validate_and_warn_skip_completed_config(
        {
            'mode': 'multi',
            'skip_completed': True,
            'multi': {'skip_completed': False},
        },
        mode='multi',
    )

    captured = capsys.readouterr()
    assert 'Deprecated config key(s) skip_completed, multi.skip_completed are ignored' in captured.err


def test_normalize_smoke_runs_keeps_all_runs_by_default():
    runs = [
        {'epochs': 10, 'target_fold': None, 'output_dir': 'output', 'id': 1},
        {'epochs': 20, 'target_fold': None, 'output_dir': 'output', 'id': 2},
    ]

    smoke_runs = launcher.normalize_smoke_runs(runs, smoke_limit=None)

    assert len(smoke_runs) == 2
    assert [run['id'] for run in smoke_runs] == [1, 2]
    assert all(run['epochs'] == 5 for run in smoke_runs)
    assert all(run['target_fold'] == 1 for run in smoke_runs)
    assert all(run['output_dir'] == 'output_smoke' for run in smoke_runs)
    assert all(run['max_train_batches'] == 1 for run in smoke_runs)
    assert all(run['max_eval_batches'] == 1 for run in smoke_runs)


def test_normalize_smoke_runs_preserves_custom_output_dir_namespaces():
    runs = [
        {'epochs': 10, 'target_fold': None, 'output_dir': 'output_conn_aux_ablation', 'id': 1},
        {'epochs': 20, 'target_fold': None, 'output_dir': 'output_conn_aux_ablation_strong', 'id': 2},
    ]

    smoke_runs = launcher.normalize_smoke_runs(runs, smoke_limit=None)

    assert [run['output_dir'] for run in smoke_runs] == [
        'output_smoke/output_conn_aux_ablation',
        'output_smoke/output_conn_aux_ablation_strong',
    ]


def test_launcher_experiment_name_suffix_changes_output_dir_name():
    config = {
        'mode': 'multi',
        'dataset': ['drive'],
        'device': 0,
        'output_dir': 'output_mnt',
        'multi': {
            'epochs': 5,
            'folds': 1,
            'experiments_only': True,
            'experiments': [
                {
                    'datasets': ['drive'],
                    'conn_num': 8,
                    'label_mode': 'dist_inverted',
                    'conn_fusion': 'dg',
                    'dist_aux_loss': 'smooth_l1',
                    'use_seg_aux': True,
                    'seg_aux_weight': 0.5,
                    'experiment_name_suffix': 'connaux_off',
                }
            ],
        },
    }

    run = launcher.build_multi_schedule(config, device=0, datasets=['drive'])[0]
    assert run['output_dir'] == 'output_mnt'
    assert run['experiment_name_suffix'] == 'connaux_off'
    assert launcher.get_run_experiment_name(run) == 'dist_inverted_dg_io_8_smooth_l1_segaux_w0.5_connaux_off'


def test_normalize_smoke_runs_respects_explicit_limit():
    runs = [
        {'epochs': 10, 'target_fold': None, 'output_dir': 'output', 'id': 1},
        {'epochs': 20, 'target_fold': None, 'output_dir': 'output', 'id': 2},
    ]

    smoke_runs = launcher.normalize_smoke_runs(runs, smoke_limit=1)

    assert len(smoke_runs) == 1
    assert smoke_runs[0]['id'] == 1


def test_build_train_cmd_forwards_optional_max_batch_limits():
    preset = {
        'dataset': 'drive',
        'data_root': 'data/DRIVE',
        'resize': (512, 512),
        'batch_size': 2,
    }
    cmd = launcher.build_train_cmd(
        pybin='python',
        repo_root=launcher.Path('.'),
        preset=preset,
        conn_num=8,
        conn_layout='standard8',
        label_mode='binary',
        dist_aux_loss='smooth_l1',
        dist_sf_l1_gamma=1.0,
        device=0,
        epochs=3,
        folds=1,
        target_fold=1,
        max_train_batches=1,
        max_eval_batches=1,
    )
    assert '--max_train_batches' in cmd
    assert '--max_eval_batches' in cmd
    assert cmd[cmd.index('--max_train_batches') + 1] == '1'
    assert cmd[cmd.index('--max_eval_batches') + 1] == '1'


def test_build_train_cmd_does_not_emit_max_batch_flags_by_default():
    preset = {
        'dataset': 'drive',
        'data_root': 'data/DRIVE',
        'resize': (512, 512),
        'batch_size': 2,
    }
    cmd = launcher.build_train_cmd(
        pybin='python',
        repo_root=launcher.Path('.'),
        preset=preset,
        conn_num=8,
        conn_layout='standard8',
        label_mode='binary',
        dist_aux_loss='smooth_l1',
        dist_sf_l1_gamma=1.0,
        device=0,
        epochs=3,
        folds=1,
        target_fold=1,
    )
    assert '--max_train_batches' not in cmd
    assert '--max_eval_batches' not in cmd


def test_pair_alignment_scores_rejects_non_four_pair_layout():
    class_pred_bin = torch.zeros((1, 1, 3, 5, 5), dtype=torch.float32)
    offsets = [(1, 0), (-1, 0), (0, 1)]
    with pytest.raises(ValueError, match='requires every offset to have a reverse pair|expects exactly 4 reverse-offset pairs'):
        pair_alignment_scores(class_pred_bin, offsets)
