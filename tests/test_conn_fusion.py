from solver import Solver
from scripts import train_launcher_from_config as launcher
from model.DconnNet import OUTER_8_NATIVE_ORDER, OUTER_8_STANDARD_ORDER, OUTER_8_TO_STANDARD8_INDEX, DconnNet, fuse_directional_logits, reorder_outer8_to_standard8
import model.DconnNet as dconn_module
import train
import torch.nn as nn
import torch
import pytest
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )


def _set_identity_1x1(conv: nn.Conv2d):
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        channels = min(conv.in_channels, conv.out_channels)
        for c_idx in range(channels):
            conv.weight[c_idx, c_idx, 0, 0] = 1.0


class _SequenceLoss:
    def __init__(self, terms_sequence, label_mode='binary'):
        self._terms_sequence = list(terms_sequence)
        self.label_mode = label_mode

    def __call__(self, logits, target, return_details=False):
        assert return_details
        assert self._terms_sequence, "No more mocked loss terms available."
        terms = self._terms_sequence.pop(0)
        return terms.get('total', torch.tensor(0.0)), terms


def test_outer8_reorder_mapping_is_fixed():
    assert OUTER_8_NATIVE_ORDER == [
        (-2, -2),
        (-2, 0),
        (-2, 2),
        (0, -2),
        (0, 2),
        (2, -2),
        (2, 0),
        (2, 2),
    ]
    assert OUTER_8_STANDARD_ORDER == [
        (2, 2),
        (2, 0),
        (2, -2),
        (0, 2),
        (0, -2),
        (-2, 2),
        (-2, 0),
        (-2, -2),
    ]
    assert OUTER_8_TO_STANDARD8_INDEX == [7, 6, 5, 4, 3, 2, 1, 0]

    x = torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1)
    y = reorder_outer8_to_standard8(x)
    assert torch.equal(y.view(-1), torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.float32))


def test_outer8_reorder_supports_multi_group_channels():
    x = torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1)
    y = reorder_outer8_to_standard8(x)
    expected = torch.tensor(
        [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8],
        dtype=torch.float32,
    )
    assert torch.equal(y.view(-1), expected)


def test_gate_fusion_operates_on_logits():
    c3 = torch.ones((1, 8, 2, 2), dtype=torch.float32)
    c5 = torch.zeros((1, 8, 2, 2), dtype=torch.float32)
    gate_conv = nn.Conv2d(16, 8, kernel_size=1, bias=True)
    with torch.no_grad():
        gate_conv.weight.zero_()
        gate_conv.bias.zero_()

    fused = fuse_directional_logits(c3, c5, mode='gate', gate_conv=gate_conv)
    assert torch.allclose(fused, torch.full_like(fused, 0.5), atol=1e-6)


def test_scaled_sum_fusion_respects_residual_scale():
    c3 = torch.ones((1, 8, 2, 2), dtype=torch.float32)
    c5 = torch.full((1, 8, 2, 2), 2.0, dtype=torch.float32)
    fused = fuse_directional_logits(c3, c5, mode='scaled_sum', residual_scale=0.2)
    assert torch.allclose(fused, torch.full_like(fused, 1.4), atol=1e-6)


def test_conv_residual_fusion_uses_residual_projection():
    c3 = torch.ones((1, 8, 2, 2), dtype=torch.float32)
    c5 = torch.full((1, 8, 2, 2), 2.0, dtype=torch.float32)
    residual_conv = nn.Conv2d(8, 8, kernel_size=1, bias=True)
    _set_identity_1x1(residual_conv)
    fused = fuse_directional_logits(c3, c5, mode='conv_residual', residual_conv=residual_conv)
    assert torch.allclose(fused, torch.full_like(fused, 3.0), atol=1e-6)


def test_dgrf_additive_profile_loss_includes_branch_aux_and_gate():
    fused_terms = {
        'vote': torch.tensor(1.0),
        'dice': torch.tensor(2.0),
        'affinity': torch.tensor(3.0),
        'total': torch.tensor(30.0),
    }
    inner_terms = {
        'affinity': torch.tensor(5.0),
        'total': torch.tensor(20.0),
    }
    outer_terms = {
        'affinity': torch.tensor(7.0),
        'total': torch.tensor(30.0),
    }

    solver = object.__new__(Solver)
    solver.fusion_enabled = True
    solver.fusion_lambda_fused = 0.3
    solver.args = SimpleNamespace(
        conn_fusion='dg',
        conn_aux_c3_weight=0.3,
        conn_aux_c5_weight=0.2,
        fusion_gate_reg_weight=0.01,
        use_seg_aux=False,
    )
    solver.loss_func = _SequenceLoss([fused_terms, inner_terms])
    solver.loss_func_outer = _SequenceLoss([outer_terms])

    output_dict = {
        'fused': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'inner': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'outer': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'fusion_gate': torch.full((1, 8, 4, 4), 0.5, dtype=torch.float32),
    }
    target = torch.zeros((1, 8, 4, 4), dtype=torch.float32)

    total, terms = solver._compute_fusion_profile_loss(
        output_dict,
        target,
        binary_gt=None,
        collect_edge_stats=False,
    )

    expected = torch.tensor(3.9 + 0.3 * 20.0 + 0.2 * 30.0 + 0.01 * 0.5, dtype=torch.float32)
    assert torch.isclose(total, expected, atol=1e-6)
    assert torch.isclose(terms['dgrf_fused_main'], torch.tensor(3.9), atol=1e-6)
    assert torch.isclose(terms['dgrf_c3_aux'], torch.tensor(20.0), atol=1e-6)
    assert torch.isclose(terms['dgrf_c5_aux'], torch.tensor(30.0), atol=1e-6)
    assert torch.isclose(terms['gate'], torch.tensor(0.5), atol=1e-6)
    assert torch.isclose(terms['total'], expected, atol=1e-6)


def test_dgrf_terms_are_not_added_for_non_dgrf_fusion():
    fused_terms = {
        'vote': torch.tensor(1.0),
        'dice': torch.tensor(2.0),
        'affinity': torch.tensor(3.0),
    }
    inner_terms = {'affinity': torch.tensor(5.0)}
    outer_terms = {'affinity': torch.tensor(7.0)}

    solver = object.__new__(Solver)
    solver.fusion_enabled = True
    solver.fusion_lambda_fused = 0.3
    solver.args = SimpleNamespace(
        conn_fusion='gate',
        conn_aux_c3_weight=0.3,
        conn_aux_c5_weight=0.2,
        fusion_gate_reg_weight=0.01,
        use_seg_aux=False,
    )
    solver.loss_func = _SequenceLoss([fused_terms, inner_terms])
    solver.loss_func_outer = _SequenceLoss([outer_terms])

    output_dict = {
        'fused': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'inner': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'outer': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
    }
    target = torch.zeros((1, 8, 4, 4), dtype=torch.float32)

    total, terms = solver._compute_fusion_profile_loss(
        output_dict,
        target,
        binary_gt=None,
        collect_edge_stats=False,
    )

    assert torch.isclose(total, torch.tensor(3.9), atol=1e-6)
    assert 'dgrf_fused_main' not in terms
    assert 'dgrf_c3_aux' not in terms
    assert 'dgrf_c5_aux' not in terms
    assert 'gate' not in terms
    assert torch.isclose(terms['total'], total, atol=1e-6)


def test_train_parse_args_validates_fusion_constraints(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--conn_layout', 'standard8',
            '--conn_fusion', 'gate',
            '--output_dir', str(tmp_path),
        ],
    )
    args = train.parse_args()
    assert args.conn_fusion == 'gate'

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '24',
            '--conn_fusion', 'gate',
            '--output_dir', str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--label_mode', 'dist_signed',
            '--output_dir', str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()

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
            '--conn_fusion', 'gate',
            '--output_dir', str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'retouch-Spectrailis',
            '--num-class', '4',
            '--conn_num', '8',
            '--conn_fusion', 'gate',
            '--output_dir', str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_experiment_name_is_legacy_stable_and_fusion_specific(tmp_path, monkeypatch):
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
        ],
    )
    legacy_args = train.parse_args()
    assert train.get_experiment_output_name(legacy_args) == 'binary_8_bce'

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--conn_fusion', 'gate',
            '--output_dir', str(tmp_path),
        ],
    )
    fusion_args = train.parse_args()
    assert train.get_experiment_output_name(fusion_args) == 'binary_gate_io_8_bce'

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--conn_fusion', 'decoder_guided',
            '--output_dir', str(tmp_path),
        ],
    )
    dg_args = train.parse_args()
    assert dg_args.conn_fusion == 'dg'
    assert train.get_experiment_output_name(dg_args) == 'binary_dg_io_8_bce'


def test_segaux_requires_explicit_weight_and_keeps_suffix_unweighted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--use_seg_aux',
            '--output_dir', str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        train.parse_args()

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train.py',
            '--dataset', 'chase',
            '--data_root', 'data/chase',
            '--num-class', '1',
            '--conn_num', '8',
            '--conn_fusion', 'dg_direct',
            '--output_dir', str(tmp_path),
        ],
    )
    args = train.parse_args()
    assert args.conn_fusion == 'dg_direct'
    assert args.seg_aux_weight is None

    assert launcher.build_experiment_output_name(
        conn_num=8,
        label_mode='binary',
        dist_aux_loss='smooth_l1',
        use_seg_aux=True,
        seg_aux_weight=None,
    ) == 'binary_8_bce_segaux'
    assert launcher.build_experiment_output_name(
        conn_num=8,
        label_mode='binary',
        dist_aux_loss='smooth_l1',
        use_seg_aux=True,
        seg_aux_weight=0.5,
    ) == 'binary_8_bce_segaux_w0.5'
    assert launcher.build_experiment_output_name(
        conn_num=8,
        label_mode='binary',
        dist_aux_loss='smooth_l1',
        conn_fusion='decoder_guided',
    ) == 'binary_dg_io_8_bce'
    assert launcher.build_experiment_output_name(
        conn_num=8,
        label_mode='binary',
        dist_aux_loss='smooth_l1',
        conn_fusion='dg_direct',
        use_seg_aux=True,
        seg_aux_weight=0.5,
    ) == 'binary_dg_direct_io_8_bce_segaux_w0.5'


def test_launcher_supports_fusion_names_and_scaled_sum_sweep():
    assert launcher.build_experiment_output_name(
        conn_num=8,
        label_mode='binary',
        dist_aux_loss='smooth_l1',
    ) == 'binary_8_bce'
    assert launcher.build_experiment_output_name(
        conn_num=8,
        label_mode='dist',
        dist_aux_loss='smooth_l1',
        conn_fusion='scaled_sum',
        fusion_residual_scale=0.3,
    ) == 'dist_scaled_sum_rs0.3_io_8_smooth_l1'

    config = {
        'mode': 'multi',
        'dataset': ['drive'],
        'device': 0,
        'multi': {
            'conn_nums': [8],
            'conn_layouts': ['standard8'],
            'conn_fusions': ['scaled_sum'],
            'fusion_residual_scales': [0.1, 0.2, 0.3],
            'label_modes': ['binary'],
            'dist_aux_losses': ['smooth_l1'],
            'epochs': 5,
            'folds': 1,
        },
    }
    runs = launcher.build_multi_schedule(config, device=0, datasets=['drive'])
    assert {round(float(run['fusion_residual_scale']), 3) for run in runs} == {0.1, 0.2, 0.3}
    assert all(run['conn_fusion'] == 'scaled_sum' for run in runs)


def test_launcher_rejects_unsupported_label_mode():
    config = {
        'mode': 'multi',
        'dataset': ['drive'],
        'device': 0,
        'multi': {
            'conn_nums': [8],
            'conn_layouts': ['standard8'],
            'label_modes': ['dist_signed'],
            'dist_aux_losses': ['smooth_l1'],
            'epochs': 5,
            'folds': 1,
        },
    }
    with pytest.raises(ValueError):
        launcher.build_multi_schedule(config, device=0, datasets=['drive'])


def test_launcher_rejects_removed_decoder_fusion_keys():
    config = {
        'mode': 'multi',
        'dataset': ['drive'],
        'device': 0,
        'multi': {
            'conn_nums': [8],
            'conn_layouts': ['standard8'],
            'conn_fusions': ['gate'],
            'decoder_fusions': ['residual_gate'],
            'label_modes': ['binary'],
            'dist_aux_losses': ['smooth_l1'],
            'epochs': 5,
            'folds': 1,
        },
    }
    with pytest.raises(ValueError, match='removed key'):
        launcher.build_multi_schedule(config, device=0, datasets=['drive'])


@pytest.mark.parametrize(
    'removed_key',
    [
        'fusion_loss_profile',
        'fusion_loss_profiles',
        'fusion_lambda_inner',
        'fusion_lambda_inners',
        'fusion_lambda_outer',
        'fusion_lambda_outers',
    ],
)
def test_launcher_rejects_removed_fusion_loss_profile_keys(removed_key):
    config = {
        'mode': 'multi',
        'dataset': ['drive'],
        'device': 0,
        'multi': {
            'conn_nums': [8],
            'conn_layouts': ['standard8'],
            'conn_fusions': ['gate'],
            removed_key: ['A'] if removed_key.endswith('s') else 'A',
            'label_modes': ['binary'],
            'dist_aux_losses': ['smooth_l1'],
            'epochs': 5,
            'folds': 1,
        },
    }
    with pytest.raises(ValueError, match='removed key'):
        launcher.build_multi_schedule(config, device=0, datasets=['drive'])


def test_launcher_supports_fusion_matrix_without_unintended_cartesian_product():
    config = {
        'mode': 'multi',
        'dataset': ['drive'],
        'device': 0,
        'multi': {
            'conn_nums': [8],
            'conn_layouts': ['standard8'],
            'label_modes': ['binary'],
            'dist_aux_losses': ['smooth_l1'],
            'epochs': 5,
            'folds': 1,
            'fusion_matrix': [
                {'conn_fusion': 'gate'},
                {'conn_fusion': 'conv_residual'},
                {
                    'conn_fusion': 'scaled_sum',
                    'fusion_residual_scales': [0.1, 0.2, 0.3, 0.5],
                },
            ],
        },
    }
    runs = launcher.build_multi_schedule(config, device=0, datasets=['drive'])
    combos = {
        (
            str(run['conn_fusion']),
            round(float(run['fusion_residual_scale']), 3),
        )
        for run in runs
    }
    expected = {
        ('gate', 0.2),
        ('conv_residual', 0.2),
        ('scaled_sum', 0.1),
        ('scaled_sum', 0.2),
        ('scaled_sum', 0.3),
        ('scaled_sum', 0.5),
    }
    assert combos == expected


def test_launcher_multi_experiments_support_experiment_specific_output_dir():
    config = {
        'mode': 'multi',
        'dataset': ['drive', 'chase'],
        'device': 0,
        'output_dir': 'output_default',
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
                    'output_dir': 'output_drive_off',
                },
                {
                    'datasets': ['chase'],
                    'conn_num': 8,
                    'label_mode': 'dist_inverted',
                    'conn_fusion': 'dg',
                    'dist_aux_loss': 'smooth_l1',
                    'use_seg_aux': True,
                    'seg_aux_weight': 0.5,
                },
            ],
        },
    }

    runs = launcher.build_multi_schedule(config, device=0, datasets=['drive', 'chase'])
    assert len(runs) == 2
    assert runs[0]['output_dir'] == 'output_drive_off'
    assert runs[1]['output_dir'] == 'output_default'


def test_launcher_rejects_multiclass_fusion():
    config = {
        'mode': 'single',
        'dataset': 'retouch-Spectrailis',
        'device': 0,
        'single': {
            'conn_num': 8,
            'conn_layout': 'standard8',
            'conn_fusion': 'gate',
            'label_mode': 'binary',
            'folds': 3,
            'target_folds': [1],
        },
    }
    with pytest.raises(ValueError):
        launcher.build_single_schedule(config, device=0)


def test_model_output_contract_legacy_vs_fusion(monkeypatch):
    monkeypatch.setattr(dconn_module, 'resnet34', lambda pretrained=True: _DummyBackbone())
    x = torch.randn((1, 3, 64, 64), dtype=torch.float32)

    legacy_model = DconnNet(
        num_class=1,
        conn_num=8,
        conn_layout='standard8',
        conn_fusion='none',
    )
    fusion_model = DconnNet(
        num_class=1,
        conn_num=8,
        conn_layout='standard8',
        conn_fusion='gate',
    )

    legacy_model.eval()
    fusion_model.eval()
    with torch.no_grad():
        legacy_out = legacy_model(x)
        fusion_out = fusion_model(x)

    assert isinstance(legacy_out, dict)
    assert {'fused', 'aux'} <= set(legacy_out.keys())
    assert torch.is_tensor(legacy_out['fused'])
    assert torch.is_tensor(legacy_out['aux'])

    assert isinstance(fusion_out, dict)
    assert {'fused', 'inner', 'outer', 'outer_aligned', 'aux'} <= set(fusion_out.keys())
    assert torch.is_tensor(fusion_out['fused'])
    assert 'seg' not in fusion_out

    legacy_state = legacy_model.state_dict()
    assert not any(key.startswith('cls_pred_inner_conv.') for key in legacy_state)
    assert not any(key.startswith('cls_pred_outer_conv.') for key in legacy_state)
    assert not any(key.startswith('fusion_gate_conv.') for key in legacy_state)

    legacy_model_2 = DconnNet(
        num_class=1,
        conn_num=8,
        conn_layout='standard8',
        conn_fusion='none',
    )
    legacy_model_2.load_state_dict(legacy_state)


def test_dg_direct_uses_decoder_only_segaux_head(monkeypatch):
    monkeypatch.setattr(dconn_module, 'resnet34', lambda pretrained=True: _DummyBackbone())
    x = torch.randn((1, 3, 64, 64), dtype=torch.float32)

    dg_direct_model = DconnNet(
        num_class=1,
        conn_num=8,
        conn_layout='standard8',
        conn_fusion='dg_direct',
        use_seg_aux=True,
    )
    dg_direct_model.eval()
    with torch.no_grad():
        output = dg_direct_model(x)

    assert {'fused', 'inner', 'outer', 'outer_aligned', 'aux', 'fusion_gate', 'mask_logit'} <= set(output.keys())
    assert output['mask_logit'].shape[1] == 1


# ---------------------------------------------------------------------------
# H5 regression and `legacy_aux_weight` invariants (fork-specific).
# ---------------------------------------------------------------------------


def test_loss_func_outer_uses_out8_layout_for_fusion():
    """Regression: when fusion is enabled, the outer-branch loss target must
    be built from the `out8` connectivity layout, not `standard8`. If this
    check ever fails, outer affinity supervision is being computed against the
    wrong target meaning (H5 bug)."""
    from src.losses.factory import build_loss_functions

    args = SimpleNamespace(
        sigma=2.0,
        tau=3.0,
        dist_aux_loss='smooth_l1',
        dist_sf_l1_gamma=1.0,
        num_class=1,
        conn_num=8,
        conn_layout='standard8',
    )
    hori = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    verti = torch.zeros((1, 1, 4, 4), dtype=torch.float32)

    loss_func, _, loss_func_outer, _, _ = build_loss_functions(
        args=args,
        hori_translation=hori,
        verti_translation=verti,
        label_mode='binary',
        fusion_enabled=True,
    )
    assert loss_func.conn_layout == 'standard8'
    assert loss_func_outer is not None
    assert loss_func_outer.conn_layout == 'out8'


def test_legacy_aux_weight_default_zero_keeps_fusion_total_unchanged():
    """When `legacy_aux_weight=0.0` (default), `_compute_fusion_profile_loss`
    must return the same total as before this option existed (fusion path
    baseline regression)."""
    fused_terms = {
        'vote': torch.tensor(1.0),
        'dice': torch.tensor(2.0),
        'affinity': torch.tensor(3.0),
        'total': torch.tensor(0.0),
    }
    inner_terms = {'affinity': torch.tensor(5.0), 'total': torch.tensor(20.0)}
    outer_terms = {'affinity': torch.tensor(7.0), 'total': torch.tensor(30.0)}

    solver = object.__new__(Solver)
    solver.fusion_enabled = True
    solver.fusion_lambda_fused = 0.3
    solver.args = SimpleNamespace(
        conn_fusion='gate',
        conn_aux_c3_weight=0.3,
        conn_aux_c5_weight=0.2,
        fusion_gate_reg_weight=0.01,
        use_seg_aux=False,
        legacy_aux_weight=0.0,
        debug_loss_breakdown=False,
    )
    solver.loss_func = _SequenceLoss([fused_terms, inner_terms])
    solver.loss_func_outer = _SequenceLoss([outer_terms])

    output_dict = {
        'fused': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'inner': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'outer': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'aux': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
    }
    target = torch.zeros((1, 8, 4, 4), dtype=torch.float32)

    total, terms = solver._compute_fusion_profile_loss(
        output_dict,
        target,
        binary_gt=None,
        collect_edge_stats=False,
    )

    # Same as the existing profile-A regression: vote+dice+0.3*aff = 3.9
    assert torch.isclose(total, torch.tensor(3.9), atol=1e-6)
    assert 'legacy_aux' not in terms
    assert 'legacy_aux_monitor' not in terms


def test_legacy_aux_weight_positive_adds_mapped_c5_term():
    """When `legacy_aux_weight>0`, fusion total must be augmented by
    `legacy_aux_weight * loss_func(output_dict['aux'], target).total`."""
    fused_terms = {
        'vote': torch.tensor(1.0),
        'dice': torch.tensor(2.0),
        'affinity': torch.tensor(3.0),
        'total': torch.tensor(0.0),
    }
    inner_terms = {'affinity': torch.tensor(5.0), 'total': torch.tensor(20.0)}
    outer_terms = {'affinity': torch.tensor(7.0), 'total': torch.tensor(30.0)}
    aux_terms = {'affinity': torch.tensor(0.0), 'total': torch.tensor(11.0)}

    solver = object.__new__(Solver)
    solver.fusion_enabled = True
    solver.fusion_lambda_fused = 0.3
    solver.args = SimpleNamespace(
        conn_fusion='gate',
        conn_aux_c3_weight=0.3,
        conn_aux_c5_weight=0.2,
        fusion_gate_reg_weight=0.01,
        use_seg_aux=False,
        legacy_aux_weight=0.3,
        debug_loss_breakdown=False,
    )
    solver.loss_func = _SequenceLoss([fused_terms, inner_terms, aux_terms])
    solver.loss_func_outer = _SequenceLoss([outer_terms])

    output_dict = {
        'fused': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'inner': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'outer': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'aux': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
    }
    target = torch.zeros((1, 8, 4, 4), dtype=torch.float32)

    total, terms = solver._compute_fusion_profile_loss(
        output_dict,
        target,
        binary_gt=None,
        collect_edge_stats=False,
    )

    # vote+dice+0.3*aff + 0.3*aux_total = 3.9 + 0.3*11.0 = 7.2
    assert torch.isclose(total, torch.tensor(7.2), atol=1e-6)
    assert 'legacy_aux' in terms
    assert torch.isclose(terms['legacy_aux'], torch.tensor(11.0), atol=1e-6)


def test_debug_loss_breakdown_monitors_legacy_aux_without_changing_total():
    """When debug_loss_breakdown is on but legacy_aux_weight=0, the term should
    be exposed for logging but must NOT be added to total."""
    fused_terms = {
        'vote': torch.tensor(1.0),
        'dice': torch.tensor(2.0),
        'affinity': torch.tensor(3.0),
        'total': torch.tensor(0.0),
    }
    inner_terms = {'affinity': torch.tensor(5.0), 'total': torch.tensor(20.0)}
    outer_terms = {'affinity': torch.tensor(7.0), 'total': torch.tensor(30.0)}
    aux_terms = {'affinity': torch.tensor(0.0), 'total': torch.tensor(11.0)}

    solver = object.__new__(Solver)
    solver.fusion_enabled = True
    solver.fusion_lambda_fused = 0.3
    solver.args = SimpleNamespace(
        conn_fusion='gate',
        conn_aux_c3_weight=0.3,
        conn_aux_c5_weight=0.2,
        fusion_gate_reg_weight=0.01,
        use_seg_aux=False,
        legacy_aux_weight=0.0,
        debug_loss_breakdown=True,
    )
    solver.loss_func = _SequenceLoss([fused_terms, inner_terms, aux_terms])
    solver.loss_func_outer = _SequenceLoss([outer_terms])

    output_dict = {
        'fused': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'inner': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'outer': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        'aux': torch.zeros((1, 8, 4, 4), dtype=torch.float32),
    }
    target = torch.zeros((1, 8, 4, 4), dtype=torch.float32)

    total, terms = solver._compute_fusion_profile_loss(
        output_dict,
        target,
        binary_gt=None,
        collect_edge_stats=False,
    )

    assert torch.isclose(total, torch.tensor(3.9), atol=1e-6)
    assert 'legacy_aux' not in terms
    assert 'legacy_aux_monitor' in terms
    assert torch.isclose(terms['legacy_aux_monitor'], torch.tensor(11.0), atol=1e-6)


def test_loss_breakdown_logger_writes_csv(tmp_path):
    """LossBreakdownLogger writes per-epoch averaged CSV rows."""
    from src.utils.loss_breakdown import LossBreakdownLogger

    save_dir = tmp_path / 'run'
    save_dir.mkdir()
    logger = LossBreakdownLogger(str(save_dir))
    logger.accumulate({'loss_total': 1.0, 'vote': 0.5})
    logger.accumulate({'loss_total': 3.0, 'vote': 1.5})
    logger.flush(epoch=1)

    logger.accumulate({'loss_total': 2.0, 'vote': 1.0})
    logger.flush(epoch=2)

    csv_path = save_dir / 'loss_breakdown.csv'
    rows = csv_path.read_text().strip().splitlines()
    assert rows[0].startswith('epoch')
    assert 'loss_total' in rows[0]
    assert 'vote' in rows[0]
    # epoch 1 averages: 2.0, 1.0
    assert '1,2.000000,1.000000' in rows[1]
    # epoch 2: single accumulate -> 2.0, 1.0
    assert '2,2.000000,1.000000' in rows[2]
