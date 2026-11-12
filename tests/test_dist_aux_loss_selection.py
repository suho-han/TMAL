from types import SimpleNamespace

import torch

from connect_loss import connect_loss
from src.losses import dist_aux_regression_loss


def _patch_cuda_to_noop(monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self, raising=False)


def _build_shift_matrix(size):
    matrix = torch.zeros((1, 1, size, size), dtype=torch.float32)
    for i in range(size - 1):
        matrix[0, 0, i, i + 1] = 1.0
    return matrix


def _make_args(dataset='chase', dist_aux_loss='smooth_l1', gamma=1.0):
    return SimpleNamespace(
        num_class=1,
        dataset=dataset,
        dist_aux_loss=dist_aux_loss,
        dist_sf_l1_gamma=gamma,
    )


def _make_loss_module(dataset='chase', dist_aux_loss='smooth_l1', gamma=1.0, size=4):
    args = _make_args(dataset=dataset, dist_aux_loss=dist_aux_loss, gamma=gamma)
    return connect_loss(
        args,
        _build_shift_matrix(size),
        _build_shift_matrix(size),
        label_mode='dist',
        conn_num=8,
        sigma=2.0,
    )


def test_dist_aux_regression_loss_is_finite_for_both_modes(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    pred = torch.full((2, 8, 4, 4), 0.4, dtype=torch.float32)
    target = torch.full((2, 8, 4, 4), 0.6, dtype=torch.float32)

    smooth_module = _make_loss_module(dist_aux_loss='smooth_l1')
    gjml_module = _make_loss_module(dist_aux_loss='gjml_sf_l1')

    smooth_loss = smooth_module.dist_aux_regression_loss(pred, target)
    gjml_loss = gjml_module.dist_aux_regression_loss(pred, target)

    assert torch.isfinite(smooth_loss)
    assert torch.isfinite(gjml_loss)


def test_cl_dice_is_finite_for_dist_aux_regression_loss(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    module = _make_loss_module(dist_aux_loss='cl_dice')
    pred = torch.full((2, 8, 4, 4), 0.4, dtype=torch.float32)
    target = torch.full((2, 8, 4, 4), 0.6, dtype=torch.float32)

    loss = module.dist_aux_regression_loss(pred, target)

    assert torch.isfinite(loss)


def test_gjml_sf_l1_is_near_zero_when_prediction_matches_target(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    module = _make_loss_module(dist_aux_loss='gjml_sf_l1')
    target = torch.linspace(0.1, 0.9, steps=2 * 8 * 4 * 4, dtype=torch.float32).view(2, 8, 4, 4)
    loss = module.dist_aux_regression_loss(target, target)
    assert float(loss.item()) < 1e-6


def test_cl_dice_is_near_zero_when_prediction_matches_target(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    module = _make_loss_module(dist_aux_loss='cl_dice')
    torch.manual_seed(0)
    target = (torch.rand((2, 8, 4, 4), dtype=torch.float32) > 0.8).float()
    same_loss = module.dist_aux_regression_loss(target, target)
    other = (torch.rand((2, 8, 4, 4), dtype=torch.float32) > 0.8).float()
    diff_loss = module.dist_aux_regression_loss(target, other)

    assert float(same_loss.item()) < 1e-6
    assert float(diff_loss.item()) > float(same_loss.item())


def test_chase_dist_mode_keeps_bicon_zero(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    module = _make_loss_module(dataset='chase', dist_aux_loss='gjml_sf_l1')
    affinity_map = torch.zeros((1, 8, 4, 4), dtype=torch.float32)
    target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    target[:, :, 1:3, 1:3] = 0.8

    _, loss_dict = module(affinity_map, target, return_details=True)

    assert float(loss_dict['bicon'].item()) == 0.0


def test_non_chase_dist_mode_computes_bicon(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    module = _make_loss_module(dataset='isic', dist_aux_loss='gjml_sf_l1')
    affinity_map = torch.zeros((1, 8, 4, 4), dtype=torch.float32)
    target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    target[:, :, 1:3, 1:3] = 0.8

    _, loss_dict = module(affinity_map, target, return_details=True)

    assert float(loss_dict['bicon'].item()) > 0.0


def test_cl_dice_dist_mode_uses_stable_affinity_and_bicon_terms(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    module = _make_loss_module(dataset='isic', dist_aux_loss='cl_dice', size=8)
    affinity_map = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
    target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    target[:, :, 2:6, 2:6] = 0.8

    _, loss_dict = module(affinity_map, target, return_details=True)

    assert torch.isfinite(loss_dict['affinity'])
    assert torch.isfinite(loss_dict['bicon'])
    assert torch.isfinite(loss_dict['total'])


def test_connect_loss_wrapper_matches_modular_dist_aux_function(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)
    module = _make_loss_module(dist_aux_loss='cl_dice')
    pred = (torch.rand((2, 8, 4, 4), dtype=torch.float32) > 0.7).float()
    target = (torch.rand((2, 8, 4, 4), dtype=torch.float32) > 0.7).float()

    wrapped = module.dist_aux_regression_loss(pred, target)
    direct = dist_aux_regression_loss(
        pred,
        target,
        loss_name='cl_dice',
        gamma=module.dist_sf_l1_gamma,
    )

    assert torch.allclose(wrapped, direct)


def test_cl_dice_modular_loss_supports_backward():
    pred = torch.rand((1, 8, 6, 6), dtype=torch.float32, requires_grad=True)
    target = (torch.rand((1, 8, 6, 6), dtype=torch.float32) > 0.7).float()

    loss = dist_aux_regression_loss(pred, target, loss_name='cl_dice', gamma=1.0)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_cl_dice_loss_is_bounded_and_finite():
    pred = torch.rand((2, 8, 16, 16), dtype=torch.float32)
    target = torch.rand((2, 8, 16, 16), dtype=torch.float32)

    loss = dist_aux_regression_loss(pred, target, loss_name='cl_dice', gamma=1.0)

    assert torch.isfinite(loss)
    assert 0.0 <= float(loss.item()) <= 1.0
