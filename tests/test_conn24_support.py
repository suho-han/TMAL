from types import SimpleNamespace

import torch

from connect_loss import Bilateral_voting, Bilateral_voting_kxk, connect_loss, distance_affinity_matrix, resolve_connectivity_layout, shift_n_directions


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


def _make_args(label_mode='binary', dataset='chase', dist_aux_loss='smooth_l1', gamma=1.0):
    return SimpleNamespace(
        num_class=1,
        dataset=dataset,
        label_mode=label_mode,
        dist_aux_loss=dist_aux_loss,
        dist_sf_l1_gamma=gamma,
    )


def test_shift_n_directions_conn24_matches_expected_offsets():
    x = torch.arange(1, 1 + 5 * 5, dtype=torch.float32).view(1, 1, 5, 5)
    layout = resolve_connectivity_layout(24)
    shifted = shift_n_directions(x, 24)

    assert len(shifted) == 24
    assert layout['offsets'][:8] == [
        (1, 1),
        (1, 0),
        (1, -1),
        (0, 1),
        (0, -1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ]

    for tensor, (dy, dx) in zip(shifted, layout['offsets']):
        expected = _reference_shift(x, dy, dx)
        assert torch.equal(tensor, expected)


def test_distance_affinity_matrix_conn24_shape():
    dist_map = torch.rand((2, 1, 6, 7), dtype=torch.float32)
    affinity = distance_affinity_matrix(dist_map, conn_num=24, sigma=2.0)
    assert affinity.shape == (2, 24, 6, 7)


def test_bilateral_voting_kxk_conn24_first8_aligns_with_bilateral(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)

    batch = 2
    height = 6
    width = 7
    hori = _build_shift_matrix(width).repeat(batch, 1, 1, 1)
    vert = _build_shift_matrix(height).repeat(batch, 1, 1, 1)

    affinity_24 = torch.zeros((batch, 1, 24, height, width), dtype=torch.float32)
    affinity_24[:, :, :8] = torch.rand((batch, 1, 8, height, width), dtype=torch.float32)

    pred_8, vote_8 = Bilateral_voting(affinity_24[:, :, :8], hori, vert)
    pred_24, vote_24 = Bilateral_voting_kxk(affinity_24, hori, vert, conn_num=5)

    assert torch.allclose(pred_8, pred_24, atol=1e-6, rtol=1e-5)
    assert torch.allclose(vote_8, vote_24[:, :, :8], atol=1e-6, rtol=1e-5)


def test_single_class_forward_supports_conn24_binary_and_dist(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)

    size = 6
    hori = _build_shift_matrix(size)
    vert = _build_shift_matrix(size)
    affinity_map = torch.zeros((1, 24, size, size), dtype=torch.float32)

    binary_target = torch.zeros((1, 1, size, size), dtype=torch.float32)
    binary_target[:, :, 2:4, 2:4] = 1.0
    binary_module = connect_loss(
        _make_args(label_mode='binary'),
        hori,
        vert,
        label_mode='binary',
        conn_num=24,
        sigma=2.0,
    )
    binary_loss, binary_terms = binary_module(affinity_map, binary_target, return_details=True)
    assert torch.isfinite(binary_loss)
    assert torch.isfinite(binary_terms['edge'])

    dist_target = torch.zeros((1, 1, size, size), dtype=torch.float32)
    dist_target[:, :, 2:4, 2:4] = 0.8
    dist_module = connect_loss(
        _make_args(label_mode='dist'),
        hori,
        vert,
        label_mode='dist',
        conn_num=24,
        sigma=2.0,
    )
    dist_loss, dist_terms = dist_module(affinity_map, dist_target, return_details=True)
    assert torch.isfinite(dist_loss)
    assert torch.isfinite(dist_terms['affinity'])
