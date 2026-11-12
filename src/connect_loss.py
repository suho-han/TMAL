import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imread, imsave
from torch.autograd import Function, Variable
from torch.nn.modules.loss import _Loss

from src.losses.dist_aux import dist_aux_regression_loss as dist_aux_regression_loss_fn

CANONICAL_8_OFFSETS = [
    (1, 1),
    (1, 0),
    (1, -1),
    (0, 1),
    (0, -1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
]

OUTER_8_OFFSETS = [
    (-2, -2),
    (-2, 0),
    (-2, 2),
    (0, -2),
    (0, 2),
    (2, -2),
    (2, 0),
    (2, 2),
]

SCALE3_8_OFFSETS = [
    (-3, -3),
    (-3, 0),
    (-3, 3),
    (0, -3),
    (0, 3),
    (3, -3),
    (3, 0),
    (3, 3),
]

SCALE4_8_OFFSETS = [
    (-4, -4),
    (-4, 0),
    (-4, 4),
    (0, -4),
    (0, 4),
    (4, -4),
    (4, 0),
    (4, 4),
]

MIDDLE_8_OFFSETS = [
    (-2, -1), (-2,  1),
    (-1, -2), (-1,  2),
    (1, -2), (1,  2),
    (2, -1), (2,  1),
]


def build_center_excluding_offsets(radius):
    offsets = [
        offset for offset in CANONICAL_8_OFFSETS
        if abs(offset[0]) <= radius and abs(offset[1]) <= radius
    ]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            if (dy, dx) not in offsets:
                offsets.append((dy, dx))
    return offsets


CONNECTIVITY_5X5_OFFSETS = build_center_excluding_offsets(radius=2)

CONNECTIVITY_LAYOUTS = {
    'standard8': {
        'name': 'standard8',
        'channel_count': 8,
        'kernel_size': 3,
        'include_center': False,
        'offsets': CANONICAL_8_OFFSETS,
    },
    'full24': {
        'name': 'full24',
        'channel_count': 24,
        'kernel_size': 5,
        'include_center': False,
        'offsets': CONNECTIVITY_5X5_OFFSETS,
    },
    'out8': {
        'name': 'out8',
        'channel_count': 8,
        'kernel_size': 5,
        'include_center': False,
        'offsets': OUTER_8_OFFSETS,
    },
    'scale3_8': {
        'name': 'scale3_8',
        'channel_count': 8,
        'kernel_size': 7,
        'include_center': False,
        'offsets': SCALE3_8_OFFSETS,
    },
    'scale4_8': {
        'name': 'scale4_8',
        'channel_count': 8,
        'kernel_size': 9,
        'include_center': False,
        'offsets': SCALE4_8_OFFSETS,
    },
    'mid8': {
        'name': 'mid8',
        'channel_count': 8,
        'kernel_size': 5,
        'include_center': False,
        'offsets': MIDDLE_8_OFFSETS,
    },
}

FINAL_DECISIONS = ('max', 'mean', 'weighted_mean')


def normalize_final_decision(value='max'):
    final_decision = 'max' if value is None else str(value)
    if final_decision not in FINAL_DECISIONS:
        raise ValueError(
            f"Unsupported final_decision {final_decision}, "
            f"supported: {list(FINAL_DECISIONS)}"
        )
    return final_decision


def default_connectivity_layout_name(conn_num):
    if conn_num == 8:
        return 'standard8'
    if conn_num == 24:
        return 'full24'
    raise ValueError(f"Unsupported conn_num {conn_num}, only 8 and 24 are supported")


def normalize_conn_layout(conn_num, conn_layout=None):
    layout_name = default_connectivity_layout_name(conn_num) if conn_layout is None else str(conn_layout)
    if layout_name not in CONNECTIVITY_LAYOUTS:
        raise ValueError(
            f"Unsupported conn_layout {layout_name}, "
            f"supported: {sorted(CONNECTIVITY_LAYOUTS)}"
        )

    if conn_num == 8 and layout_name not in {'standard8', 'out8', 'mid8', 'scale3_8', 'scale4_8'}:
        raise ValueError("conn_num=8 supports only conn_layout in {'standard8', 'mid8', 'out8', 'scale3_8', 'scale4_8'}")
    if conn_num == 24 and layout_name != 'full24':
        raise ValueError("conn_num=24 supports only conn_layout='full24'")
    return layout_name


def resolve_connectivity_layout(conn_num, conn_layout=None):
    layout_name = normalize_conn_layout(conn_num, conn_layout)
    layout = dict(CONNECTIVITY_LAYOUTS[layout_name])
    layout['conn_num'] = layout['channel_count']
    layout['offsets'] = list(layout['offsets'])
    return layout


def is_default_connectivity_layout(conn_num, conn_layout):
    return normalize_conn_layout(conn_num, conn_layout) == default_connectivity_layout_name(conn_num)


def _shift_tensor(x, dy, dx):
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


def connectivity_matrix(multimask, class_num):

    ##### converting segmentation masks to directional affinity maps ####

    [batch, _, rows, cols] = multimask.shape
    # batch = 1
    conn = torch.zeros([batch, class_num*8, rows, cols]).cuda()
    for i in range(class_num):
        mask = multimask[:, i, :, :]
        # print(mask.shape)
        up = torch.zeros([batch, rows, cols]).cuda()  # move the orignal mask to up
        down = torch.zeros([batch, rows, cols]).cuda()
        left = torch.zeros([batch, rows, cols]).cuda()
        right = torch.zeros([batch, rows, cols]).cuda()
        up_left = torch.zeros([batch, rows, cols]).cuda()
        up_right = torch.zeros([batch, rows, cols]).cuda()
        down_left = torch.zeros([batch, rows, cols]).cuda()
        down_right = torch.zeros([batch, rows, cols]).cuda()

        up[:, :rows-1, :] = mask[:, 1:rows, :]
        down[:, 1:rows, :] = mask[:, 0:rows-1, :]
        left[:, :, :cols-1] = mask[:, :, 1:cols]
        right[:, :, 1:cols] = mask[:, :, :cols-1]
        up_left[:, 0:rows-1, 0:cols-1] = mask[:, 1:rows, 1:cols]
        up_right[:, 0:rows-1, 1:cols] = mask[:, 1:rows, 0:cols-1]
        down_left[:, 1:rows, 0:cols-1] = mask[:, 0:rows-1, 1:cols]
        down_right[:, 1:rows, 1:cols] = mask[:, 0:rows-1, 0:cols-1]

        conn[:, (i*8)+0, :, :] = mask*down_right
        conn[:, (i*8)+1, :, :] = mask*down
        conn[:, (i*8)+2, :, :] = mask*down_left
        conn[:, (i*8)+3, :, :] = mask*right
        conn[:, (i*8)+4, :, :] = mask*left
        conn[:, (i*8)+5, :, :] = mask*up_right
        conn[:, (i*8)+6, :, :] = mask*up
        conn[:, (i*8)+7, :, :] = mask*up_left

    conn = conn.float()
    conn = conn.squeeze()
    # keep batch axis for batch-size 1 to avoid shape mismatch in downstream BCE
    if conn.dim() == 3:
        conn = conn.unsqueeze(0)
    # print(conn.shape)
    return conn


def connectivity_matrix_by_layout(multimask, class_num, conn_num, conn_layout=None):
    layout = resolve_connectivity_layout(conn_num, conn_layout)
    if layout['name'] == 'standard8':
        return connectivity_matrix(multimask, class_num)

    [batch, _, rows, cols] = multimask.shape
    offsets = layout['offsets']
    conn = torch.zeros([batch, class_num * len(offsets), rows, cols]).cuda()
    for i in range(class_num):
        mask = multimask[:, i:i + 1, :, :]
        for channel_idx, shifted in enumerate(shift_n_directions(mask, conn_num, conn_layout=layout['name'])):
            conn[:, (i * len(offsets)) + channel_idx, :, :] = (mask * shifted).squeeze(1)

    conn = conn.float()
    conn = conn.squeeze()
    if conn.dim() == 3:
        conn = conn.unsqueeze(0)
    return conn


def shift_n_directions(x, conn_num, conn_layout=None):
    # x: (B, 1, H, W)
    offsets = resolve_connectivity_layout(conn_num, conn_layout)['offsets']
    return [_shift_tensor(x, dy, dx) for dy, dx in offsets]


def distance_affinity_matrix(dist_map, conn_num, sigma=2.0, conn_layout=None):
    """
    dist_map : (B, H, W) distance map for each class
    return: (B, conn_num, H, W) directional affinity map for each class

    우선 single-class만 구현
    """

    if dist_map.ndim == 4:
        if dist_map.shape[1] != 1:
            raise ValueError(f"single-class dist_map must have shape (B,1,H,W), got {dist_map.shape}")
        dist_map = dist_map[:, 0, :, :]
    elif dist_map.ndim != 3:
        raise ValueError(f"dist_map must have shape (B,H,W) or (B,1,H,W), got {dist_map.shape}")

    D = dist_map.unsqueeze(1)  # (B,1,H,W)

    neighbors_D = shift_n_directions(D, conn_num, conn_layout=conn_layout)

    conn_list = []
    for D_k in neighbors_D:
        Ck = D * D_k * torch.exp(-torch.abs(D - D_k) / sigma)
        conn_list.append(Ck)

    conn = torch.cat(conn_list, dim=1)   # (B,conn_num,H,W)
    return conn


def Bilateral_voting(c_map, hori_translation, verti_translation, final_decision='max'):

    #### bilateral voting and convert directional affinity into a distance score map. ####

    final_decision = normalize_final_decision(final_decision)
    batch, class_num, channel, row, column = c_map.size()
    vote_out = torch.zeros([batch, class_num, channel, row, column]).cuda()
    # print(vote_out.shape,affinity_map.shape,hori_translation.shape)
    right = (torch.bmm(c_map[:, :, 4].contiguous().view(-1, row, column), hori_translation.view(-1, column, column))).view(batch, class_num, row, column)

    left = (torch.bmm(c_map[:, :, 3].contiguous().view(-1, row, column), hori_translation.transpose(3, 2).view(-1, column, column))).view(batch, class_num, row, column)

    left_bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row), c_map[:, :, 5].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    left_bottom = (torch.bmm(left_bottom.view(-1, row, column), hori_translation.transpose(3, 2).view(-1, column, column))).view(batch, class_num, row, column)
    right_above = (torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 2].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    right_above = (torch.bmm(right_above.view(-1, row, column), hori_translation.view(-1, column, column))).view(batch, class_num, row, column)
    left_above = (torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 0].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    left_above = (torch.bmm(left_above.view(-1, row, column), hori_translation.transpose(3, 2).view(-1, column, column))).view(batch, class_num, row, column)
    bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row), c_map[:, :, 6].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    up = (torch.bmm(verti_translation.view(-1, row, row), c_map[:, :, 1].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    right_bottom = (torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row), c_map[:, :, 7].contiguous().view(-1, row, column))).view(batch, class_num, row, column)
    right_bottom = (torch.bmm(right_bottom.view(-1, row, column), hori_translation.view(-1, column, column))).view(batch, class_num, row, column)

    vote_out[:, :, 0] = (c_map[:, :, 0]) * (right_bottom)
    vote_out[:, :, 1] = (c_map[:, :, 1]) * (bottom)
    vote_out[:, :, 2] = (c_map[:, :, 2]) * (left_bottom)
    vote_out[:, :, 3] = (c_map[:, :, 3]) * (right)
    vote_out[:, :, 4] = (c_map[:, :, 4]) * (left)
    vote_out[:, :, 5] = (c_map[:, :, 5]) * (right_above)
    vote_out[:, :, 6] = (c_map[:, :, 6]) * (up)
    vote_out[:, :, 7] = (c_map[:, :, 7]) * (left_above)

    if final_decision == 'max':
        pred_mask, _ = torch.max(vote_out, dim=2)
    elif final_decision == 'mean':
        pred_mask = torch.mean(vote_out, dim=2)
    else:
        # Straight directions have weight 1; diagonals have weight 0.5.
        weights = c_map.new_tensor(
            [0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.5]
        ).view(1, 1, channel, 1, 1)
        pred_mask = torch.sum(vote_out * weights, dim=2) / weights.sum()

    ###
    # vote_out = vote_out.view(batch,-1, row, column)
    return pred_mask, vote_out


def shift_map(x, hori_translation, verti_translation, dy, dx):
    # x: (N, H, W)

    out = x

    if hori_translation.dim() == 4:
        hori_translation = hori_translation.squeeze(1)
    if verti_translation.dim() == 4:
        verti_translation = verti_translation.squeeze(1)

    if hori_translation.dim() != 3 or verti_translation.dim() != 3:
        raise ValueError("translation tensors must be 3D or 4D")

    n = out.shape[0]
    if hori_translation.shape[0] != n:
        if hori_translation.shape[0] == 1:
            hori_translation = hori_translation.repeat(n, 1, 1)
        elif n % hori_translation.shape[0] == 0:
            hori_translation = hori_translation.repeat_interleave(n // hori_translation.shape[0], dim=0)
        else:
            raise ValueError(f"hori_translation batch mismatch: got {hori_translation.shape[0]}, expected {n}")

    if verti_translation.shape[0] != n:
        if verti_translation.shape[0] == 1:
            verti_translation = verti_translation.repeat(n, 1, 1)
        elif n % verti_translation.shape[0] == 0:
            verti_translation = verti_translation.repeat_interleave(n // verti_translation.shape[0], dim=0)
        else:
            raise ValueError(f"verti_translation batch mismatch: got {verti_translation.shape[0]}, expected {n}")

    # vertical shift
    if dy > 0:
        for _ in range(dy):
            out = torch.bmm(verti_translation.transpose(2, 1), out)
    elif dy < 0:
        for _ in range(-dy):
            out = torch.bmm(verti_translation, out)

    # horizontal shift
    if dx > 0:
        for _ in range(dx):
            out = torch.bmm(out, hori_translation)
    elif dx < 0:
        for _ in range(-dx):
            out = torch.bmm(out, hori_translation.transpose(2, 1))

    return out


def Bilateral_voting_kxk(
    affinity_map,
    hori_translation,
    verti_translation,
    conn_num=5,
    second_weight=1.0,
    offsets=None,
):
    # affinity_map: (B, C, K, H, W)

    B, C, K, H, W = affinity_map.shape

    if offsets is not None:
        offsets = list(offsets)
    else:
        # Legacy compatibility path: older call-sites infer offsets from
        # kernel radius/conn_num. New layout-aware paths should pass offsets
        # explicitly via resolve_connectivity_layout(...)[\"offsets\"].
        radius = conn_num // 2
        full_k = (2 * radius + 1) ** 2
        without_center_k = full_k - 1
        if K == full_k:
            include_center = True
        elif K == without_center_k:
            include_center = False
        else:
            raise ValueError(
                f"affinity_map channel mismatch for conn_num={conn_num}: "
                f"got K={K}, expected {without_center_k} or {full_k}"
            )

        offsets = [
            offset for offset in CANONICAL_8_OFFSETS
            if abs(offset[0]) <= radius and abs(offset[1]) <= radius
        ]
        if include_center and (0, 0) not in offsets:
            offsets.append((0, 0))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if not include_center and dy == 0 and dx == 0:
                    continue
                if (dy, dx) not in offsets:
                    offsets.append((dy, dx))

    if K != len(offsets):
        raise ValueError(f"Internal offset size mismatch: K={K}, offsets={len(offsets)}")

    offset_to_idx = {o: i for i, o in enumerate(offsets)}

    vote_out = torch.zeros_like(affinity_map)

    for dy, dx in offsets:
        idx = offset_to_idx[(dy, dx)]
        reverse_offset = (-dy, -dx)
        if reverse_offset not in offset_to_idx:
            raise ValueError(f"Missing reverse offset {reverse_offset} for offset {(dy, dx)}")
        rev_idx = offset_to_idx[reverse_offset]

        rev_map = affinity_map[:, :, rev_idx].contiguous().view(-1, H, W)
        shifted_rev = shift_map(rev_map, hori_translation, verti_translation, dy, dx)
        shifted_rev = shifted_rev.view(B, C, H, W)

        vote_out[:, :, idx] = affinity_map[:, :, idx] * shifted_rev

    pred_mask, _ = torch.max(vote_out, dim=2)
    return pred_mask, vote_out


class dice_loss(nn.Module):
    def __init__(self, bin_wide, density):
        super(dice_loss, self).__init__()
        self.bin_wide = bin_wide
        self.density = density

    def soft_dice_coeff(self, y_pred, y_true, class_i=None):
        smooth = 0.0001  # may change

        i = torch.sum(y_true, dim=(1, 2))
        j = torch.sum(y_pred, dim=(1, 2))
        intersection = torch.sum(y_true * y_pred, dim=(1, 2))

        score = (2. * intersection + smooth) / (i + j + smooth)

        if self.bin_wide:
            weight = density_weight(self.bin_wide[class_i], i, self.density[class_i])
            return (1-score)*weight
        else:
            return (1-score)

    def soft_dice_loss(self, y_pred, y_true, class_i=None):
        loss = self.soft_dice_coeff(y_true, y_pred, class_i)
        return loss.mean()

    def __call__(self, y_pred, y_true, class_i=None):

        b = self.soft_dice_loss(y_true, y_pred, class_i)
        return b


def density_weight(bin_wide, gt_cnt, density):

    index = gt_cnt//bin_wide

    selected_density = [density[index[i].long()] for i in range(gt_cnt.shape[0])]
    selected_density = torch.tensor(selected_density).cuda()
    log_inv_density = torch.log(1/(selected_density+0.0001))

    return log_inv_density


class connect_loss(nn.Module):
    def __init__(
        self,
        args,
        hori_translation,
        verti_translation,
        density=None,
        bin_wide=None,
        label_mode=None,
        conn_num=8,
        sigma=2.0,
        conn_layout=None,
        final_decision='max',
    ):
        super(connect_loss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.BCEloss = nn.BCELoss(reduction='none')
        self.dice_loss = dice_loss(bin_wide=bin_wide, density=density)
        self.bin_wide = bin_wide
        self.density = density
        self.args = args

        self.verti_translation = verti_translation
        self.hori_translation = hori_translation

        self.label_mode = label_mode
        self.layout = resolve_connectivity_layout(conn_num, conn_layout)
        self.conn_layout = self.layout['name']
        self.conn_num = self.layout['channel_count']
        self.final_decision = normalize_final_decision(final_decision)
        if self.final_decision != 'max' and self.conn_layout != 'standard8':
            raise ValueError(
                "final_decision other than 'max' is supported only for "
                "conn_layout='standard8'"
            )
        self.sigma = sigma
        self.dist_aux_loss_name = getattr(self.args, 'dist_aux_loss', 'smooth_l1')
        self.dist_sf_l1_gamma = getattr(self.args, 'dist_sf_l1_gamma', 1.0)
        self._collect_dist_edge_stats = False
        self._dist_edge_stats = {
            'edge_mean_sum': 0.0,
            'edge_nonzero_sum': 0.0,
            'count': 0,
        }

    def set_dist_edge_stat_collection(self, enabled):
        self._collect_dist_edge_stats = bool(enabled)

    def reset_dist_edge_stats(self):
        self._dist_edge_stats = {
            'edge_mean_sum': 0.0,
            'edge_nonzero_sum': 0.0,
            'count': 0,
        }

    def get_dist_edge_stats(self):
        count = self._dist_edge_stats['count']
        if count == 0:
            return None
        return {
            'edge_mean': self._dist_edge_stats['edge_mean_sum'] / count,
            'edge_nonzero_ratio': self._dist_edge_stats['edge_nonzero_sum'] / count,
            'count': count,
        }

    def dist_target_to_mask(self, target):
        return (target > 0).float()

    def dist_aux_regression_loss(self, pred, target):
        # Keep method-level API for backward compatibility while delegating
        # implementation to fork-specific modularized losses.
        return dist_aux_regression_loss_fn(
            pred,
            target,
            loss_name=self.dist_aux_loss_name,
            gamma=self.dist_sf_l1_gamma,
        )

    def binary_edge_target_from_affinity(self, affinity_target):
        # `affinity_target` here is expected to be a binary-style connectivity map.
        # For distance mode, pass connectivity rebuilt from `mask_target`, not the
        # raw `distance_affinity_matrix(...)` output.
        class_conn = affinity_target.view(
            [
                affinity_target.shape[0],
                self.args.num_class,
                self.conn_num,
                affinity_target.shape[2],
                affinity_target.shape[3],
            ]
        )
        sum_conn = torch.sum(class_conn, dim=2)
        return torch.where(
            (sum_conn < self.conn_num) & (sum_conn > 0),
            torch.full_like(sum_conn, 1),
            torch.full_like(sum_conn, 0),
        )

    def edge_loss(self, vote_out, edge):
        pred_mask_min, _ = torch.min(vote_out.cuda(), dim=2)
        pred_mask_min = pred_mask_min*edge
        minloss = self.BCEloss(pred_mask_min, torch.full_like(pred_mask_min, 0))
        return (minloss.sum()/(pred_mask_min.sum().clamp(min=1e-8)))  # +maxloss

    def dist_edge_loss(self, vote_out, mask_target):
        # Distance mode still optimizes continuous directional affinity, but the
        # final segmentation target is binary. Build edge supervision from the
        # binary vessel mask instead of the weak continuous 4n(1-n) target.
        # Rebuild binary connectivity from the derived vessel mask before
        # computing the edge target. This keeps the edge definition aligned with
        # the binary path instead of reinterpreting raw distance affinity values.
        binary_affinity_target = connectivity_matrix_by_layout(
            mask_target,
            self.args.num_class,
            conn_num=self.conn_num,
            conn_layout=self.conn_layout,
        )
        edge = self.binary_edge_target_from_affinity(binary_affinity_target)
        return self.edge_loss(vote_out, edge), edge

    def forward(self, affinity_map, target, return_details=False):
        if self.args.num_class == 1:
            loss, loss_dict = self.single_class_forward(affinity_map, target)
        else:
            loss, loss_dict = self.multi_class_forward(affinity_map, target)

        if return_details:
            return loss, loss_dict
        return loss

    def multi_class_forward(self, affinity_map, target):
        #######
        # affinity_map: (B, 8*C, H, W), B: batch, C: class number
        # target: (B, H, W)
        #######
        if self.layout['name'] != 'standard8':
            raise ValueError("multi-class mode currently supports only conn_layout='standard8'")
        target = target.type(torch.LongTensor).cuda()
        batch_num = affinity_map.shape[0]
        onehotmask = F.one_hot(target.long(), self.args.num_class)  # change it to your class number if needed
        onehotmask = onehotmask.permute(0, 3, 1, 2)
        onehotmask = onehotmask.float()

        ### build directional affinity target ###
        affinity_target = connectivity_matrix(onehotmask, self.args.num_class)  # (B, 8*C, H, W)

        # matrix for shifting
        hori_translation = self.hori_translation.repeat(batch_num, 1, 1, 1).cuda()
        verti_translation = self.verti_translation.repeat(batch_num, 1, 1, 1).cuda()
        # shape for hori_translation: [Batch, NumClass, H, H]
        # shape for verti_translation: [Batch, NumClass, W, W]

        dice_l = 0

        ### get edges gt###
        class_conn = affinity_target.view([affinity_map.shape[0], self.args.num_class, 8, affinity_map.shape[2], affinity_map.shape[3]])
        sum_conn = torch.sum(class_conn, dim=2)
        # edge: (B, C, H, W)
        edge = torch.where((sum_conn < 8) & (sum_conn > 0), torch.full_like(sum_conn, 1), torch.full_like(sum_conn, 0))

        ### bilateral voting #####
        # final pred: (B, C, H, W), vote_out: (B, C, 8, H, W), bicon_map: (B, 8*C, H, W)
        class_pred = affinity_map.view([affinity_map.shape[0], self.args.num_class, 8, affinity_map.shape[2], affinity_map.shape[3]])
        final_pred, vote_out = Bilateral_voting(class_pred, hori_translation, verti_translation, self.final_decision)
        _, bicon_map = Bilateral_voting(F.sigmoid(class_pred), hori_translation, verti_translation, self.final_decision)
        bicon_map = bicon_map.view(affinity_target.shape)

        edge_l = self.edge_loss(F.sigmoid(vote_out), edge)

        pred = F.softmax(final_pred, dim=1)

        for j in range(1, self.args.num_class):
            dice_l += self.dice_loss(pred[:, j, :, :], onehotmask[:, j], j-1)

        ce_loss = F.cross_entropy(final_pred, target)
        affinity_l = self.BCEloss(F.sigmoid(affinity_map), affinity_target).mean()
        bicon_l = self.BCEloss(bicon_map, affinity_target).mean()
        loss = ce_loss + affinity_l + edge_l + 0.2 * bicon_l + dice_l  # + bce_loss# +loss_out_dice# +sum_l # + edge_l+loss_out_dice
        loss_dict = {
            'total': loss,
            'ce': ce_loss,
            'affinity': affinity_l,
            'edge': edge_l,
            'bicon': bicon_l,
            'dice': dice_l,
        }

        return loss, loss_dict

    def single_class_forward(self, c_map, target):
        #######
        # affinity_map: (B, 8, H, W), B: batch, C: class number
        # target: (B, 1, H, W)
        #######

        batch_num = c_map.shape[0]
        target = target.float()

        ### build directional affinity target ###
        if self.label_mode == 'binary':
            con_target = connectivity_matrix_by_layout(
                target,
                self.args.num_class,
                conn_num=self.conn_num,
                conn_layout=self.conn_layout,
            )
            affinity_target = con_target
        else:
            affinity_target = distance_affinity_matrix(
                target,
                conn_num=self.conn_num,
                sigma=self.sigma,
                conn_layout=self.conn_layout,
            )

        # matrix for shifting
        hori_translation = self.hori_translation.repeat(batch_num, 1, 1, 1).cuda()
        verti_translation = self.verti_translation.repeat(batch_num, 1, 1, 1).cuda()

        c_map = F.sigmoid(c_map)

        ### get edges gt###
        class_conn = affinity_target.view([c_map.shape[0], self.args.num_class, self.conn_num, c_map.shape[2], c_map.shape[3]])
        sum_conn = torch.sum(class_conn, dim=2)

        # edge: (B, 1, H, W)

        ### bilateral voting #####
        # pred: (B, 1, H, W) score map (distance score map in distance label_mode),
        # bicon_map: (B, 1, 8, H, W) directional affinity map
        class_pred = c_map.view([c_map.shape[0], self.args.num_class, self.conn_num, c_map.shape[2], c_map.shape[3]])
        if self.layout['name'] == 'standard8':
            pred, bicon_map = Bilateral_voting(class_pred, hori_translation, verti_translation, self.final_decision)
        else:
            pred, bicon_map = Bilateral_voting_kxk(
                class_pred,
                hori_translation,
                verti_translation,
                conn_num=self.layout['kernel_size'],
                offsets=self.layout['offsets'],
            )

        if self.label_mode == 'binary':
            edge = torch.where((sum_conn < self.conn_num) & (sum_conn > 0), torch.full_like(sum_conn, 1), torch.full_like(sum_conn, 0))
            edge_l = self.edge_loss(bicon_map, edge)
            dice_l = self.dice_loss(pred[:, 0], target[:, 0])

            bce_loss = self.BCEloss(pred, target).mean()
            conn_l = self.BCEloss(c_map, con_target).mean()

            if self.args.dataset == 'chase':
                loss = bce_loss + conn_l + edge_l + dice_l
                bicon_l = torch.zeros_like(loss)
            else:
                bicon_l = self.BCEloss(bicon_map.squeeze(1), con_target).mean()
                loss = bce_loss + conn_l + edge_l + 0.2 * bicon_l + dice_l  # + bce_loss# +loss_out_dice# +sum_l # + edge_l+loss_out_dice
            loss_dict = {
                'total': loss,
                'vote': bce_loss,
                'affinity': conn_l,
                'edge': edge_l,
                'bicon': bicon_l,
                'dice': dice_l,
            }

        elif self.label_mode in ['dist', 'dist_inverted']:
            mask_target = self.dist_target_to_mask(target)
            edge_l, edge = self.dist_edge_loss(bicon_map, mask_target)

            if self._collect_dist_edge_stats:
                with torch.no_grad():
                    edge_detached = edge.detach()
                    self._dist_edge_stats['edge_mean_sum'] += edge_detached.mean().item()
                    self._dist_edge_stats['edge_nonzero_sum'] += (edge_detached > 1e-6).float().mean().item()
                    self._dist_edge_stats['count'] += 1

            # Binary path predicts the final vessel mask; keep that property for
            # distance mode as well, while using the distance map as an affinity
            # regression target.
            # vote_loss = self.dist_aux_regression_loss(pred, mask_target).mean()
            vote_loss = self.dist_aux_regression_loss(pred, target).mean()
            dice_l = self.dice_loss(pred[:, 0], mask_target[:, 0])
            if self.dist_aux_loss_name == 'cl_dice':
                # clDice on dense affinity maps (B,8,H,W) is memory-prohibitive at
                # high resolutions; keep clDice on vote-map supervision and use
                # SmoothL1 for affinity/bicon regression to avoid OOM.
                affinity_l = F.smooth_l1_loss(c_map, affinity_target, reduction='mean')
            else:
                affinity_l = self.dist_aux_regression_loss(c_map, affinity_target)

            bicon_map = bicon_map.view(affinity_target.shape)   # (B,8,H,W)
            if self.args.dataset == 'chase':
                bicon_l = torch.zeros_like(vote_loss)
                loss = vote_loss + affinity_l + edge_l + dice_l
            else:
                if self.dist_aux_loss_name == 'cl_dice':
                    bicon_l = F.smooth_l1_loss(bicon_map, affinity_target, reduction='mean')
                else:
                    bicon_l = self.dist_aux_regression_loss(bicon_map, affinity_target)
                loss = vote_loss + affinity_l + edge_l + 0.2 * bicon_l + dice_l
            loss_dict = {
                'total': loss,
                'vote': vote_loss,
                'affinity': affinity_l,
                'edge': edge_l,
                'bicon': bicon_l,
                'dice': dice_l,
            }

        return loss, loss_dict
