import torch
import torch.nn.functional as F


def gjml_loss(pred, target, eps=1e-8):
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    sum_norm = torch.sum(torch.abs(pred_flat + target_flat), dim=1)
    diff_norm = torch.sum(torch.abs(pred_flat - target_flat), dim=1)
    jaccard_term = (sum_norm - diff_norm) / (sum_norm + diff_norm + eps)
    return (1 - jaccard_term).mean()


def stable_focal_l1_loss(pred, target, gamma):
    diff = torch.abs(target - pred)
    indicator = (target * pred >= 0).float()
    return (diff * torch.pow(diff, gamma) * indicator).mean()


def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, num_iter=10):
    img_open = soft_open(img)
    skel = F.relu(img - img_open)
    for _ in range(num_iter):
        img = soft_erode(img)
        img_open = soft_open(img)
        delta = F.relu(img - img_open)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(pred, target, smooth=1.0, num_iter=10):
    # Keep clDice computation on the original device and align the formula with
    # the upstream jocpae/clDice implementation.
    pred = pred.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)

    skel_pred = soft_skeletonize(pred, num_iter=num_iter)
    skel_target = soft_skeletonize(target, num_iter=num_iter)

    tprec = (torch.sum(skel_pred * target) + smooth) / (torch.sum(skel_pred) + smooth)
    tsens = (torch.sum(skel_target * pred) + smooth) / (torch.sum(skel_target) + smooth)
    return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)


def dist_aux_regression_loss(pred, target, loss_name, gamma=1.0):
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction="mean")
    if loss_name == "gjml_sf_l1":
        return gjml_loss(pred, target) + stable_focal_l1_loss(pred, target, gamma=gamma)
    if loss_name == "cl_dice":
        return soft_cldice_loss(pred, target)
    raise ValueError(
        f"Unsupported dist_aux_loss {loss_name}, expected 'smooth_l1', 'gjml_sf_l1', or 'cl_dice'"
    )
