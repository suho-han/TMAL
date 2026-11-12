import torch
import torch.nn.functional as F


class SimpleLoss:
    """BCE + soft-Dice loss without connectivity supervision.

    Same calling interface as connect_loss so the solver can use it
    interchangeably when --simple_loss is set.
    """

    def __init__(self, label_mode: str = 'binary'):
        self.label_mode = label_mode

    # no-op stubs — solver calls these on loss_func unconditionally
    def reset_dist_edge_stats(self): pass
    def set_dist_edge_stat_collection(self, flag: bool): pass
    def get_dist_edge_stats(self): return None

    def __call__(self, pred, target, return_details=False):
        # pred:   (B, 1, H, W) logits
        # target: (B, 1, H, W) binary mask or distance map
        gt = (target > 0.5).float()
        bce = F.binary_cross_entropy_with_logits(pred, gt)
        p = torch.sigmoid(pred)
        inter = (p * gt).sum()
        dice = 1.0 - (2.0 * inter + 1.0) / (p.sum() + gt.sum() + 1.0)
        loss = bce + dice
        if return_details:
            return loss, {'bce': bce.detach(), 'dice': dice.detach()}
        return loss
