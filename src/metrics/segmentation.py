import torch
import torch.nn.functional as F


def compute_binary_precision_accuracy(pred_mask, gt_mask, eps=1e-6):
    pred_bin = (pred_mask > 0.5).float()
    gt_bin = (gt_mask > 0.5).float()

    tp = torch.sum(pred_bin * gt_bin, dim=(1, 2, 3))
    fp = torch.sum(pred_bin * (1.0 - gt_bin), dim=(1, 2, 3))
    tn = torch.sum((1.0 - pred_bin) * (1.0 - gt_bin), dim=(1, 2, 3))
    fn = torch.sum((1.0 - pred_bin) * gt_bin, dim=(1, 2, 3))

    precision = (tp + eps) / (tp + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    return precision, accuracy


def compute_multiclass_precision_accuracy(pred_label, true_label, num_class, eps=1e-6):
    batch_size = pred_label.shape[0]
    precision_vals = []
    accuracy_vals = []

    class_ids = list(range(1, num_class))
    if len(class_ids) == 0:
        class_ids = list(range(num_class))

    for b_idx in range(batch_size):
        pred_b = pred_label[b_idx]
        true_b = true_label[b_idx]

        class_precisions = []
        for class_id in class_ids:
            pred_pos = pred_b == class_id
            true_pos = true_b == class_id
            tp = torch.sum(pred_pos & true_pos).float()
            fp = torch.sum(pred_pos & (~true_pos)).float()
            class_precisions.append((tp + eps) / (tp + fp + eps))

        precision_vals.append(torch.mean(torch.stack(class_precisions)))
        accuracy_vals.append(torch.mean((pred_b == true_b).float()))

    return torch.stack(precision_vals), torch.stack(accuracy_vals)


def per_class_dice(y_pred, y_true):
    eps = 0.0001

    fn = torch.sum((1 - y_pred) * y_true, dim=(2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(2, 3))
    pred = y_pred
    gt = y_true
    inter = torch.sum(gt * pred, dim=(2, 3))

    union = torch.sum(gt, dim=(2, 3)) + torch.sum(pred, dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    jac = (inter + eps) / (inter + fp + fn + eps)

    return dice, jac


def one_hot(target, shape, num_class):
    one_hot_mat = torch.zeros([shape[0], num_class, shape[2], shape[3]]).cuda()
    target = target.cuda()
    one_hot_mat.scatter_(1, target, 1)
    return one_hot_mat


def get_mask(output):
    output = F.softmax(output, dim=1)
    _, pred = output.topk(1, dim=1)
    return pred
