from .betti_error import betti_error


def getBettiErrors(binaryPredict, masks):
    metrics = betti_error(binaryPredict, masks)
    return [metrics['betti_error_0']], [metrics['betti_error_1']]


def getBetti(binaryPredict, masks):
    betti0_error_ls, betti1_error_ls = getBettiErrors(
        binaryPredict, masks
    )
    return [betti0 + betti1 for betti0, betti1 in zip(betti0_error_ls, betti1_error_ls)]
