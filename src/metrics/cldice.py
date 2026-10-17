import numpy as np
from skimage.morphology import skeletonize


def cl_score(v, s):
    s_sum = np.sum(s)
    if s_sum == 0:
        return 0.0
    return np.sum(v * s) / s_sum


def clDice(v_p, v_l):
    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif len(v_p.shape) == 3:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    else:
        return 0.0
    denom = tprec + tsens
    if denom == 0:
        return 0.0
    return 2 * tprec * tsens / denom
