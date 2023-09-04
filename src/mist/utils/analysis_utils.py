""" Utility functions for analyzing data."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import partial
from mist import utils


def ll(pred, targ):
    """nll.

    Args:
        pred:
        targ:
    """
    print(pred.shape, targ.shape)
    log = partial(utils.clamped_log_np, _min=-5)
    ll = targ * log(pred) + (1 - targ) * log(1 - pred)
    return ll


def ll_bit(pred, targ):
    """nll.

    Args:
        pred:
        targ:
    """
    log = partial(utils.clamped_log_np, _min=-5)
    ll = targ * log(pred) + (1 - targ) * log(1 - pred)
    return ll.mean(0)


def ll_spec(pred, targ):
    """nll.

    Args:
        pred:
        targ:
    """
    log = partial(utils.clamped_log_np, _min=-5)
    ll = targ * log(pred) + (1 - targ) * log(1 - pred)
    return ll.mean(1)


def cos_sim(pred, targ):
    """nll.

    Args:
        pred:
        targ:
    """
    pred

    sim = cosine_similarity(pred, targ)
    sim = np.diag(sim)
    return sim


def tani(pred, targ, thresh=0.5):
    """tani.

    Args:
        pred:
        targ:
    """
    pred = np.copy(pred)
    above_thresh = pred >= thresh
    pred[above_thresh] = 1.0
    pred[~above_thresh] = 0.0

    pred, targ = pred.astype(bool), targ.astype(bool)
    denom = np.logical_or(pred, targ).sum(-1)
    num = np.logical_and(pred, targ).sum(-1)
    res = num / denom
    return res


bit_metrics = {
    "LL_bit": ll_bit,
}
spec_metrics = {
    "LL_spec": ll_spec,
    "Cosine": cos_sim,
    "Tani": tani,
}
