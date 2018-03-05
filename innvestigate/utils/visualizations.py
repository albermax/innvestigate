# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    "project",
    "heatmap",
    "graymap",
]


###############################################################################
###############################################################################
###############################################################################


def project(X, output_range=(0, 1), absmax=None, input_is_postive_only=False):

    if absmax is None:
        absmax = np.max(np.abs(X),
                        axis=tuple(range(1, len(X.shape))))
    absmax = np.asarray(absmax)

    mask = absmax != 0
    if mask.sum() > 0:
        X[mask] /= absmax[mask]

    if input_is_postive_only is False:
        X = (X+1)/2  # [0, 1]
    X = X.clip(0, 1)

    X = output_range[0] + (X * (output_range[1]-output_range[0]))
    return X


def heatmap(X, cmap_type="seismic", reduce_op="sum", reduce_axis=-1, **kwargs):
    cmap = plt.cm.get_cmap(cmap_type)

    tmp = X
    shape = tmp.shape

    if reduce_op == "sum":
        tmp = tmp.sum(axis=reduce_axis)
    elif reduce_op == "absmax":
        pos_max = tmp.max(axis=reduce_axis)
        neg_max = (-tmp).max(axis=reduce_axis)
        abs_neg_max = -neg_max
        tmp = np.select([pos_max >= abs_neg_max, pos_max < abs_neg_max],
                        [pos_max, neg_max])
    else:
        raise NotImplementedError()

    tmp = project(tmp, output_range=(0, 255), **kwargs).astype(np.int64)

    tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[reduce_axis] = 3
    return tmp.reshape(shape).astype(np.float32)


def graymap(X, **kwargs):
    return heatmap(X, cmap_type="gray", **kwargs)


def clip_quantile(X, quantile=1):

    if not isinstance(quantile, (list, tuple)):
        quantile = (quantile, 100-quantile)

    low = np.percentile(X, quantile[0])
    high = np.percentile(X, quantile[1])
    X[X < low] = low
    X[X > high] = high

    return X


def batch_flatten(x):
    # Flattens all but the first dimensions of a numpy array, i.e. flatten each sample in a batch
    if not isinstance(x, np.ndarray):
        raise TypeError("Only applicable to Numpy arrays.")
    return x.reshape(x.shape[0], -1)
