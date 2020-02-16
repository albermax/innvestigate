# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import range


###############################################################################
###############################################################################
###############################################################################

import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    "project",
    "heatmap",
    "graymap",
    "gamma",
    "clip_quantile",
]


###############################################################################
###############################################################################
###############################################################################


def project(X, output_range=(0, 1), absmax=None, input_is_positive_only=False):
    """Projects a tensor into a value range.

    Projects the tensor values into the specified range.

    :param X: A tensor.
    :param output_range: The output value range.
    :param absmax: A tensor specifying the absmax used for normalizing.
      Default the absmax along the first axis.
    :param input_is_positive_only: Is the input value range only positive.
    :return: The tensor with the values project into output range.
    """

    if absmax is None:
        absmax = np.max(np.abs(X),
                        axis=tuple(range(1, len(X.shape))))
    absmax = np.asarray(absmax)

    mask = absmax != 0
    if mask.sum() > 0:
        X[mask] /= absmax[mask]

    if input_is_positive_only is False:
        X = (X+1)/2  # [0, 1]
    X = X.clip(0, 1)

    X = output_range[0] + (X * (output_range[1]-output_range[0]))
    return X


def heatmap(X, cmap_type="seismic", reduce_op="sum", reduce_axis=-1, alpha_cmap=False, **kwargs):
    """Creates a heatmap/color map.

    Create a heatmap or colormap out of the input tensor.

    :param X: A image tensor with 4 axes.
    :param cmap_type: The color map to use. Default 'seismic'.
    :param reduce_op: Operation to reduce the color axis.
      Either 'sum' or 'absmax'.
    :param reduce_axis: Axis to reduce.
    :param alpha_cmap: Should the alpha component of the cmap be included.
    :param kwargs: Arguments passed on to :func:`project`
    :return: The tensor as color-map.
    """
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

    if alpha_cmap:
        tmp = cmap(tmp.flatten()).T
    else:
        tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[reduce_axis] = 3 + alpha_cmap
    return tmp.reshape(shape).astype(np.float32)


def graymap(X, **kwargs):
    """Same as :func:`heatmap` but uses a gray colormap."""
    return heatmap(X, cmap_type="gray", **kwargs)


def gamma(X, gamma=0.5, minamp=0, maxamp=None):
    """
    Apply gamma correction to an input array X
    while maintaining the relative order of entries,
    also for negative vs positive values in X.
    the fxn firstly determines the max
    amplitude in both positive and negative
    direction and then applies gamma scaling
    to the positive and negative values of the
    array separately, according to the common amplitude.

    :param gamma: the gamma parameter for gamma scaling
    :param minamp: the smallest absolute value to consider.
        if not given assumed to be zero (neutral value for relevance,
        min value for saliency, ...). values above and below
        minamp are treated separately.
    :param maxamp: the largest absolute value to consider relative
        to the neutral value minamp
        if not given determined from the given data.
    """

    #prepare return array
    Y = np.zeros_like(X)

    X = X - minamp # shift to given/assumed center
    if maxamp is None: maxamp = np.abs(X).max() #infer maxamp if not given
    X = X / maxamp # scale linearly

    #apply gamma correction for both positive and negative values.
    i_pos = X > 0
    i_neg = np.invert(i_pos)
    Y[i_pos] = X[i_pos]**gamma
    Y[i_neg] = -(-X[i_neg])**gamma

    #reconstruct original scale and center
    Y *= maxamp
    Y += minamp

    return Y


def clip_quantile(X, quantile=1):
    """Clip the values of X into the given quantile."""
    if not isinstance(quantile, (list, tuple)):
        quantile = (quantile, 100-quantile)

    low = np.percentile(X, quantile[0])
    high = np.percentile(X, quantile[1])
    X[X < low] = low
    X[X > high] = high

    return X
