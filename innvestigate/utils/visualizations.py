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
    "preprocess_images",
    "postprocess_images",

    "project",
    "heatmap",
]


###############################################################################
###############################################################################
###############################################################################


def preprocess_images(images, color_coding=None):

    ret = images
    image_data_format = K.image_data_format()
    channels_first = images.shape[1] == 3
    if image_data_format == "channels_first" and not channels_first:
        ret = ret.transpose(0, 3, 1, 2)
    if image_data_format == "channels_last" and channels_first:
        ret = ret.transpose(0, 2, 3, 1)

    assert color_coding in [None, "RGBtoBGR", "BGRtoRGB"]
    if color_coding in ["RGBtoBGR", "BGRtoRGB"]:
        if image_data_format == "channels_first":
            ret = ret[:, ::-1, :, :]
        if image_data_format == "channels_last":
            ret = ret[:, :, :, ::-1]

    return ret


def postprocess_images(images, color_coding=None, channels_first=None):

    ret = images
    image_data_format = K.image_data_format()
    assert color_coding in [None, "RGBtoBGR", "BGRtoRGB"]
    if color_coding in ["RGBtoBGR", "BGRtoRGB"]:
        if image_data_format == "channels_first":
            ret = ret[:, ::-1, :, :]
        if image_data_format == "channels_last":
            ret = ret[:, :, :, ::-1]

    if image_data_format == "channels_first" and not channels_first:
        ret = ret.transpose(0, 2, 3, 1)
    if image_data_format == "channels_last" and channels_first:
        ret = ret.transpose(0, 3, 1, 2)

    return ret


###############################################################################
###############################################################################
###############################################################################


def project(X, output_range=(0, 1), absmax=None, input_is_postive_only=False):

    if absmax is None:
        absmax = np.max(np.abs(X),
                        axis=tuple(range(1, len(X.shape))))
    absmax = np.asarray(absmax)

    mask = absmax != 0
    X[mask] /= absmax[mask]

    if input_is_postive_only is False:
        X = (X+1)/2  # [0, 1]
    X = X.clip(0, 1)

    X = output_range[0] + (X * (output_range[1]-output_range[0]))
    return X


def heatmap(X, cmap_type="seismic", reduce_op="sum"):
    cmap = plt.cm.get_cmap(cmap_type)

    tmp = X
    shape = tmp.shape

    # has color channels?
    if shape[1] == 3 or (len(shape) >= 4 and shape[3] == 3):
        # reduce color channels
        color_axis = 1 if shape[1] == 3 else 3

        if reduce_op == "sum":
            tmp = tmp.sum(axis=color_axis)
        elif reduce_op == "absmax":
            pos_max = tmp.max(axis=color_axis)
            neg_max = (-tmp).max(axis=color_axis)
            abs_neg_max = -neg_max
            tmp = np.select([pos_max >= abs_neg_max, pos_max < abs_neg_max],
                            [pos_max, neg_max])
        else:
            raise NotImplementedError()

    tmp = project(tmp, output_range=(0, 255)).astype(np.int64)

    tmp = cmap(tmp.flatten())[:, :3]
    if shape[1] == 3:
        tmp = tmp.T

    return tmp.reshape(shape).astype(np.float32)


def clip_quantile(X, quantile=1):

    if not isinstance(quantile, (list, tuple)):
        quantile = (quantile, 100-quantile)

    low = np.percentile(X, quantile[0])
    high = np.percentile(X, quantile[1])
    X[X < low] = low
    X[X > high] = high

    return X
