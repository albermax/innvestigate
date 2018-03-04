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
import keras.utils
import math


__all__ = [
    "to_list",

    "BatchSequence",
    "TargetAugmentedSequence",

    "preprocess_images",
    "postprocess_images",
]


###############################################################################
###############################################################################
###############################################################################


def to_list(l):
    if not isinstance(l, list):
        return [l, ]
    else:
        return l


###############################################################################
###############################################################################
###############################################################################


class BatchSequence(keras.utils.Sequence):

    def __init__(self, Xs, batch_size=32):
        self.Xs = to_list(Xs)
        self.single_tensor = len(Xs) == 1
        self.batch_size = batch_size

        if not self.single_tensor:
            for X in self.Xs[1:]:
                assert X.shape[0] == self.Xs[0].shape[0]
        super(BatchSequence, self).__init__()

    def __len__(self):
        return int(math.ceil(float(len(self.Xs[0])) / self.batch_size))

    def __getitem__(self, idx):
        ret = [X[idx*self.batch_size:(idx+1)*self.batch_size]
               for X in self.Xs]

        if self.single_tensor:
            return ret[0]
        else:
            return tuple(ret)


class TargetAugmentedSequence(keras.utils.Sequence):

    def __init__(self, sequence, augment_f):
        self.sequence = sequence
        self.augment_f = augment_f

        super(TargetAugmentedSequence, self).__init__()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        inputs = self.sequence[idx]
        if isinstance(inputs, tuple):
            assert len(inputs) == 1
            inputs = inputs[0]

        targets = self.augment_f(to_list(inputs))
        return inputs, targets


###############################################################################
###############################################################################
###############################################################################


def preprocess_images(images, color_coding=None):

    ret = images
    image_data_format = K.image_data_format()
    # todo: not very general:
    channels_first = images.shape[1] in [1, 3]
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
