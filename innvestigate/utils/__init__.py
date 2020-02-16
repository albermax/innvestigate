# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import tensorflow.keras.backend as K
import tensorflow.keras.utils as keras_utils
import math


__all__ = [
    "model_wo_softmax",
    "to_list",

    "BatchSequence",
    "TargetAugmentedSequence",

    "preprocess_images",
    "postprocess_images",
]


###############################################################################
###############################################################################
###############################################################################


def model_wo_softmax(*args, **kwargs):
    # Break cyclic import
    from .keras.graph import model_wo_softmax

    return model_wo_softmax(*args, **kwargs)


###############################################################################
###############################################################################
###############################################################################


def to_list(l):
    """ If not list, wraps parameter into a list."""
    if not isinstance(l, list):
        return [l, ]
    else:
        return l


###############################################################################
###############################################################################
###############################################################################


class BatchSequence(keras_utils.Sequence):
    """Batch sequence generator.

    Take a (list of) input tensors and a batch size
    and creates a generators that creates a sequence of batches.

    :param Xs: One or a list of tensors. First axis needs to have same length.
    :param batch_size: Batch size. Default 32.
    """

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


class TargetAugmentedSequence(keras_utils.Sequence):
    """Augments a sequence with a target on the fly.

    Takes a sequence/generator and a function that
    creates on the fly for each batch a target.
    The generator takes a batch from that sequence,
    computes the target and returns both.

    :param sequence: A sequence or generator.
    :param augment_f: Takes a batch and returns a target.
    """

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
    """Image preprocessing

    Takes a batch of images and:
    * Adjust the color axis to the Keras format.
    * Fixes the color coding.

    :param images: Batch of images with 4 axes.
    :param color_coding: Determines the color coding.
      Can be None, 'RGBtoBGR' or 'BGRtoRGB'.
    :return: The preprocessed batch.
    """

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
    """Image postprocessing

    Takes a batch of images and reverts the preprocessing.

    :param images: A batch of images with 4 axes.
    :param color_coding: The initial color coding,
      see :func:`preprocess_images`.
    :param channels_first: The output channel format.
    :return: The postprocessed images.
    """

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
