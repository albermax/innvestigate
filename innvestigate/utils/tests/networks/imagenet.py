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


import keras.applications.resnet50
import keras.applications.vgg16
import keras.applications.vgg19
import keras.applications.inception_v3
import keras.applications.inception_resnet_v2
import keras.layers
import numpy as np

from . import base
from ...keras import graph as kgraph

__all__ = [
    "vgg16",
    "vgg19",
    "resnet50",
    "inception_v3",
    "inception_resnet_v2",
]


###############################################################################
###############################################################################
###############################################################################


def _prepare_keras_net(clazz, input_shape, output_n, weights=None):
    model = clazz(weights=None)
    net = {}
    net["in"] = model.inputs
    net["sm_out"] = model.outputs
    net["out"] = kgraph.pre_softmax_tensors(model.outputs)
    net["input_shape"] = input_shape
    net["output_n"] = output_n
    return net


###############################################################################
###############################################################################
###############################################################################


VGG16_OFFSET = np.array([103.939, 116.779, 123.68])


def vgg16_preprocess(X):
    if X.shape[1] == 3:
        shape = [1, 3, 1, 1]
    else:
        shape = [1, 1, 1, 3]

    offset = VGG16_OFFSET.reshape(shape)
    # Remove pixel-wise mean.
    X -= offset
    return X


def vgg16_preprocess(X):
    return keras.applications.vgg16.preprocess_input(X)


def vgg16(activation=None, weights=None):
    return _prepare_keras_net(
        keras.applications.vgg16.VGG16,
        [None, 3, 224, 224],
        1000,
        weights=weights)


###############################################################################
###############################################################################
###############################################################################


def vgg19_preprocess(X):
    return keras.applications.vgg19.preprocess_input(X)


def vgg19(activation=None, weights=None):
    return _prepare_keras_net(
        keras.applications.vgg19.VGG19,
        [None, 3, 224, 224],
        1000,
        weights=weights)


###############################################################################
###############################################################################
###############################################################################


def resnet50_preprocess(X):
    return keras.applications.resnet50.preprocess_input(X)


def resnet50(activation=None, weights=None):
    return _prepare_keras_net(
        keras.applications.resnet50.ResNet50,
        [None, 3, 224, 224],
        1000,
        weights=weights)


###############################################################################
###############################################################################
###############################################################################


def inception_v3_preprocess(X):
    return keras.applications.inception_v3.preprocess_input(X)


def inception_v3(activation=None, weights=None):
    return _prepare_keras_net(
        keras.applications.inception_v3.InceptionV3,
        [None, 3, 299, 299],
        1000,
        weights=weights)


###############################################################################
###############################################################################
###############################################################################


def inception_resnet_v2_preprocess(X):
    return keras.applications.inception_resnet_v2.preprocess_input(X)


def inception_resnet_v2(activation=None, weights=None):
    return _prepare_keras_net(
        keras.applications.inception_resnet_v2.InceptionResNetV2,
        [None, 3, 299, 299],
        1000,
        weights=weights)
