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


import keras.layers
import numpy as np

from . import base


__all__ = [
    "vgg16",
    #"vgg16_all_conv",

    #"caffenet",
    #"googlenet",
]


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


def vgg16_invert_preprocess(X):
    if X.shape[1] == 3:
        shape = [1, 3, 1, 1]
    else:
        shape = [1, 1, 1, 3]

    offset = VGG16_OFFSET.reshape(shape)
    # Add pixel-wise mean.
    X += offset
    return X


def vgg16(activation=None):
    if activation is None:
        activation = "relu"

    input_shape = [None, 3, 224, 224]
    output_n = 1000

    net = {}
    net["in"] = base.input_layer(shape=input_shape)

    net.update(base.conv_pool(
        net["in"], 2, "conv_1", 64,
        activation=activation,
        # todo: take care of theano to keras port:
        # flip_filters=False))
    ))
    net.update(base.conv_pool(
        net["conv_1_pool"], 2, "conv_2", 128,
        activation=activation,
        # todo: take care of theano to keras port:
        # flip_filters=False))
    ))
    net.update(base.conv_pool(
        net["conv_2_pool"], 3, "conv_3", 256,
        activation=activation,
        # todo: take care of theano to keras port:
        # flip_filters=False))
    ))
    net.update(base.conv_pool(
        net["conv_3_pool"], 3, "conv_4", 512,
        activation=activation,
        # todo: take care of theano to keras port:
        # flip_filters=False))
    ))
    net.update(base.conv_pool(
        net["conv_4_pool"], 3, "conv_5", 512,
        activation=activation,
        # todo: take care of theano to keras port:
        # flip_filters=False))
    ))

    net["conv_flat"] = keras.layers.Flatten()(net["conv_5_pool"])
    net["dense_1"] = base.dense_layer(net["conv_flat"], units=4096,
                                      activation=activation,
                                      kernel_initializer="glorot_uniform")
    net["dense_1_dropout"] = base.dropout_layer(net["dense_1"], 0.5)
    net["dense_2"] = base.dense_layer(net["dense_1_dropout"], units=4096,
                                      activation=activation,
                                      kernel_initializer="glorot_uniform")
    net["dense_2_dropout"] = base.dropout_layer(net["dense_2"], 0.5)
    net["out"] = base.dense_layer(net["dense_2_dropout"], units=output_n,
                                  kernel_initializer="glorot_uniform")
    net["sm_out"] = base.softmax(net["out"])

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


def vgg16_all_conv(activation=None):
    if activation is None:
        activation = "relu"

    input_shape = [None, 3, 256, 256]
    output_n = 1000
    
    net = {}
    net["in"] = base.input_layer(shape=input_shape)

    net.update(base.conv_pool(
        net["in"], 2, "conv_1", 64,
        activation=activation,
        # todo: take care of theano to keras port:
        # flip_filters=False))
    ))
    net.update(base.conv_pool(
        net["conv_1_pool"], 2, "conv_2", 128,
        activation=activation,
        # todo: take care of theano to keras port:
        # flip_filters=False))
    ))
    net.update(base.conv_pool(
        net["conv_2_pool"], 3, "conv_3", 256,
        activation=activation,
        # todo: take care of theano to keras port:
        # flip_filters=False))
    ))
    net.update(base.conv_pool(
        net["conv_3_pool"], 3, "conv_4", 512,
        activation=activation,
        # todo: take care of theano to keras port:
        # flip_filters=False))
    ))
    net.update(base.conv_pool(
        net["conv_4_pool"], 3, "conv_5", 512,
        activation=activation,
        # todo: take care of theano to keras port:
        # flip_filters=False))
    ))

    net["dense_1"] = base.conv_layer(net["conv_5_pool"],
                                     filters=4096, kernel_size=7,
                                     padding="valid",
                                     # todo: take care of theano to keras port:
                                     # flip_filters=False,
                                     activation=activation,
                                     kernel_initializer="glorot_uniform")
    net["dense_1_dropout"] = base.dropout_layer(net["dense_1"], 0.5)
    net["dense_2"] = base.conv_layer(net["dense_1_dropout"],
                                     filters=4096, kernel_size=1,
                                     padding="valid",
                                     # todo: take care of theano to keras port:
                                     # flip_filters=False,
                                     activation=activation,
                                     kernel_initializer="glorot_uniform")
    net["dense_2_dropout"] = base.dropout_layer(net["dense_2"], 0.5)
    net["dense_3"] = base.conv_layer(net["dense_2_dropout"], output_n, 1,
                                     padding="valid",
                                     # todo: take care of theano to keras port:
                                     # flip_filters=False,
                                     activation=None,
                                     kernel_initializer="glorot_uniform")
    # todo: fix:
    net["global_pool"] = lasagne.layers.GlobalPool(
        net["dense_3"], pool_function=theano.tensor.sum)
    net["out"] = lasagne.layers.Activation(
        net["global_pool"], activation="softmax")

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net
