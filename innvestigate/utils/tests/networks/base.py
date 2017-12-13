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


__all__ = [
    "log_reg",

    "mlp_1dense",
    "mlp_2dense",

    "cnn_1convb_1dense",
    "cnn_2convb_1dense",
    "cnn_2convb_2dense",
    "cnn_3convb_2dense",
]


###############################################################################
###############################################################################
###############################################################################


def input_layer(shape, *args, **kwargs):
    return keras.layers.Input(shape=shape[1:], *args, **kwargs)


def dense_layer(layer_in, *args, **kwargs):
    return keras.layers.Dense(*args, **kwargs)(layer_in)


def conv_layer(layer_in, *args, **kwargs):
    return keras.layers.Conv2D(*args, **kwargs)(layer_in)


def conv_pool(layer_in, n_conv, prefix, n_filter, **kwargs):
    conv_prefix = "%s_%%i" % prefix

    ret = {}
    current_layer = layer_in
    for i in range(n_conv):
        conv = conv_layer(current_layer, filters=n_filter,
                          kernel_size=(3, 3), strides=(1, 1), padding="same", 
                          kernel_initializer="glorot_uniform", **kwargs)
        current_layer = conv
        ret[conv_prefix % i] = conv

        ret["%s_pool" % prefix] = keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        )(current_layer)
    return ret


def dropout_layer(layer_in, *args, **kwargs):
    return keras.layers.Dropout(*args, **kwargs)(layer_in)


###############################################################################
###############################################################################
###############################################################################


def log_reg(input_shape, output_n, activation=None):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net["out"] = dense_layer(net["in"], units=output_n,
                             activation="softmax",
                             kernel_initializer="glorot_uniform")

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


###############################################################################
###############################################################################
###############################################################################


def mlp_1dense(input_shape, output_n, activation=None,
               dense_units=512, dropout_rate=0.25):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net["dense_1"] = dense_layer(net["in"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], dropout_rate)
    net["out"] = dense_layer(net["dense_1_dropout"], units=output_n,
                             activation="softmax",
                             kernel_initializer="glorot_uniform")

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


def mlp_2dense(input_shape, output_n, activation=None,
               dense_units=512, dropout_rate=0.25):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net["dense_1"] = dense_layer(net["in"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], dropout_rate)
    net["dense_2"] = dense_layer(net["dense_1_dropout"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net['dense_2_dropout'] = dropout_layer(net['dense_2'], dropout_rate)
    net["out"] = dense_layer(net["dense_2_dropout"], units=output_n,
                             activation="softmax",
                             kernel_initializer="glorot_uniform")

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


###############################################################################
###############################################################################
###############################################################################


def cnn_1convb_1dense(input_shape, output_n, activation=None,
                      dense_units=512, dropout_rate=0.25):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net.update(conv_pool(net["in"], 2, "conv_1", 128,
                         activation=activation))
    net["dense_1"] = dense_layer(net["conv_1_pool"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], dropout_rate)
    net["out"] = dense_layer(net["dense_1_dropout"], units=output_n,
                             activation="softmax",
                             kernel_initializer="glorot_uniform")

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


def cnn_2convb_1dense(input_shape, output_n, activation=None,
                      dense_units=512, dropout_rate=0.25):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net.update(conv_pool(net["in"], 2, "conv_1", 128,
                         activation=activation))
    net.update(conv_pool(net["conv_1_pool"], 2, "conv_2", 128,
                         activation=activation))
    net["dense_1"] = dense_layer(net["conv_2_pool"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], dropout_rate)
    net["out"] = dense_layer(net["dense_1_dropout"], units=output_n,
                             activation="softmax",
                             kernel_initializer="glorot_uniform")

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


def cnn_2convb_2dense(input_shape, output_n, activation=None,
                      dense_units=512, dropout_rate=0.25):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net.update(conv_pool(net["in"], 2, "conv_1", 128,
                         activation=activation))
    net.update(conv_pool(net["conv_1_pool"], 2, "conv_2", 128,
                         activation=activation))
    net["dense_1"] = dense_layer(net["conv_2_pool"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], dropout_rate)
    net["dense_2"] = dense_layer(net["dense_1_dropout"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net['dense_2_dropout'] = dropout_layer(net['dense_2'], dropout_rate)
    net["out"] = dense_layer(net["dense_2_dropout"], units=output_n,
                             activation="softmax",
                             kernel_initializer="glorot_uniform")

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


def cnn_3convb_2dense(input_shape, output_n, activation=None,
                      dense_units=512, dropout_rate=0.25):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net.update(conv_pool(net["in"], 2, "conv_1", 128,
                         activation=activation))
    net.update(conv_pool(net["conv_1_pool"], 2, "conv_2", 128,
                         activation=activation))
    net.update(conv_pool(net["conv_2_pool"], 2, "conv_3", 128,
                         activation=activation))
    net["dense_1"] = dense_layer(net["conv_3_pool"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], dropout_rate)
    net["dense_2"] = dense_layer(net["dense_1_dropout"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net['dense_2_dropout'] = dropout_layer(net['dense_2'], dropout_rate)
    net["out"] = dense_layer(net["dense_2_dropout"], units=output_n,
                             activation="softmax",
                             kernel_initializer="glorot_uniform")

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


def cnn_3convb_3dense(input_shape, output_n, activation=None):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net.update(conv_pool(net["in"], 2, "conv_1", 64,
                         activation=activation))
    net.update(conv_pool(net["conv_1_pool"], 2, "conv_2", 128,
                         activation=activation))
    net.update(conv_pool(net["conv_2_pool"], 3, "conv_3", 256,
                         activation=activation))
    net['conv_3_dropout'] = dropout_layer(net['conv_3_pool'], 0.25)

    net["dense_1"] = dense_layer(net["conv_3_dropout"], units=1024,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], 0.5)

    net["dense_2"] = dense_layer(net["dense_1_dropout"], units=1024,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net['dense_2_dropout'] = dropout_layer(net['dense_2'], 0.5)
    net["out"] = dense_layer(net["dense_2_dropout"], units=output_n,
                             activation="softmax",
                             kernel_initializer="glorot_uniform")

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net
