# Get Python six functionality:
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings

import keras.backend as K
import keras.layers
import numpy as np

from innvestigate.applications import imagenet

from tests.networks import base, mnist

__all__ = [
    "vgg16_custom",
    "vgg16",
    "vgg19",
    "resnet50",
    "inception_v3",
    "inception_resnet_v2",
    "densenet121",
    "densenet169",
    "densenet201",
    "nasnet_large",
    "nasnet_mobile",
]


VGG16_OFFSET = np.array([103.939, 116.779, 123.68])


def vgg16_custom_preprocess(X):
    import innvestigate.utils.visualizations as ivis

    X = ivis.preprocess_images(X, color_coding="RGBtoBGR")

    if X.shape[1] == 3:
        shape = [1, 3, 1, 1]
    else:
        shape = [1, 1, 1, 3]

    offset = VGG16_OFFSET.reshape(shape)
    # Remove pixel-wise mean.
    X -= offset
    return X


def vgg16_custom(activation=None):
    if activation is None:
        activation = "relu"

    if K.image_data_format() == "channels_first":
        input_shape = [None, 3, 224, 224]
    else:
        input_shape = [None, 224, 224, 3]
    output_n = 1000

    net = {}
    net["in"] = base.input_layer(shape=input_shape)

    net.update(
        base.conv_pool(
            net["in"],
            2,
            "conv_1",
            64,
            activation=activation,
        )
    )
    net.update(
        base.conv_pool(
            net["conv_1_pool"],
            2,
            "conv_2",
            128,
            activation=activation,
        )
    )
    net.update(
        base.conv_pool(
            net["conv_2_pool"],
            3,
            "conv_3",
            256,
            activation=activation,
        )
    )
    net.update(
        base.conv_pool(
            net["conv_3_pool"],
            3,
            "conv_4",
            512,
            activation=activation,
        )
    )
    net.update(
        base.conv_pool(
            net["conv_4_pool"],
            3,
            "conv_5",
            512,
            activation=activation,
        )
    )

    net["conv_flat"] = keras.layers.Flatten()(net["conv_5_pool"])
    net["dense_1"] = base.dense_layer(
        net["conv_flat"],
        units=4096,
        activation=activation,
        kernel_initializer="glorot_uniform",
    )
    net["dense_1_dropout"] = base.dropout_layer(net["dense_1"], 0.5)
    net["dense_2"] = base.dense_layer(
        net["dense_1_dropout"],
        units=4096,
        activation=activation,
        kernel_initializer="glorot_uniform",
    )
    net["dense_2_dropout"] = base.dropout_layer(net["dense_2"], 0.5)
    net["out"] = base.dense_layer(
        net["dense_2_dropout"], units=output_n, kernel_initializer="glorot_uniform"
    )
    net["sm_out"] = base.softmax(net["out"])

    net.update(
        {
            "input_shape": input_shape,
            "preprocess_f": vgg16_custom_preprocess,
            "output_n": output_n,
        }
    )
    return net


###############################################################################
###############################################################################
###############################################################################


def vgg16():
    ret = imagenet.vgg16()
    ret["output_n"] = 1000
    return ret


def vgg19():
    ret = imagenet.vgg19()
    ret["output_n"] = 1000
    return ret


###############################################################################
###############################################################################
###############################################################################


def resnet50():
    ret = imagenet.resnet50()
    ret["output_n"] = 1000
    return ret


###############################################################################
###############################################################################
###############################################################################


def inception_v3():
    ret = imagenet.inception_v3()
    ret["output_n"] = 1000
    return ret


###############################################################################
###############################################################################
###############################################################################


def inception_resnet_v2():
    ret = imagenet.inception_resnet_v2()
    ret["output_n"] = 1000
    return ret


###############################################################################
###############################################################################
###############################################################################


def densenet121():
    ret = imagenet.densenet121()
    ret["output_n"] = 1000
    return ret


def densenet169():
    ret = imagenet.densenet169()
    ret["output_n"] = 1000
    return ret


def densenet201():
    ret = imagenet.densenet201()
    ret["output_n"] = 1000
    return ret


###############################################################################
###############################################################################
###############################################################################


def nasnet_large():
    if K.image_data_format() == "channels_first":
        warnings.warn(
            "NASNet is not available for channels first. " "Return dummy net."
        )
        return mnist.log_reg()

    ret = imagenet.nasnet_large()
    ret["output_n"] = 1000
    return ret


def nasnet_mobile():
    if K.image_data_format() == "channels_first":
        warnings.warn(
            "NASNet is not available for channels first. " "Return dummy net."
        )
        return mnist.log_reg()

    ret = imagenet.nasnet_mobile()
    ret["output_n"] = 1000
    return ret
