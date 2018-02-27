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
import keras.applications.densenet
import keras.applications.nasnet
import keras.backend as K
import keras.layers
import numpy as np
import warnings

from . import base
from . import mnist
from ...keras import graph as kgraph


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


###############################################################################
###############################################################################
###############################################################################


def _prepare_keras_net(clazz, input_shape, output_n,
                       preprocess_f,
                       color_coding="RGB", weights=None):
    model = clazz(weights=weights)
    net = {}
    net["in"] = model.inputs
    net["sm_out"] = model.outputs
    net["out"] = kgraph.pre_softmax_tensors(model.outputs)
    if K.image_data_format() == "channels_first":
        net["input_shape"] = [None, 3]+input_shape
    else:
        net["input_shape"] = [None]+input_shape+[3]
    net["color_coding"] = color_coding
    net["preprocess_f"] = preprocess_f
    net["output_n"] = output_n
    return net


###############################################################################
###############################################################################
###############################################################################


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

    net.update(base.conv_pool(
        net["in"], 2, "conv_1", 64,
        activation=activation,
    ))
    net.update(base.conv_pool(
        net["conv_1_pool"], 2, "conv_2", 128,
        activation=activation,
    ))
    net.update(base.conv_pool(
        net["conv_2_pool"], 3, "conv_3", 256,
        activation=activation,
    ))
    net.update(base.conv_pool(
        net["conv_3_pool"], 3, "conv_4", 512,
        activation=activation,
    ))
    net.update(base.conv_pool(
        net["conv_4_pool"], 3, "conv_5", 512,
        activation=activation,
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
        "preprocess_f": vgg16_custom_preprocess,
        "output_n": output_n,
    })
    return net


###############################################################################
###############################################################################
###############################################################################


def vgg16(weights=None):
    return _prepare_keras_net(
        keras.applications.vgg16.VGG16,
        [224, 224],
        1000,
        preprocess_f=keras.applications.vgg16.preprocess_input,
        color_coding="BGR",
        weights=weights)


def vgg19(weights=None):
    return _prepare_keras_net(
        keras.applications.vgg19.VGG19,
        [224, 224],
        1000,
        preprocess_f=keras.applications.vgg19.preprocess_input,
        color_coding="BGR",
        weights=weights)


###############################################################################
###############################################################################
###############################################################################


def resnet50(weights=None):
    return _prepare_keras_net(
        keras.applications.resnet50.ResNet50,
        [224, 224],
        1000,
        preprocess_f=keras.applications.resnet50.preprocess_input,
        color_coding="BGR",
        weights=weights)


###############################################################################
###############################################################################
###############################################################################


def inception_v3(weights=None):
    return _prepare_keras_net(
        keras.applications.inception_v3.InceptionV3,
        [299, 299],
        1000,
        preprocess_f=keras.applications.inception_v3.preprocess_input,
        weights=weights)


###############################################################################
###############################################################################
###############################################################################


def inception_resnet_v2(weights=None):
    return _prepare_keras_net(
        keras.applications.inception_resnet_v2.InceptionResNetV2,
        [299, 299],
        1000,
        preprocess_f=keras.applications.inception_resnet_v2.preprocess_input,
        weights=weights)


###############################################################################
###############################################################################
###############################################################################


def densenet121(weights=None):
    return _prepare_keras_net(
        keras.applications.densenet.DenseNet121,
        [224, 224],
        1000,
        preprocess_f=keras.applications.densenet.preprocess_input,
        weights=weights)


def densenet169(weights=None):
    return _prepare_keras_net(
        keras.applications.densenet.DenseNet169,
        [224, 224],
        1000,
        preprocess_f=keras.applications.densenet.preprocess_input,
        weights=weights)


def densenet201(weights=None):
    return _prepare_keras_net(
        keras.applications.densenet.DenseNet201,
        [224, 224],
        1000,
        preprocess_f=keras.applications.densenet.preprocess_input,
        weights=weights)


###############################################################################
###############################################################################
###############################################################################


def nasnet_large(weights=None):
    if K.image_data_format() == "channels_first":
        warnings.warn("NASNet is not available for channels first. "
                      "Return dummy net.")
        return mnist.log_reg()

    return _prepare_keras_net(
        keras.applications.nasnet.NASNetLarge,
        [331, 331],
        1000,
        color_coding="BGR",
        preprocess_f=keras.applications.nasnet.preprocess_input,
        weights=weights)


def nasnet_mobile(weights=None):
    if K.image_data_format() == "channels_first":
        warnings.warn("NASNet is not available for channels first. "
                      "Return dummy net.")
        return mnist.log_reg()

    return _prepare_keras_net(
        keras.applications.nasnet.NASNetMobile,
        [224, 224],
        1000,
        color_coding="BGR",
        preprocess_f=keras.applications.nasnet.preprocess_input,
        weights=weights)
