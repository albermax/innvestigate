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
import keras.utils.data_utils

from ..utils.keras import graph as kgraph


__all__ = [
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


PATTERN_BASE_URL="to be set"


def _prepare_keras_net(clazz, image_shape,
                       preprocess_f,
                       color_coding="RGB",
                       load_weights=False,
                       load_patterns=False):
    weights = None
    if load_weights is True:
        weights = "imagenet"

    model = clazz(weights=weights)

    net = {}
    net["model"] = model
    net["in"] = model.inputs
    net["sm_out"] = model.outputs
    net["out"] = kgraph.pre_softmax_tensors(model.outputs)

    net["image_shape"] = image_shape
    if K.image_data_format() == "channels_first":
        net["input_shape"] = [None, 3]+image_shape
    else:
        net["input_shape"] = [None]+image_shape+[3]
    net["color_coding"] = color_coding
    net["preprocess_f"] = preprocess_f

    net["patterns"] = None
    if load_patterns is not False:
        weights_path = keras.utils.data_utils.get_file(
            load_patterns["file_name"],
            PATTERN_BASE_URL % load_patterns["file_name"],
            cache_subdir="innvestigate_patterns",
            file_hash=load_patterns["hash"])

    return net


###############################################################################
###############################################################################
###############################################################################


def vgg16(load_weights=False, load_patterns=False):
    if load_patterns is True:
        load_patterns = {
            "file_name": "vgg16_pattern_type_relu_tf_dim_ordering_tf_kernels.npz",
            "hash": "",
        }

    return _prepare_keras_net(
        keras.applications.vgg16.VGG16,
        [224, 224],
        preprocess_f=keras.applications.vgg16.preprocess_input,
        color_coding="BGR",
        load_weights=load_weights,
        load_patterns=load_patterns)


def vgg19(load_weights=False, load_patterns=False):
    if load_patterns is True:
        load_patterns = {
            "file_name": "vgg19_pattern_type_relu_tf_dim_ordering_tf_kernels.npz",
            "hash": "",
        }

    return _prepare_keras_net(
        keras.applications.vgg19.VGG19,
        [224, 224],
        preprocess_f=keras.applications.vgg19.preprocess_input,
        color_coding="BGR",
        load_weights=load_weights,
        load_patterns=load_patterns)


###############################################################################
###############################################################################
###############################################################################


def resnet50(load_weights=False, load_patterns=False):
    if load_patterns is True:
        load_patterns = {
            "file_name": "resnet50_pattern_type_relu_tf_dim_ordering_tf_kernels.npz",
            "hash": "",
        }

    return _prepare_keras_net(
        keras.applications.resnet50.ResNet50,
        [224, 224],
        preprocess_f=keras.applications.resnet50.preprocess_input,
        color_coding="BGR",
        load_weights=load_weights,
        load_patterns=load_patterns)


###############################################################################
###############################################################################
###############################################################################


def inception_v3(load_weights=False, load_patterns=False):
    if load_patterns is True:
        load_patterns = {
            "file_name": "inception_v3_pattern_type_relu_tf_dim_ordering_tf_kernels.npz",
            "hash": "",
        }

    return _prepare_keras_net(
        keras.applications.inception_v3.InceptionV3,
        [299, 299],
        preprocess_f=keras.applications.inception_v3.preprocess_input,
        load_weights=load_weights,
        load_patterns=load_patterns)


###############################################################################
###############################################################################
###############################################################################


def inception_resnet_v2(load_weights=False, load_patterns=False):
    if load_patterns is True:
        load_patterns = {
            "file_name": "inception_resnet_v2_pattern_type_relu_tf_dim_ordering_tf_kernels.npz",
            "hash": "",
        }

    return _prepare_keras_net(
        keras.applications.inception_resnet_v2.InceptionResNetV2,
        [299, 299],
        preprocess_f=keras.applications.inception_resnet_v2.preprocess_input,
        load_weights=load_weights,
        load_patterns=load_patterns)


###############################################################################
###############################################################################
###############################################################################


def densenet121(load_weights=False, load_patterns=False):
    if load_patterns is True:
        load_patterns = {
            "file_name": "densenet_121_pattern_type_relu_tf_dim_ordering_tf_kernels.npz",
            "hash": "",
        }

    return _prepare_keras_net(
        keras.applications.densenet.DenseNet121,
        [224, 224],
        preprocess_f=keras.applications.densenet.preprocess_input,
        load_weights=load_weights,
        load_patterns=load_patterns)


def densenet169(load_weights=False, load_patterns=False):
    if load_patterns is True:
        load_patterns = {
            "file_name": "densenet169_pattern_type_relu_tf_dim_ordering_tf_kernels.npz",
            "hash": "",
        }

    return _prepare_keras_net(
        keras.applications.densenet.DenseNet169,
        [224, 224],
        preprocess_f=keras.applications.densenet.preprocess_input,
        load_weights=load_weights,
        load_patterns=load_patterns)


def densenet201(load_weights=False, load_patterns=False):
    if load_patterns is True:
        load_patterns = {
            "file_name": "densene201_pattern_type_relu_tf_dim_ordering_tf_kernels.npz",
            "hash": "",
        }

    return _prepare_keras_net(
        keras.applications.densenet.DenseNet201,
        [224, 224],
        preprocess_f=keras.applications.densenet.preprocess_input,
        load_weights=load_weights,
        load_patterns=load_patterns)


###############################################################################
###############################################################################
###############################################################################


def nasnet_large(load_weights=False, load_patterns=False):
    if load_patterns is True:
        load_patterns = {
            "file_name": "nasnet_large_pattern_type_relu_tf_dim_ordering_tf_kernels.npz",
            "hash": "",
        }

    if K.image_data_format() == "channels_first":
        raise Exception("NASNet is not available for channels first.")

    return _prepare_keras_net(
        keras.applications.nasnet.NASNetLarge,
        [331, 331],
        color_coding="BGR",
        preprocess_f=keras.applications.nasnet.preprocess_input,
        load_weights=load_weights,
        load_patterns=load_patterns)


def nasnet_mobile(load_weights=False, load_patterns=False):
    if load_patterns is True:
        load_patterns = {
            "file_name": "nasnet_mobile_pattern_type_relu_tf_dim_ordering_tf_kernels.npz",
            "hash": "",
        }

    if K.image_data_format() == "channels_first":
        raise Exception("NASNet is not available for channels first.")

    return _prepare_keras_net(
        keras.applications.nasnet.NASNetMobile,
        [224, 224],
        color_coding="BGR",
        preprocess_f=keras.applications.nasnet.preprocess_input,
        load_weights=load_weights,
        load_patterns=load_patterns)
