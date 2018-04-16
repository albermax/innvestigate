"""Example applications for image classifcation.

Each function returns a pretrained ImageNet model.
The models are based on keras.applications models and
contain additionally pretrained patterns.

The returned dictionary contains the following
keys\: model, in, sm_out, out, image_shape, color_coding,
preprocess_f, patterns.

Function parameters\:

:param load_weights: Download or access cached weights.
:param load_patterns: Download or access cached patterns.
"""
# todo: rename in, sm_out, out to input_tensors, output_tensors,
# todo: softmax_output_tenors
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
import keras.applications.resnet50
import keras.applications.vgg16
import keras.applications.vgg19
import keras.applications.inception_v3
import keras.applications.inception_resnet_v2
import keras.applications.densenet
import keras.applications.nasnet
import keras.backend as K
import keras.utils.data_utils
import numpy as np

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


PATTERNS = {
    "vgg16_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "https://www.dropbox.com/s/15lip81fzvbgkaa/vgg16_pattern_type_relu_tf_dim_ordering_tf_kernels.npz?dl=1",
        "hash": ""
    },
    "vgg19_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "https://www.dropbox.com/s/nc5empj78rfe9hm/vgg19_pattern_type_relu_tf_dim_ordering_tf_kernels.npz?dl=1",
        "hash": ""
    },
    "resnet50_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "https://www.dropbox.com/s/57jekbe8peer46i/resnet50_pattern_type_relu_tf_dim_ordering_tf_kernels.npz?dl=1",
        "hash": ""
    },
    "inception_v3_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "",
        "hash": ""
    },
    "inception_resnet_v2_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "",
        "hash": ""
    },
    "densenet121_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "",
        "hash": ""
    },
    "densenet169_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "https://www.dropbox.com/s/v6lkmvck0hrc1he/densenet169_pattern_type_relu_tf_dim_ordering_tf_kernels.npz?dl=1",
        "hash": "d1c82edf2e473d43739664605bb777e",
    },
    "densenet201_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "",
        "hash": ""
    },
    "nasnet_large_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "",
        "hash": ""
    },
    "nasnet_mobile_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "",
        "hash": ""
    },
}


def _get_patterns_info(netname, pattern_type):
    if pattern_type is True:
        pattern_type = "relu"

    file_name = ("%s_pattern_type_%s_tf_dim_ordering_tf_kernels.npz" %
                 (netname, pattern_type))

    return {"file_name": file_name,
            "url": PATTERNS[file_name]["url"],
            "hash": PATTERNS[file_name]["hash"]}


###############################################################################
###############################################################################
###############################################################################


def _prepare_keras_net(clazz, image_shape,
                       preprocess_f,
                       color_coding="RGB",
                       load_weights=False,
                       load_patterns=False):
    net = {}
    net["image_shape"] = image_shape
    if K.image_data_format() == "channels_first":
        net["input_shape"] = [None, 3]+image_shape
    else:
        net["input_shape"] = [None]+image_shape+[3]

    weights = None
    if load_weights is True:
        weights = "imagenet"
    model = clazz(weights=weights, input_shape=net["input_shape"][1:])
    net["model"] = model

    net["in"] = model.inputs
    net["sm_out"] = model.outputs
    net["out"] = kgraph.pre_softmax_tensors(model.outputs)

    net["color_coding"] = color_coding
    net["preprocess_f"] = preprocess_f

    net["patterns"] = None
    if load_patterns is not False:
        patterns_path = keras.utils.data_utils.get_file(
            load_patterns["file_name"],
            load_patterns["url"],
            cache_subdir="innvestigate_patterns",
            file_hash=None,#load_patterns["hash"],
            hash_algorithm="md5")
        patterns_file = np.load(patterns_path)
        patterns = [patterns_file["arr_%i" % i]
                    for i in range(len(patterns_file.keys()))]
        net["patterns"] = patterns
    return net


###############################################################################
###############################################################################
###############################################################################


def vgg16(load_weights=False, load_patterns=False):
    if load_patterns is not False:
        load_patterns = _get_patterns_info("vgg16", load_patterns)

    return _prepare_keras_net(
        keras.applications.vgg16.VGG16,
        [224, 224],
        preprocess_f=keras.applications.vgg16.preprocess_input,
        color_coding="BGR",
        load_weights=load_weights,
        load_patterns=load_patterns)


def vgg19(load_weights=False, load_patterns=False):
    if load_patterns is not False:
        load_patterns = _get_patterns_info("vgg19", load_patterns)

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
    if load_patterns is not False:
        load_patterns = _get_patterns_info("resnet50", load_patterns)

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
    if load_patterns is not False:
        load_patterns = _get_patterns_info("inception_v3", load_patterns)

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
    if load_patterns is not False:
        load_patterns = _get_patterns_info("inception_resnet_v2",
                                           load_patterns)

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
    if load_patterns is not False:
        load_patterns = _get_patterns_info("densenet121", load_patterns)

    return _prepare_keras_net(
        keras.applications.densenet.DenseNet121,
        [224, 224],
        preprocess_f=keras.applications.densenet.preprocess_input,
        load_weights=load_weights,
        load_patterns=load_patterns)


def densenet169(load_weights=False, load_patterns=False):
    if load_patterns is not False:
        load_patterns = _get_patterns_info("densenet169", load_patterns)

    return _prepare_keras_net(
        keras.applications.densenet.DenseNet169,
        [224, 224],
        preprocess_f=keras.applications.densenet.preprocess_input,
        load_weights=load_weights,
        load_patterns=load_patterns)


def densenet201(load_weights=False, load_patterns=False):
    if load_patterns is not False:
        load_patterns = _get_patterns_info("densenet201", load_patterns)

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
    if load_patterns is not False:
        load_patterns = _get_patterns_info("nasnet_large", load_patterns)

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
    if load_patterns is not False:
        load_patterns = _get_patterns_info("nasnet_mobile", load_patterns)

    if K.image_data_format() == "channels_first":
        raise Exception("NASNet is not available for channels first.")

    return _prepare_keras_net(
        keras.applications.nasnet.NASNetMobile,
        [224, 224],
        color_coding="BGR",
        preprocess_f=keras.applications.nasnet.preprocess_input,
        load_weights=load_weights,
        load_patterns=load_patterns)
