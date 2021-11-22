"""Example applications for image classifcation.

Each function returns a pretrained ImageNet model.
The models are based on keras.applications models and
contain additionally pretrained patterns.

The returned dictionary contains the following keys:
    model, in, sm_out, out, image_shape, color_coding,
    preprocess_f, patterns.

Function parameters:
    :param load_weights: Download or access cached weights.
    :param load_patterns: Download or access cached patterns.
"""
# TODO: rename in, sm_out, out to
# TODO: input_tensors, output_tensors, softmax_output_tenors

from __future__ import annotations

import warnings
from builtins import range

import numpy as np
import tensorflow.keras.applications as kapplications
import tensorflow.keras.backend as kbackend
import tensorflow.python.keras.utils as kutils

import innvestigate.utils.keras.graph as igraph

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


PATTERNS = {
    "vgg16_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "https://www.dropbox.com/s/15lip81fzvbgkaa/vgg16_pattern_type_relu_tf_dim_ordering_tf_kernels.npz?dl=1",  # noqa
        "hash": "8c2abe648e116a93fd5027fab49177b0",
    },
    "vgg19_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "https://www.dropbox.com/s/nc5empj78rfe9hm/vgg19_pattern_type_relu_tf_dim_ordering_tf_kernels.npz?dl=1",  # noqa
        "hash": "3258b6c64537156afe75ca7b3be44742",
    },
}


def _get_patterns_info(netname, pattern_type):
    if pattern_type is True:
        pattern_type = "relu"

    file_name = "%s_pattern_type_%s_tf_dim_ordering_tf_kernels.npz" % (
        netname,
        pattern_type,
    )

    return {
        "file_name": file_name,
        "url": PATTERNS[file_name]["url"],
        "hash": PATTERNS[file_name]["hash"],
    }


###############################################################################


def _prepare_keras_net(
    netname,
    clazz,
    image_shape,
    preprocess_f,
    preprocess_mode=None,
    color_coding="RGB",
    load_weights=False,
    load_patterns=False,
):
    net = {}
    net["name"] = netname
    net["image_shape"] = image_shape
    if kbackend.image_data_format() == "channels_first":
        net["input_shape"] = [None, 3] + image_shape
    else:
        net["input_shape"] = [None] + image_shape + [3]

    weights = None
    if load_weights is True:
        weights = "imagenet"

    model = clazz(weights=weights, input_shape=tuple(net["input_shape"][1:]))
    net["model"] = model

    net["in"] = model.inputs
    net["sm_out"] = model.outputs
    net["out"] = igraph.pre_softmax_tensors(model.outputs)

    net["color_coding"] = color_coding
    net["preprocess_f"] = preprocess_f
    net["input_range"] = {
        None: (-128, 128),
        "caffe": (-128, 128),
        "tf": (-1, 1),
        "torch": (-3, 3),
    }[preprocess_mode]

    net["patterns"] = None
    if load_patterns is not False:
        try:
            pattern_info = _get_patterns_info(netname, load_patterns)
        except KeyError:
            warnings.warn("There are no patterns for network '%s'." % netname)
        else:
            patterns_path = kutils.data_utils.get_file(
                pattern_info["file_name"],
                pattern_info["url"],
                cache_subdir="innvestigate_patterns",
                hash_algorithm="md5",
                file_hash=pattern_info["hash"],
            )
            patterns_file = np.load(patterns_path)
            patterns = [
                patterns_file["arr_%i" % i] for i in range(len(patterns_file.keys()))
            ]
            net["patterns"] = patterns
    return net


###############################################################################


def vgg16(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "vgg16",
        kapplications.vgg16.VGG16,
        [224, 224],
        preprocess_f=kapplications.vgg16.preprocess_input,
        preprocess_mode="caffe",
        color_coding="BGR",
        load_weights=load_weights,
        load_patterns=load_patterns,
    )


def vgg19(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "vgg19",
        kapplications.vgg19.VGG19,
        [224, 224],
        preprocess_f=kapplications.vgg19.preprocess_input,
        preprocess_mode="caffe",
        color_coding="BGR",
        load_weights=load_weights,
        load_patterns=load_patterns,
    )


###############################################################################


def resnet50(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "resnet50",
        kapplications.resnet50.ResNet50,
        [224, 224],
        preprocess_f=kapplications.resnet50.preprocess_input,
        preprocess_mode="caffe",
        color_coding="BGR",
        load_weights=load_weights,
        load_patterns=load_patterns,
    )


###############################################################################


def inception_v3(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "inception_v3",
        kapplications.inception_v3.InceptionV3,
        [299, 299],
        preprocess_f=kapplications.inception_v3.preprocess_input,
        preprocess_mode="tf",
        load_weights=load_weights,
        load_patterns=load_patterns,
    )


###############################################################################


def inception_resnet_v2(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "inception_resnet_v2",
        kapplications.inception_resnet_v2.InceptionResNetV2,
        [299, 299],
        preprocess_f=kapplications.inception_resnet_v2.preprocess_input,
        preprocess_mode="tf",
        load_weights=load_weights,
        load_patterns=load_patterns,
    )


###############################################################################


def densenet121(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "densenet121",
        kapplications.densenet.DenseNet121,
        [224, 224],
        preprocess_f=kapplications.densenet.preprocess_input,
        preprocess_mode="torch",
        load_weights=load_weights,
        load_patterns=load_patterns,
    )


def densenet169(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "densenet169",
        kapplications.densenet.DenseNet169,
        [224, 224],
        preprocess_f=kapplications.densenet.preprocess_input,
        preprocess_mode="torch",
        load_weights=load_weights,
        load_patterns=load_patterns,
    )


def densenet201(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "densenet201",
        kapplications.densenet.DenseNet201,
        [224, 224],
        preprocess_f=kapplications.densenet.preprocess_input,
        preprocess_mode="torch",
        load_weights=load_weights,
        load_patterns=load_patterns,
    )


###############################################################################


def nasnet_large(load_weights=False, load_patterns=False):
    if kbackend.image_data_format() == "channels_first":
        raise Exception("NASNet is not available for channels first.")

    return _prepare_keras_net(
        "nasnet_large",
        kapplications.nasnet.NASNetLarge,
        [331, 331],
        color_coding="BGR",
        preprocess_f=kapplications.nasnet.preprocess_input,
        preprocess_mode="tf",
        load_weights=load_weights,
        load_patterns=load_patterns,
    )


def nasnet_mobile(load_weights=False, load_patterns=False):
    if kbackend.image_data_format() == "channels_first":
        raise Exception("NASNet is not available for channels first.")

    return _prepare_keras_net(
        "nasnet_mobile",
        kapplications.nasnet.NASNetMobile,
        [224, 224],
        color_coding="BGR",
        preprocess_f=kapplications.nasnet.preprocess_input,
        preprocess_mode="tf",
        load_weights=load_weights,
        load_patterns=load_patterns,
    )
