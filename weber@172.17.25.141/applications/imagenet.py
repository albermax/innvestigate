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
# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import range


###############################################################################
###############################################################################
###############################################################################


import tensorflow.keras.backend as K
import tensorflow.keras.applications.resnet50 as keras_applications_resnet50
import tensorflow.keras.applications.vgg16 as keras_applications_vgg16
import tensorflow.keras.applications.vgg19 as keras_applications_vgg19
import tensorflow.keras.applications.inception_v3 as keras_applications_inception_v3
import tensorflow.keras.applications.inception_resnet_v2 as keras_applications_inception_resnet_v2
import tensorflow.keras.applications.densenet as keras_applications_densenet
import tensorflow.keras.applications.nasnet as keras_applications_nasnet
import tensorflow.keras.utils as keras_utils_data_utils
import numpy as np
import warnings

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
        "hash": "8c2abe648e116a93fd5027fab49177b0",
    },
    "vgg19_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
        "url": "https://www.dropbox.com/s/nc5empj78rfe9hm/vgg19_pattern_type_relu_tf_dim_ordering_tf_kernels.npz?dl=1",
        "hash": "3258b6c64537156afe75ca7b3be44742",
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


def _prepare_keras_net(netname,
                       clazz,
                       image_shape,
                       preprocess_f,
                       preprocess_mode=None,
                       color_coding="RGB",
                       load_weights=False,
                       load_patterns=False):
    net = {}
    net["name"] = netname
    net["image_shape"] = image_shape
    if K.image_data_format() == "channels_first":
        net["input_shape"] = [None, 3]+image_shape
    else:
        net["input_shape"] = [None]+image_shape+[3]

    weights = None
    if load_weights is True:
        weights = "imagenet"

    model = clazz(weights=weights,
                  input_shape=tuple(net["input_shape"][1:]))
    net["model"] = model

    net["in"] = model.inputs
    net["sm_out"] = model.outputs
    net["out"] = kgraph.pre_softmax_tensors(model.outputs)

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
            patterns_path = keras_utils_data_utils.get_file(
                pattern_info["file_name"],
                pattern_info["url"],
                cache_subdir="innvestigate_patterns",
                hash_algorithm="md5",
                file_hash=pattern_info["hash"])
            patterns_file = np.load(patterns_path)
            patterns = [patterns_file["arr_%i" % i]
                        for i in range(len(patterns_file.keys()))]
            net["patterns"] = patterns
    return net


###############################################################################
###############################################################################
###############################################################################


def vgg16(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "vgg16",
        keras_applications_vgg16.VGG16,
        [224, 224],
        preprocess_f=keras_applications_vgg16.preprocess_input,
        preprocess_mode="caffe",
        color_coding="BGR",
        load_weights=load_weights,
        load_patterns=load_patterns)


def vgg19(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "vgg19",
        keras_applications_vgg19.VGG19,
        [224, 224],
        preprocess_f=keras_applications_vgg19.preprocess_input,
        preprocess_mode="caffe",
        color_coding="BGR",
        load_weights=load_weights,
        load_patterns=load_patterns)


###############################################################################
###############################################################################
###############################################################################


def resnet50(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "resnet50",
        keras_applications_resnet50.ResNet50,
        [224, 224],
        preprocess_f=keras_applications_resnet50.preprocess_input,
        preprocess_mode="caffe",
        color_coding="BGR",
        load_weights=load_weights,
        load_patterns=load_patterns)


###############################################################################
###############################################################################
###############################################################################


def inception_v3(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "inception_v3",
        keras_applications_inception_v3.InceptionV3,
        [299, 299],
        preprocess_f=keras_applications_inception_v3.preprocess_input,
        preprocess_mode="tf",
        load_weights=load_weights,
        load_patterns=load_patterns)


###############################################################################
###############################################################################
###############################################################################


def inception_resnet_v2(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "inception_resnet_v2",
        keras_applications_inception_resnet_v2.InceptionResNetV2,
        [299, 299],
        preprocess_f=keras_applications_inception_resnet_v2.preprocess_input,
        preprocess_mode="tf",
        load_weights=load_weights,
        load_patterns=load_patterns)


###############################################################################
###############################################################################
###############################################################################


def densenet121(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "densenet121",
        keras_applications_densenet.DenseNet121,
        [224, 224],
        preprocess_f=keras_applications_densenet.preprocess_input,
        preprocess_mode="torch",
        load_weights=load_weights,
        load_patterns=load_patterns)


def densenet169(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "densenet169",
        keras_applications_densenet.DenseNet169,
        [224, 224],
        preprocess_f=keras_applications_densenet.preprocess_input,
        preprocess_mode="torch",
        load_weights=load_weights,
        load_patterns=load_patterns)


def densenet201(load_weights=False, load_patterns=False):
    return _prepare_keras_net(
        "densenet201",
        keras_applications_densenet.DenseNet201,
        [224, 224],
        preprocess_f=keras_applications_densenet.preprocess_input,
        preprocess_mode="torch",
        load_weights=load_weights,
        load_patterns=load_patterns)


###############################################################################
###############################################################################
###############################################################################


def nasnet_large(load_weights=False, load_patterns=False):
    if K.image_data_format() == "channels_first":
        raise Exception("NASNet is not available for channels first.")

    return _prepare_keras_net(
        "nasnet_large",
        keras_applications_nasnet.NASNetLarge,
        [331, 331],
        color_coding="BGR",
        preprocess_f=keras_applications_nasnet.preprocess_input,
        preprocess_mode="tf",
        load_weights=load_weights,
        load_patterns=load_patterns)


def nasnet_mobile(load_weights=False, load_patterns=False):
    if K.image_data_format() == "channels_first":
        raise Exception("NASNet is not available for channels first.")

    return _prepare_keras_net(
        "nasnet_mobile",
        keras_applications_nasnet.NASNetMobile,
        [224, 224],
        color_coding="BGR",
        preprocess_f=keras_applications_nasnet.preprocess_input,
        preprocess_mode="tf",
        load_weights=load_weights,
        load_patterns=load_patterns)
