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


PATTERN_BASE_URL = "to be set"
PATTERN_HASHES = {}


def _get_patterns_info(netname, pattern_type):
    if pattern_type is True:
        pattern_type = "relu"

    file_name = ("%s_pattern_type_%s_tf_dim_ordering_tf_kernels.npz" %
                 (netname, pattern_type))

    return {"file_name": file_name,
            "hash": PATTERN_HASHES.get(file_name, None)}


###############################################################################
###############################################################################
###############################################################################


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
        # Temporary workaround to keep the examples going:
        pattern_file = "./imagenet_224_vgg_16.pattern_file.A_only.npz"
        pattern_url = "https://www.dropbox.com/s/v7e0px44jqwef5k/imagenet_224_vgg_16.patterns.A_only.npz?dl=1"

        import os
        import shutil
        import numpy as np

        def download(url, filename):
            if not os.path.exists(filename):
                print("Download: %s ---> %s" % (url, filename))
                response = six.moves.urllib.request.urlopen(url)
                with open(filename, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)

        def load_patterns(filename):
            f = np.load(filename)

            ret = {}
            for prefix in ["A", "r", "mu"]:
                l = sum([x.startswith(prefix) for x in f.keys()])
                ret.update({prefix: [f["%s_%i" % (prefix, i)] for i in range(l)]})

            return ret["A"]

        def lasagne_weights_to_keras_weights(weights):
            ret = []
            for w in weights:
                if len(w.shape) < 4:
                    ret.append(w)
                else:
                    ret.append(w.transpose(2, 3, 1, 0))
            return ret

        # Download the necessary parameters for VGG16 and the according patterns.
        download(pattern_url, pattern_file)
        patterns = lasagne_weights_to_keras_weights(load_patterns(pattern_file))
        net["patterns"] = patterns

        # Code that should be used in the future:
        if False:
            weights_path = keras.utils.data_utils.get_file(
                load_patterns["file_name"],
                PATTERN_BASE_URL % load_patterns["file_name"],
                cache_subdir="innvestigate_patterns",
                file_hash=load_patterns["hash"])
            # todo: add loading too.
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
