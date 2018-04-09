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

import os
import keras.layers
from keras.utils.data_utils import get_file
from keras.models import load_model, clone_model
from keras.models import Sequential




__all__ = [
    "log_reg",

    "mlp_2dense",
    "mlp_3dense",

    "cnn_1convb_2dense",
    "cnn_2convb_2dense",
    "cnn_2convb_3dense",
    "cnn_3convb_3dense",

    "pt_plos_long_relu",
    "pt_plos_short_relu",
    "pt_plos_long_tanh",
    "pt_plos_short_tanh",
]


###############################################################################
###############################################################################
###############################################################################


# TODO: more consistent nameing

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
            pool_size=(2, 2),
            strides=(2, 2),
        )(current_layer)
    return ret


def dropout_layer(layer_in, *args, **kwargs):
    return keras.layers.Dropout(*args, **kwargs)(layer_in)


def softmax(layer_in):
    return keras.layers.Activation("softmax")(layer_in)


###############################################################################
###############################################################################
###############################################################################


def log_reg(input_shape, output_n, activation=None):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net["in_flat"] = keras.layers.Flatten()(net["in"])
    net["out"] = dense_layer(net["in_flat"], units=output_n,
                             kernel_initializer="glorot_uniform")
    net["sm_out"] = softmax(net["out"])

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


###############################################################################
###############################################################################
###############################################################################


def mlp_2dense(input_shape, output_n, activation=None,
               dense_units=512, dropout_rate=0.25):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net["in_flat"] = keras.layers.Flatten()(net["in"])
    net["dense_1"] = dense_layer(net["in_flat"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net["dense_1_dropout"] = dropout_layer(net["dense_1"], dropout_rate)
    net["out"] = dense_layer(net["dense_1_dropout"], units=output_n,
                             kernel_initializer="glorot_uniform")
    net["sm_out"] = softmax(net["out"])

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


def mlp_3dense(input_shape, output_n, activation=None,
               dense_units=512, dropout_rate=0.25):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net["in_flat"] = keras.layers.Flatten()(net["in"])
    net["dense_1"] = dense_layer(net["in_flat"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net["dense_1_dropout"] = dropout_layer(net["dense_1"], dropout_rate)
    net["dense_2"] = dense_layer(net["dense_1_dropout"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net["dense_2_dropout"] = dropout_layer(net["dense_2"], dropout_rate)
    net["out"] = dense_layer(net["dense_2_dropout"], units=output_n,
                             kernel_initializer="glorot_uniform")
    net["sm_out"] = softmax(net["out"])

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


###############################################################################
###############################################################################
###############################################################################


def cnn_1convb_2dense(input_shape, output_n, activation=None,
                      dense_units=512, dropout_rate=0.25):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net.update(conv_pool(net["in"], 2, "conv_1", 128,
                         activation=activation))
    net["conv_flat"] = keras.layers.Flatten()(net["conv_1_pool"])
    net["dense_1"] = dense_layer(net["conv_flat"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net["dense_1_dropout"] = dropout_layer(net["dense_1"], dropout_rate)
    net["out"] = dense_layer(net["dense_1_dropout"], units=output_n,
                             kernel_initializer="glorot_uniform")
    net["sm_out"] = softmax(net["out"])

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
    net["conv_flat"] = keras.layers.Flatten()(net["conv_2_pool"])
    net["dense_1"] = dense_layer(net["conv_flat"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net["dense_1_dropout"] = dropout_layer(net["dense_1"], dropout_rate)
    net["out"] = dense_layer(net["dense_1_dropout"], units=output_n,
                             kernel_initializer="glorot_uniform")
    net["sm_out"] = softmax(net["out"])

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


def cnn_2convb_3dense(input_shape, output_n, activation=None,
                      dense_units=512, dropout_rate=0.25):
    if activation is None:
        activation = "relu"

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net.update(conv_pool(net["in"], 2, "conv_1", 128,
                         activation=activation))
    net.update(conv_pool(net["conv_1_pool"], 2, "conv_2", 128,
                         activation=activation))
    net["conv_flat"] = keras.layers.Flatten()(net["conv_2_pool"])
    net["dense_1"] = dense_layer(net["conv_flat"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net["dense_1_dropout"] = dropout_layer(net["dense_1"], dropout_rate)
    net["dense_2"] = dense_layer(net["dense_1_dropout"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net["dense_2_dropout"] = dropout_layer(net["dense_2"], dropout_rate)
    net["out"] = dense_layer(net["dense_2_dropout"], units=output_n,
                             kernel_initializer="glorot_uniform")
    net["sm_out"] = softmax(net["out"])

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net


def cnn_3convb_3dense(input_shape, output_n, activation=None,
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
    net["conv_flat"] = keras.layers.Flatten()(net["conv_3_pool"])
    net["dense_1"] = dense_layer(net["conv_flat"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net["dense_1_dropout"] = dropout_layer(net["dense_1"], dropout_rate)
    net["dense_2"] = dense_layer(net["dense_1_dropout"], units=dense_units,
                                 activation=activation,
                                 kernel_initializer="glorot_uniform")
    net["dense_2_dropout"] = dropout_layer(net["dense_2"], dropout_rate)
    net["out"] = dense_layer(net["dense_2_dropout"], units=output_n,
                             kernel_initializer="glorot_uniform")
    net["sm_out"] = softmax(net["out"])

    net.update({
        "input_shape": input_shape,

        "output_n": output_n,
    })
    return net



PRETRAINED_MODELS = {"pt_plos_long_relu":
                        {"file":"plos-mnist-rect-long.h5",
                         "url" : "https://www.dropbox.com/s/26w7i58qqcuosn4/plos-mnist-rect-long.h5"
                        },
                     "pt_plos_short_relu":
                        {"file":"plos-mnist-rect-short.h5",
                         "url":"https://www.dropbox.com/s/89nvwyls55xycmw/plos-mnist-rect-short.h5"
                        },
                     "pt_plos_long_tanh":
                        {"file":"plos-mnist-tanh-long.h5",
                         "url":"https://www.dropbox.com/s/61e3a4gdbjo9bca/plos-mnist-tanh-long.h5"
                        },
                     "pt_plos_short_tanh":
                        {"file":"plos-mnist-tanh-short.h5",
                         "url":"https://www.dropbox.com/s/foqv60kot0retfr/plos-mnist-tanh-short.h5"
                        },
                    }

def _load_pretrained_net(modelname, new_input_shape):
    #TODO: adapt / tidy up to your liking
    filename = PRETRAINED_MODELS[modelname]["file"]
    urlname = PRETRAINED_MODELS[modelname]["url"]
    #model_path = get_file(filename, urlname) #TODO: FIX! corrupts the file?
    model_path = filename
    print (model_path)

    #workaround the more elegant, but dysfunctional solution.
    if not os.path.isfile(model_path):
        os.system("wget {}".format(urlname))


    model = load_model(model_path)
    #create replacement input layer with new shape.
    model.layers[0] = keras.layers.InputLayer(input_shape=new_input_shape, name="input_1")
    model = Sequential(layers=model.layers)

    model_w_sm = clone_model(model)
    model_w_sm.set_weights(model.get_weights())
    model_w_sm.add(keras.layers.Activation("softmax"))
    return model, model_w_sm


def pt_plos_long_relu(input_shape, **kwargs):
    return _load_pretrained_net("pt_plos_long_relu", input_shape)

def pt_plos_short_relu(input_shape, **kwargs):
    return _load_pretrained_net("pt_plos_short_relu", input_shape)

def pt_plos_long_tanh(input_shape, **kwargs):
    return _load_pretrained_net("pt_plos_long_tanh", input_shape)

def pt_plos_short_tanh(input_shape, **kwargs):
    return _load_pretrained_net("pt_plos_short_tanh", input_shape)
