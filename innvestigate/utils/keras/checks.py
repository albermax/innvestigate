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


import inspect
import keras.backend as K
import keras.engine.topology
import keras.layers


__all__ = [
    "contains_activation",
    "contains_kernel",
    "is_container",
    "is_convnet_layer",
    "is_relu_convnet_layer",
]


###############################################################################
###############################################################################
###############################################################################


def contains_activation(layer, activation=None):
    """
    Check whether the layer contains an activation function.
    activation is None then we only check if layer can contain an activation.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "activation"):
        if activation is not None:
            return layer.activation == keras.activations.get(activation)
        else:
            return True
    else:
        return False


def contains_kernel(layer):
    """
    Check whether the layer contains a kernel.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "kernel"):
        return True
    else:
        return False


def contains_bias(layer):
    """
    Check whether the layer contains a bias.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "bias"):
        return True
    else:
        return False


def is_container(layer):
    return isinstance(layer, keras.engine.topology.Container)


def is_convnet_layer(layer):
    # todo: add checks, e.g., no recurrent layers
    return True


def is_relu_convnet_layer(layer):
    return (is_convnet_layer(layer) and
            (not contains_activation(layer) or
             contains_activation(layer, None) or
             contains_activation(layer, "linear") or
             contains_activation(layer, "relu")))


def is_input_layer(layer):
    # Triggers if ALL inputs of layer are connected
    # to a Keras input layer object or
    # the layer itself is the first layer.

    layer_inputs = get_input_layers(layer)
    # We ignore certain layers, that do not modify
    # the data content.
    # todo: update this list!
    IGNORED_LAYERS = (
        keras.layers.Flatten,
        keras.layers.Permute,
        keras.layers.Reshape,
    )
    while any([isinstance(x, IGNORED_LAYERS) for x in layer_inputs]):
        tmp = set()
        for l in layer_inputs:
            if isinstance(l, IGNORED_LAYERS):
                tmp.update(get_input_layers(l))
            else:
                tmp.add(l)
        layer_inputs = tmp

    if all([isinstance(x, keras.layers.InputLayer)
            for x in layer_inputs]):
        return True
    elif getattr(layer, "input_shape", None) is not None:
        # relies on Keras convention
        return True
    elif getattr(layer, "batch_input_shape", None) is not None:
        # relies on Keras convention
        return True
    else:
        return False
