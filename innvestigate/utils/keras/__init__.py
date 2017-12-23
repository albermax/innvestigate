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


import keras.activations

from ... import utils as iutils


__all__ = [
    "easy_apply",

    "contains_activation",
    "contains_kernel",
]


###############################################################################
###############################################################################
###############################################################################


def easy_apply(layer, inputs):
    """
    Apply a layer to input[s].

    We don't know at this point if layer can handle a list of tensor and
    just try therefore.
    """

    if isinstance(inputs, list) and len(inputs) > 1:
        try:
            ret = layer(inputs)
        except (TypeError, AttributeError):
            # layer expects a single tensor.
            if len(inputs) != 1:
                raise ValueError("Layer expects only a single input!")
            ret = layer(inputs[0])
    else:
        ret = layer(inputs[0])

    return iutils.listify(ret)


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
