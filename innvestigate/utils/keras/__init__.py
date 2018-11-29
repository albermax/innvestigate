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
import keras.backend as K
import numpy as np


from ... import utils as iutils


__all__ = [
    "apply",
    "broadcast_np_tensors_to_keras_tensors",
]


###############################################################################
###############################################################################
###############################################################################


def apply(layer, inputs):
    """
    Apply a layer to input[s].

    A flexible apply that tries to fit input to layers expected input.
    This is useful when one doesn't know if a layer expects a single tensor
    or many.
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

    return iutils.to_list(ret)


def broadcast_np_tensors_to_keras_tensors(keras_tensors, np_tensors):
    def none_to_one(tmp):
        return [1 if x is None else x for x in tmp]

    keras_tensors = iutils.to_list(keras_tensors)

    if isinstance(np_tensors, list):
        ret = [np.broadcast_to(ri, none_to_one(K.int_shape(x)))
               for x, ri in zip(keras_tensors, np_tensors)]
    else:
        ret = [np.broadcast_to(np_tensors,
                               none_to_one(K.int_shape(x)))
               for x in keras_tensors]

    return ret
