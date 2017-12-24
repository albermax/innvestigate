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


from . import utils
from .utils.keras import backend as iK

import keras
import keras.backend as K
from keras.engine.topology import Layer


__all__ = [
    "FiniteCheck",
    "OnesLike",

    "Gradient",
    "GradientWRT",

    "Max",
    "Sum",
]


###############################################################################
###############################################################################
###############################################################################


class FiniteCheck(keras.layers.Layer):
    def call(self, x):
        return [K.sum(K.cast(iK.is_not_finite(tmp), K.floatx()))
                for tmp in utils.listify(x)]


class OnesLike(keras.layers.Layer):
    def call(self, x):
        return [K.ones_like(tmp) for tmp in utils.listify(x)]


###############################################################################
###############################################################################
###############################################################################


class Gradient(keras.layers.Layer):
    "Returns gradient of sum(output), expects inputs+[output,]."

    def call(self, x):
        inputs, output = x[:-1], x[-1]
        return K.gradients(K.sum(output), inputs)

    def compute_output_shape(self, input_shapes):
        return input_shapes[:-1]


class GradientWRT(keras.layers.Layer):
    "Returns gradient wrt to another layer and given gradient,"
    " expects inputs+[output,]."

    def __init__(self, n_inputs, *args, **kwargs):
        self.n_inputs = n_inputs
        super(GradientWRT, self).__init__(*args, **kwargs)

    def call(self, x):
        assert isinstance(x, (list, tuple))
        Xs, tmp_Ys = x[:self.n_inputs], x[self.n_inputs:]
        assert len(tmp_Ys) % 2 == 0
        len_Ys = len(tmp_Ys) // 2
        Ys, known_Ys = tmp_Ys[:len_Ys], tmp_Ys[len_Ys:]
        return iK.gradients(Xs, Ys, known_Ys)

    def compute_output_shape(self, input_shapes):
        return input_shapes[:self.n_inputs]


###############################################################################
###############################################################################
###############################################################################


class Max(keras.layers.Layer):
    "Returns maximum along the last dimension."

    def call(self, x):
        return K.max(x, axis=-1)

    def compute_output_shape(self, input_shapes):
        return input_shapes[:-1]


class Sum(keras.layers.Layer):
    "Returns sum along the last dimension."

    def call(self, x):
        return K.sum(x, axis=-1)

    def compute_output_shape(self, input_shapes):
        return input_shapes[:-1]
