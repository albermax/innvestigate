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

import keras
import keras.backend as K
from keras.engine.topology import Layer
import numpy as np


from . import utils
from .utils.keras import backend as iK


__all__ = [
    "Constant",
    "Zero",
    "One",
    "ZerosLike",
    "OnesLike",
    "AsFloatX",
    "FiniteCheck",

    "Gradient",
    "GradientWRT",

    "Min",
    "Max",
    "Greater",
    "Less",
    "Sum",
    "Square",
    "CountNonZero",

    "Transpose",
    "Dot",
    "SaveDivide",
]


###############################################################################
###############################################################################
###############################################################################


def Constant(c, reference=None):
    if reference is None:
        return K.constant(c)
    else:
        dtype = K.dtype(reference)
        return K.constant(np.dtype(dtype)(c), dtype=dtype)


def Zero(reference=None):
    return Constant(0, reference=reference)


def One(reference=None):
    return Constant(1, reference=reference)


class ZerosLike(keras.layers.Layer):
    def call(self, x):
        return [K.zeros_like(tmp) for tmp in utils.listify(x)]


class OnesLike(keras.layers.Layer):
    def call(self, x):
        return [K.ones_like(tmp) for tmp in utils.listify(x)]


class AsFloatX(keras.layers.Layer):
    def call(self, x):
        return [iK.to_floatx(tmp) for tmp in utils.listify(x)]


class FiniteCheck(keras.layers.Layer):
    def call(self, x):
        return [K.sum(iK.to_floatx(iK.is_not_finite(tmp)))
                for tmp in utils.listify(x)]


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


class _Reduce(keras.layers.Layer):

    def __init__(self, axis=-1, *args, **kwargs):
        self.axis = axis
        super(_Reduce, self).__init__(*args, **kwargs)

    def call(self, x):
        return self._apply_reduce(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape[:self.axis]) +
                     list(input_shape[self.axis+1:]))

    def _apply_reduce(self, x, axis):
        raise NotImplementedError()


class Min(_Reduce):
    def _apply_reduce(self, x, axis):
        return K.min(x, axis=axis)


class Max(_Reduce):
    def _apply_reduce(self, x, axis):
        return K.sum(x, axis=axis)


class Sum(_Reduce):
    def _apply_reduce(self, x, axis):
        return K.sum(x, axis=axis)


class CountNonZero(_Reduce):
    def _apply_reduce(self, x, axis):
        return K.sum(iK.to_floatx(K.not_equal(x, K.constant(0))), axis=axis)


###############################################################################
###############################################################################
###############################################################################


class _Map(keras.layers.Layer):

    def call(self, x):
        return self._apply_map(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _apply_map(self, x):
        raise NotImplementedError()


class Square(_Map):
    def _apply_map(self, x):
        return K.square(x)


###############################################################################
###############################################################################
###############################################################################


class Greater(keras.layers.Layer):
    def call(self, x):
        a, b = x
        return K.greater(a, b)


class Less(keras.layers.Layer):
    def call(self, x):
        a, b = x
        return K.less(a, b)


class GreaterThanZero(keras.layers.Layer):
    def call(self, x):
        return K.greater(x, K.constant(0))


class LessThanZero(keras.layers.Layer):
    def call(self, x):
        return K.less(x, K.constant(0))


class Transpose(keras.layers.Layer):

    def call(self, x):
        return K.transpose(x)

    def compute_output_shape(self, input_shape):
        return input_shape[::-1]


class Dot(keras.layers.Layer):

    def call(self, x):
        a, b = x
        return K.dot(a, b)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], input_shapes[1][1])


class SaveDivide(keras.layers.Layer):

    def call(self, x):
        a, b = x
        return a / (b + iK.to_floatx(K.equal(b, K.constant(0))))

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]
