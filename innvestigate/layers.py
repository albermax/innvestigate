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


import keras
import keras.backend as K
from keras.engine.topology import Layer


__all__ = ["Gradient", "Max", "Sum"]


class Gradient(keras.layers.Layer):
    "Returns gradient of sum(output), expects inputs+[output,]."

    def call(self, x):
        inputs, output = x[:-1], x[-1]
        return K.gradients(K.sum(output), inputs)

    def compute_output_shape(self, input_shapes):
        return input_shapes[:-1]


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
