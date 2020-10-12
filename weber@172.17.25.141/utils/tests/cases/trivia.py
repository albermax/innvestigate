# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import numpy as np

from .... import backend

from . import helper


__all__ = [
    "dot",
    "skip_connection",
]


###############################################################################
###############################################################################
###############################################################################


def dot():
    input_shape = (1, 2)
    data = np.random.rand(*input_shape)

    if backend.name() == "tensorflow":
        layers = backend.keras.layers

        inputs = layers.Input(shape=input_shape[1:])
        outputs = layers.Dense(units=1, activation="linear")(inputs)
        model = helper.build_keras_model(inputs, outputs)

    else:
        raise NotImplementedError()

    return model, data


def skip_connection():
    input_shape = (1, 1)
    data = np.random.rand(*input_shape)

    if backend.name() == "tensorflow":
        layers = backend.keras.layers

        inputs = layers.Input(shape=input_shape[1:])
        tmp = layers.Dense(units=1, activation="linear")(inputs)
        outputs = layers.Add()([inputs, tmp])
        model = helper.build_keras_model(inputs, outputs)
    else:
        raise NotImplementedError()

    return model, data
