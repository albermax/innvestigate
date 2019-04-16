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
    "batchnorm",
    "dropout",
]


###############################################################################
###############################################################################
###############################################################################


def _base(special_layer):
    input_shape = (1, 2)
    data = np.random.rand(*input_shape)

    if backend.name() == "tensorflow":
        layers = backend.keras.layers

        inputs = layers.Input(shape=input_shape[1:])
        tmp = layers.Dense(units=2, activation="linear")(inputs)
        tmp = special_layer(tmp)
        outputs = layers.Dense(units=1, activation="linear")(tmp)
        model = helper.build_keras_model(inputs, outputs)
    else:
        raise NotImplementedError()

    return model, data


def batchnorm():
    if backend.name() == "tensorflow":
        special_layer = backend.keras.layers.BatchNormalization()
    else:
        raise NotImplementedError()

    return _base(special_layer)


def dropout():
    if backend.name() == "tensorflow":
        special_layer = backend.keras.layers.Dropout(0.5)
    else:
        raise NotImplementedError()

    return _base(special_layer)
