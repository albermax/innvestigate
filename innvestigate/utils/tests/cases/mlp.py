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
    "mlp2",
    "mlp3",
]


###############################################################################
###############################################################################
###############################################################################


def _mlp(n_layers):
    input_shape = (1, 2)
    data = np.random.rand(*input_shape)

    if backend.name() == "tensorflow":
        layers = backend.keras.layers

        inputs = layers.Input(shape=input_shape[1:])
        tmp = inputs
        for i in range(n_layers-1):
            tmp = layers.Dense(units=2, activation="relu")(tmp)
        outputs = layers.Dense(units=1, activation="linear")(tmp)
        model = helper.build_keras_model(inputs, outputs)
    else:
        raise NotImplementedError()

    return model, data


def mlp2():
    return _mlp(2)


def mlp3():
    return _mlp(3)
