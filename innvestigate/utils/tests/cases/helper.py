# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import numpy as np


from .... import backend


__all__ = [
    "build_keras_model",
]


###############################################################################
###############################################################################
###############################################################################


def _set_zero_weights_to_random(weights):
    ret = []
    for weight in weights:
        if weight.sum() == 0:
            weight = np.random.rand(*weight.shape)
        ret.append(weight)
    return ret


###############################################################################
###############################################################################
###############################################################################


def build_keras_model(inputs, outputs):
    model = backend.keras.models.Model(inputs=inputs,
                                       outputs=outputs)
    model.set_weights(_set_zero_weights_to_random(model.get_weights()))
    return model
