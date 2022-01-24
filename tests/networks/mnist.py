"""Apply network architectures defined in base.py using MNIST input shape,
512 units per dense layer and a dropout rate of 0.25.
"""
from __future__ import annotations

import tensorflow.keras.backend as kbackend

from innvestigate.backend.types import Model

from tests.networks import base

__all__ = [
    "log_reg",
    "mlp_2dense",
    "cnn_2conv_2dense",
]

# Create dummy input data
if kbackend.image_data_format == "channels_first":
    INPUT_SHAPE = (1, 28, 28)
else:
    INPUT_SHAPE = (28, 28, 1)
N_OUTPUTS = 10
DENSE_UNITS = 512
DROPOUT = 0.25


def log_reg(activation: str = None) -> Model:
    return base.log_reg(INPUT_SHAPE, N_OUTPUTS, activation=activation)


def mlp_2dense(activation: str = None) -> Model:
    return base.mlp_2dense(
        INPUT_SHAPE,
        N_OUTPUTS,
        activation=activation,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT,
    )


def cnn_2conv_2dense(activation: str = None) -> Model:
    return base.cnn_2conv_2dense(
        INPUT_SHAPE,
        N_OUTPUTS,
        activation=activation,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT,
    )
