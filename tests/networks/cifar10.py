"""Apply network architectures defined in base.py using CIFAR-10 input shape,
1024 units per dense layer and a dropout rate of 0.5.
"""
from __future__ import annotations

import tensorflow.keras.backend as kbackend

from innvestigate.utils.types import Model

from tests.networks import base

__all__ = [
    "log_reg",
    "mlp_2dense",
    "mlp_3dense",
    "cnn_1convb_2dense",
    "cnn_2convb_2dense",
    "cnn_2convb_3dense",
    "cnn_3convb_3dense",
]

if kbackend.image_data_format == "channels_first":
    INPUT_SHAPE = (3, 32, 32)
else:
    INPUT_SHAPE = (32, 32, 3)
N_OUTPUTS = 10
DENSE_UNITS = 1024
DROPOUT = 0.5


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


def mlp_3dense(activation: str = None) -> Model:
    return base.mlp_3dense(
        INPUT_SHAPE,
        N_OUTPUTS,
        activation=activation,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT,
    )


def cnn_1convb_2dense(activation: str = None) -> Model:
    return base.cnn_1convb_2dense(
        INPUT_SHAPE,
        N_OUTPUTS,
        activation=activation,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT,
    )


def cnn_2convb_2dense(activation: str = None) -> Model:
    return base.cnn_2convb_2dense(
        INPUT_SHAPE,
        N_OUTPUTS,
        activation=activation,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT,
    )


def cnn_2convb_3dense(activation: str = None) -> Model:
    return base.cnn_2convb_3dense(
        INPUT_SHAPE,
        N_OUTPUTS,
        activation=activation,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT,
    )


def cnn_3convb_3dense(activation: str = None) -> Model:
    return base.cnn_3convb_3dense(
        INPUT_SHAPE,
        N_OUTPUTS,
        activation=activation,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT,
    )
