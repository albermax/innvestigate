"""Predefined network architectures for testing."""
from __future__ import annotations

import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

from innvestigate.utils.types import Layer, Model, ShapeTuple

__all__ = [
    "log_reg",
    "mlp_2dense",
    "cnn_2conv_2dense",
]


def _conv_layer() -> Layer:
    return klayers.Conv2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer="glorot_uniform",
    )


def _pooling_layer() -> Layer:
    return klayers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same",
    )


def log_reg(input_shape: ShapeTuple, output_n: int, activation=None) -> Model:
    model = kmodels.Sequential(
        [
            klayers.Input(shape=input_shape),
            klayers.Flatten(),
            klayers.Dense(output_n),
        ],
        name="log_reg",
    )
    return model


def mlp_2dense(
    input_shape: ShapeTuple,
    output_n: int,
    activation=None,
    dense_units: int = 1024,
    dropout_rate=0.4,
) -> Model:
    if activation is None:
        activation = "relu"

    model = kmodels.Sequential(
        [
            klayers.Input(shape=input_shape),
            klayers.Flatten(),
            klayers.Dense(dense_units, activation=activation),
            klayers.Dropout(dropout_rate),
            klayers.Dense(output_n),
        ],
        name="mlp_2dense",
    )
    return model


def cnn_2conv_2dense(
    input_shape: ShapeTuple,
    output_n: int,
    activation=None,
    dense_units: int = 1024,
    dropout_rate=0.4,
) -> Model:
    if activation is None:
        activation = "relu"

    model = kmodels.Sequential(
        [
            klayers.Input(shape=input_shape),
            _conv_layer(),
            _pooling_layer(),
            _conv_layer(),
            _pooling_layer(),
            klayers.Flatten(),
            klayers.Dense(dense_units, activation=activation),
            klayers.Dropout(dropout_rate),
            klayers.Dense(output_n),
        ],
        name="cnn_2conv_2dense",
    )
    return model
