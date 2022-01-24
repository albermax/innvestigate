"""Predefined network architectures for testing."""
from __future__ import annotations

import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

from innvestigate.backend.types import Layer, Model, ShapeTuple

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
    inputs = klayers.Input(shape=input_shape)
    x = klayers.Flatten()(inputs)
    outputs = klayers.Dropout(0.25)(x, training=False)
    return kmodels.Model(inputs=inputs, outputs=outputs, name="log_reg")


def mlp_2dense(
    input_shape: ShapeTuple,
    output_n: int,
    activation=None,
    dense_units: int = 1024,
    dropout_rate=0.4,
) -> Model:
    if activation is None:
        activation = "relu"

    inputs = klayers.Input(shape=input_shape)
    x = klayers.Flatten()(inputs)
    x = klayers.Dense(dense_units, activation=activation)(x)
    x = klayers.Dropout(dropout_rate)(x, training=False)
    outputs = klayers.Dense(output_n)(x)
    return kmodels.Model(inputs=inputs, outputs=outputs, name="mlp_2dense")


def cnn_2conv_2dense(
    input_shape: ShapeTuple,
    output_n: int,
    activation=None,
    dense_units: int = 1024,
    dropout_rate=0.4,
) -> Model:
    if activation is None:
        activation = "relu"

    inputs = klayers.Input(shape=input_shape)
    x = _conv_layer()(inputs)
    x = _pooling_layer()(x)
    x = _conv_layer()(x)
    x = _pooling_layer()(x)
    x = klayers.Flatten()(inputs)
    x = klayers.Dense(dense_units, activation=activation)(x)
    x = klayers.Dropout(dropout_rate)(x, training=False)
    outputs = klayers.Dense(output_n)(x)
    return kmodels.Model(inputs=inputs, outputs=outputs, name="cnn_2conv_2dense")
