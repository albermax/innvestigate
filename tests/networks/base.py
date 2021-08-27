"""Predefined network architectures for testing."""
from __future__ import annotations

import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

from innvestigate.utils.types import Layer, Model, ShapeTuple

__all__ = [
    "log_reg",
    "mlp_2dense",
    "mlp_3dense",
    "cnn_1convb_2dense",
    "cnn_2convb_2dense",
    "cnn_2convb_3dense",
    "cnn_3convb_3dense",
]


def _conv_layer() -> Layer:
    return klayers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="glorot_uniform",
    )


def _pooling_layer() -> Layer:
    return klayers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
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
    dense_units: int = 512,
    dropout_rate=0.25,
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


def mlp_3dense(
    input_shape: ShapeTuple,
    output_n: int,
    activation=None,
    dense_units: int = 512,
    dropout_rate=0.25,
) -> Model:
    if activation is None:
        activation = "relu"

    model = kmodels.Sequential(
        [
            klayers.Input(shape=input_shape),
            klayers.Flatten(),
            klayers.Dense(dense_units, activation=activation),
            klayers.Dropout(dropout_rate),
            klayers.Dense(dense_units, activation=activation),
            klayers.Dropout(dropout_rate),
            klayers.Dense(output_n),
        ],
        name="mlp_3dense",
    )
    return model


def cnn_1convb_2dense(
    input_shape: ShapeTuple,
    output_n: int,
    activation=None,
    dense_units: int = 512,
    dropout_rate=0.25,
) -> Model:
    if activation is None:
        activation = "relu"

    model = kmodels.Sequential(
        [
            klayers.Input(shape=input_shape),
            _conv_layer(),
            _pooling_layer(),
            klayers.Flatten(),
            klayers.Dense(dense_units, activation=activation),
            klayers.Dropout(dropout_rate),
            klayers.Dense(output_n),
        ],
        name="cnn_1convb_2dense",
    )
    return model


def cnn_2convb_2dense(
    input_shape: ShapeTuple,
    output_n: int,
    activation=None,
    dense_units: int = 512,
    dropout_rate=0.25,
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
        name="cnn_2convb_2dense",
    )
    return model


def cnn_2convb_3dense(
    input_shape: ShapeTuple,
    output_n: int,
    activation=None,
    dense_units: int = 512,
    dropout_rate=0.25,
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
            klayers.Dense(dense_units, activation=activation),
            klayers.Dropout(dropout_rate),
            klayers.Dense(output_n),
        ],
        name="cnn_2convb_3dense",
    )
    return model


def cnn_3convb_3dense(
    input_shape: ShapeTuple,
    output_n: int,
    activation=None,
    dense_units: int = 512,
    dropout_rate=0.25,
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
            _conv_layer(),
            _pooling_layer(),
            klayers.Flatten(),
            klayers.Dense(dense_units, activation=activation),
            klayers.Dropout(dropout_rate),
            klayers.Dense(dense_units, activation=activation),
            klayers.Dropout(dropout_rate),
            klayers.Dense(output_n),
        ],
        name="cnn_3convb_3dense",
    )
    return model
