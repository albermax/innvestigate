"""Simple model architectures for fast testing of analyzers."""
from __future__ import annotations

import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

from innvestigate.backend.types import Model

__all__ = [
    "dot",
    # TODO: check why this makes problems to wrapper implementation.
    "skip_connection",
]


def dot() -> Model:
    inputs = klayers.Input(shape=(2,))
    outputs = klayers.Dense(1, activation="linear")(inputs)
    return kmodels.Model(inputs=inputs, outputs=outputs, name="dot")


def skip_connection() -> Model:
    inputs = klayers.Input(shape=(1,))
    dense = klayers.Dense(1, activation="linear", use_bias=False)
    outputs = klayers.Add()([inputs, dense(inputs)])
    return kmodels.Model(inputs=inputs, outputs=outputs, name="skip_connection")
