"""Test function 'innvestigate.create_analyzer'"""
from __future__ import annotations

import pytest
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

import innvestigate


@pytest.mark.fast
@pytest.mark.graph
@pytest.mark.precommit
def test_model_wo_output_activation_sequential_softmax():
    model = kmodels.Sequential(
        [klayers.Dense(10, input_shape=(10,), activation="softmax")]
    )
    model = innvestigate.model_wo_output_activation(model, "softmax")
    assert model.layers[-1].activation.__name__ == "linear"


@pytest.mark.fast
@pytest.mark.graph
@pytest.mark.precommit
def test_model_wo_output_activation_sequential_softmax():
    model = kmodels.Sequential(
        [klayers.Dense(1, input_shape=(10,), activation="sigmoid")]
    )
    model = innvestigate.model_wo_output_activation(model, "sigmoid")
    assert model.layers[-1].activation.__name__ == "linear"


@pytest.mark.fast
@pytest.mark.graph
@pytest.mark.precommit
def test_model_wo_output_activation_functional_softmax():
    inputs = klayers.Input(shape=(10,))
    outputs = klayers.Dense(10, activation="softmax")(inputs)
    model = kmodels.Model(inputs=inputs, outputs=outputs)
    model = innvestigate.model_wo_output_activation(model, "softmax")
    assert model.layers[-1].activation.__name__ == "linear"


@pytest.mark.fast
@pytest.mark.graph
@pytest.mark.precommit
def test_model_wo_output_activation_functional_sigmoid():
    inputs = klayers.Input(shape=(10,))
    outputs = klayers.Dense(1, activation="sigmoid")(inputs)
    model = kmodels.Model(inputs=inputs, outputs=outputs)
    model = innvestigate.model_wo_output_activation(model, "sigmoid")
    assert model.layers[-1].activation.__name__ == "linear"


@pytest.mark.fast
@pytest.mark.graph
@pytest.mark.precommit
def test_model_wo_output_activation_exception():
    with pytest.raises(Exception):
        model = kmodels.Sequential(
            [klayers.Dense(10, input_shape=(10,), activation="relu")]
        )
        model = innvestigate.model_wo_output_activation(model, "softmax")
