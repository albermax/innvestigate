from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

import innvestigate.layers as ilayers


@pytest.mark.graph
@pytest.mark.ilayers
@pytest.mark.fast
@pytest.mark.precommit
def test_max_neuron_selection_layer():
    inputs = klayers.Input(shape=(6,))
    outputs = ilayers.MaxNeuronSelection()(inputs)
    model = kmodels.Model(inputs=inputs, outputs=outputs, name="TestMaxNeuronSelection")

    x = np.array([[1, 2, 3, 6, 5, 4], [7, 42, 9, 10, 11, 12]])
    out = model.predict(x)
    print(out)
    assert np.all(out == np.array([6.0, 42.0]))

    inputs = klayers.Input(shape=(2, 3))
    outputs = ilayers.MaxNeuronSelection()(inputs)
    model = kmodels.Model(inputs=inputs, outputs=outputs, name="TestMaxNeuronSelection")

    x = np.array([[[1, 2, 3], [6, 5, 4]], [[7, 42, 9], [10, 11, 12]]])
    out = model.predict(x)
    assert np.all(out == np.array([[3.0, 6.0], [42.0, 12.0]]))


@pytest.mark.graph
@pytest.mark.ilayers
@pytest.mark.fast
@pytest.mark.precommit
def test_max_neuron_index_layer():
    inputs = klayers.Input(shape=(6,))
    outputs = ilayers.MaxNeuronIndex()(inputs)
    model = kmodels.Model(inputs=inputs, outputs=outputs, name="TestMaxNeuronIndex")

    x = np.array([[1, 2, 3, 6, 5, 4], [7, 42, 9, 10, 11, 12]])
    out = model.predict(x)
    assert np.all(out == np.array([3, 1]))

    inputs = klayers.Input(shape=(2, 3))
    outputs = ilayers.MaxNeuronIndex()(inputs)
    model = kmodels.Model(inputs=inputs, outputs=outputs, name="TestMaxNeuronIndex")

    x = np.array([[[1, 2, 3], [6, 5, 4]], [[7, 42, 9], [10, 11, 12]]])
    out = model.predict(x)
    assert np.all(out == np.array([[2, 0], [1, 2]]))


@pytest.mark.graph
@pytest.mark.ilayers
@pytest.mark.fast
@pytest.mark.precommit
def test_neuron_selection_layer():
    neuron_selection_array = tf.constant([[0, 3], [1, 1]])
    inputs = klayers.Input(shape=(6,))
    outputs = ilayers.NeuronSelection()([inputs, neuron_selection_array])
    model = kmodels.Model(inputs=inputs, outputs=outputs, name="TestNeuronSelection")

    x = np.array([[1, 2, 3, 6, 5, 4], [7, 42, 9, 10, 11, 12]])
    out = model.predict(x)
    assert np.all(out == np.array([6, 42]))
