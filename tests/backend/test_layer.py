from __future__ import annotations

import numpy as np
import pytest
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

import innvestigate.layers as ilayers
from innvestigate.analyzer.gradient_based import Gradient


@pytest.mark.ilayers
@pytest.mark.fast
@pytest.mark.precommit
def test_repeat_layer():
    inputs = klayers.Input(shape=(2, 3))
    outputs = ilayers.Repeat(8)(inputs)
    model = kmodels.Model(inputs=inputs, outputs=outputs, name="TestRepeat")

    x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    out = model.predict(x)

    assert np.shape(out) == (2, 8, 2, 3)
    assert np.all(
        out
        == np.array(
            [
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                ],
                [
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
            ]
        )
    )


@pytest.mark.ilayers
@pytest.mark.fast
@pytest.mark.precommit
def test_reducemean_layer():
    inputs = klayers.Input(shape=(2, 3))
    repeated = ilayers.Repeat(8)(inputs)
    outputs = ilayers.ReduceMean()(repeated)
    model = kmodels.Model(inputs=inputs, outputs=outputs, name="TestRepeat")

    x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    out = model.predict(x)

    assert np.all(out == x)


@pytest.mark.graph
@pytest.mark.fast
@pytest.mark.precommit
def test_fast_one_layer():
    model = kmodels.Sequential([klayers.Dense(2, input_shape=(3,), use_bias=False)])
    weights = [np.asarray(((1, 2), (3, 4), (5, 6)))]
    model.set_weights(weights)

    inputs = np.asarray((1, 2, 3)).reshape((1, 3))
    outputs = model.predict_on_batch(inputs)

    analyzer = Gradient(model)
    analysis = analyzer.analyze(inputs)

    # Analyzer takes node with max output.
    i = np.argmax(outputs)
    gradient = np.dot(weights[0][:, i], np.ones_like(outputs[0][i]))
    assert np.allclose(analysis, gradient)


@pytest.mark.graph
@pytest.mark.fast
@pytest.mark.precommit
def test_fast_two_layers():
    model = kmodels.Sequential(
        [
            klayers.Dense(2, input_shape=(3,), use_bias=False),
            klayers.Dense(2, use_bias=False),
        ]
    )
    weights = [np.asarray(((1, 2), (3, 4), (5, 6))), np.asarray(((7, 8), (9, 1)))]
    model.set_weights(weights)

    inputs = np.asarray((1, 2, 3)).reshape((1, 3))
    outputs = model.predict_on_batch(inputs)

    analyzer = Gradient(model)
    analysis = analyzer.analyze(inputs)

    # Analyzer takes node with max output.
    i = np.argmax(outputs)
    gradient_middle = np.dot(weights[1][:, i], np.ones_like(outputs[0][i]))
    gradient = np.dot(weights[0], gradient_middle)
    assert np.allclose(analysis, gradient)
