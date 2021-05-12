# Get Python six functionality:
from __future__ import absolute_import, division, print_function, unicode_literals

import keras.layers
import numpy as np
import pytest

from innvestigate.analyzer.gradient_based import Gradient

# Prevent pytest from collecting this class:
from tests.layer import TestAnalysisHelper as AnalysisHelper


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__TestAnalysisHelper_one_layer():

    layer = keras.layers.Dense(2, input_shape=(3,), use_bias=False)
    analyzer = Gradient
    weights = [np.asarray(((1, 2), (3, 4), (5, 6)))]

    helper = AnalysisHelper(layer, analyzer, weights)

    inputs = np.asarray((1, 2, 3))
    outputs, analysis = helper.run(inputs)

    # Analyzer takes node with max output.
    i = np.argmax(outputs)
    gradient = np.dot(weights[0][:, i], np.ones_like(outputs[i]))
    assert np.allclose(analysis, gradient)


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__TestAnalysisHelper_two_layers():

    layers = [
        keras.layers.Dense(2, input_shape=(3,), use_bias=False),
        keras.layers.Dense(2, use_bias=False),
    ]
    analyzer = Gradient
    weights = [np.asarray(((1, 2), (3, 4), (5, 6))), np.asarray(((7, 8), (9, 1)))]

    helper = AnalysisHelper(layers, analyzer, weights)

    inputs = np.asarray((1, 2, 3))
    outputs, analysis = helper.run(inputs)

    # Analyzer takes node with max output.
    i = np.argmax(outputs)
    gradient_middle = np.dot(weights[1][:, i], np.ones_like(outputs[i]))
    gradient = np.dot(weights[0], gradient_middle)
    assert np.allclose(analysis, gradient)
