import keras
import pytest

import innvestigate.utils.keras.checks as ichecks


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize(
    "activation,expected",
    [
        ("selu", False),
        ("relu", True),
        ("softmax", False),
        ("linear", False),
        (None, False),
    ],
)
def test_contains_activation_relu(activation, expected):
    layer = keras.layers.Dense(5, activation=activation)
    assert ichecks.contains_activation(layer, "relu") == expected


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize(
    "activation,expected",
    [
        ("selu", False),
        ("relu", False),
        ("softmax", True),
        ("linear", False),
        (None, False),
    ],
)
def test_contains_activation_softmax(activation, expected):
    layer = keras.layers.Dense(5, activation=activation)
    assert ichecks.contains_activation(layer, "softmax") == expected


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize(
    "activation,expected",
    [
        ("selu", True),
        ("relu", True),
        ("softmax", True),
        ("linear", True),
        (None, True),  # defaults to linear
    ],
)
def test_contains_activation_general(activation, expected):
    layer = keras.layers.Dense(5, activation=activation)
    assert ichecks.contains_activation(layer) == expected
