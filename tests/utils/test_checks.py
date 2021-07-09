import keras
import pytest

import innvestigate.utils.keras.checks as ichecks


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize(
    "activation,layer,expected",
    [
        ("relu", keras.layers.Dense(5, activation="relu"), True),
        ("linear", keras.layers.Dense(5, activation="linear"), True),
        ("none", keras.layers.Dense(5, activation=None), True),  # defaults to linear
        ("selu", keras.layers.Dense(5, activation="selu"), False),
        ("softmax", keras.layers.Dense(5, activation="softmax"), False),
        ("relu", keras.layers.ReLU(), True),
        ("elu", keras.layers.ELU(), False),
    ],
)
def test_only_relu_activation(activation, layer, expected):
    assert ichecks.only_relu_activation(layer) == expected


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize(
    "activation,layer,expected",
    [
        ("selu", keras.layers.Dense(5, activation="selu"), False),
        ("relu", keras.layers.Dense(5, activation="relu"), True),
        ("softmax", keras.layers.Dense(5, activation="softmax"), False),
        ("linear", keras.layers.Dense(5, activation="linear"), False),
        ("relu", keras.layers.ReLU(), True),
        ("elu", keras.layers.ELU(), False),
        ("none", keras.layers.Dense(5, activation=None), False),  # defaults to linear
    ],
)
def test_contains_activation_relu(activation, layer, expected):
    assert ichecks.contains_activation(layer, "relu") == expected


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize(
    "activation,layer,expected",
    [
        ("elu", keras.layers.Dense(5, activation="elu"), True),
        ("relu", keras.layers.Dense(5, activation="relu"), False),
        ("softmax", keras.layers.Dense(5, activation="softmax"), False),
        ("linear", keras.layers.Dense(5, activation="linear"), False),
        ("relu", keras.layers.ReLU(), False),
        ("elu", keras.layers.ELU(), True),
        ("none", keras.layers.Dense(5, activation=None), False),  # defaults to linear
    ],
)
def test_contains_activation_elu(activation, layer, expected):
    assert ichecks.contains_activation(layer, "elu") == expected


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
