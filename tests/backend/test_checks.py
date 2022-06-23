import pytest
import tensorflow.keras.layers as klayers

import innvestigate.backend.checks as ichecks


@pytest.mark.graph
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize(
    "activation,layer,expected",
    [
        ("relu", klayers.Dense(5, activation="relu"), True),
        ("linear", klayers.Dense(5, activation="linear"), True),
        ("none", klayers.Dense(5, activation=None), True),  # defaults to linear
        ("selu", klayers.Dense(5, activation="selu"), False),
        ("softmax", klayers.Dense(5, activation="softmax"), False),
        ("relu", klayers.ReLU(), True),
        ("elu", klayers.ELU(), False),
    ],
)
def test_only_relu_activation(activation, layer, expected):
    assert ichecks.only_relu_activation(layer) == expected


@pytest.mark.graph
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize(
    "activation,layer,expected",
    [
        ("selu", klayers.Dense(5, activation="selu"), False),
        ("relu", klayers.Dense(5, activation="relu"), True),
        ("softmax", klayers.Dense(5, activation="softmax"), False),
        ("linear", klayers.Dense(5, activation="linear"), False),
        ("relu", klayers.ReLU(), True),
        ("elu", klayers.ELU(), False),
        ("none", klayers.Dense(5, activation=None), False),  # defaults to linear
    ],
)
def test_contains_activation_relu(activation, layer, expected):
    assert ichecks.contains_activation(layer, "relu") == expected


@pytest.mark.graph
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize(
    "activation,layer,expected",
    [
        ("elu", klayers.Dense(5, activation="elu"), True),
        ("relu", klayers.Dense(5, activation="relu"), False),
        ("softmax", klayers.Dense(5, activation="softmax"), False),
        ("linear", klayers.Dense(5, activation="linear"), False),
        ("relu", klayers.ReLU(), False),
        ("elu", klayers.ELU(), True),
        ("none", klayers.Dense(5, activation=None), False),  # defaults to linear
    ],
)
def test_contains_activation_elu(activation, layer, expected):
    assert ichecks.contains_activation(layer, "elu") == expected


@pytest.mark.graph
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
    layer = klayers.Dense(5, activation=activation)
    assert ichecks.contains_activation(layer, "softmax") == expected


@pytest.mark.graph
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
    layer = klayers.Dense(5, activation=activation)
    assert ichecks.contains_activation(layer) == expected
