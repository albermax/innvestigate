import pytest
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

from innvestigate.analyzer.relevance_based.relevance_analyzer import LRP


@pytest.mark.graph
@pytest.mark.fast
@pytest.mark.lrp
def test_composite_lrp():
    model = kmodels.Sequential(
        [
            klayers.Input(shape=(28, 28, 3)),
            klayers.Conv2D(8, 3, activation="relu"),
            klayers.Conv2D(4, 5, activation="relu"),
            klayers.Flatten(),
            klayers.Dense(16, activation="relu"),
            klayers.Dense(2, activation="softmax"),
        ]
    )
    analyzer = LRP(
        model,
        rule="Z",
        input_layer_rule="Flat",
        until_layer_idx=2,
        until_layer_rule="Epsilon",
    )
    correct_rules = [
        "Flat",
        "Epsilon",
        "Epsilon",
        "Z",
        "Z",
    ]  # Correct rules corresponding to analyzer input args

    for i, layer in enumerate(model.layers):
        for condition, rule in analyzer._rules:
            if condition(layer):
                rule_class = rule
                break
        assert rule_class == correct_rules[i]
