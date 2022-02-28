from __future__ import annotations

import pytest
import tensorflow as tf

from innvestigate.analyzer import BaselineGradient, Gradient, Input, Random

from tests import dryrun


class CustomAnalyzerIndex0(Gradient):
    def analyze(self, X):
        return super().analyze(X, neuron_selection=0)


class CustomAnalyzerIndex3(Gradient):
    def analyze(self, X):
        return super().analyze(X, neuron_selection=3)


methods_serializable = {
    "Input": (Input, {}),
    "Random": (Random, {}),
    "AnalyzerNetworkBase_neuron_selection_max": (
        Gradient,
        {"neuron_selection_mode": "max_activation"},
    ),
    "BaseReverseNetwork_reverse_debug": (Gradient, {"reverse_verbose": True}),
    "BaseReverseNetwork_reverse_check_minmax": (
        Gradient,
        {"reverse_verbose": True, "reverse_check_min_max_values": True},
    ),
    "BaseReverseNetwork_reverse_check_finite": (
        Gradient,
        {"reverse_verbose": True, "reverse_check_finite": True},
    ),
    "Gradient": (Gradient, {}),
    "BaselineGradient": (BaselineGradient, {}),
}

# TODO: Custom methods currently cannot be serialized as the process requires
# the class name to be known by iNNvestigate.
methods = methods_serializable.copy()
methods.update(
    {
        "AnalyzerNetworkBase_neuron_selection_index_0": (
            CustomAnalyzerIndex0,
            {"neuron_selection_mode": "index"},
        ),
        "AnalyzerNetworkBase_neuron_selection_index_3": (
            CustomAnalyzerIndex3,
            {"neuron_selection_mode": "index"},
        ),
    }
)


# Dryrun all methods
@pytest.mark.base
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize(
    "method, kwargs",
    methods_serializable.values(),
    ids=list(methods_serializable.keys()),
)
def test_fast(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "trivia.*:mnist.log_reg")


@pytest.mark.base
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize(
    "method, kwargs",
    methods_serializable.values(),
    ids=list(methods_serializable.keys()),
)
def test_fast_serialize(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_serialize_analyzer(analyzer, "trivia.*:mnist.log_reg")


@pytest.mark.base
@pytest.mark.mnist
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_precommit(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "mnist.*")


#######


@pytest.mark.base
@pytest.mark.fast
@pytest.mark.precommit
def test_fast__BasicGraphReversal():
    tf.keras.backend.clear_session()

    def method1(model):
        return BaselineGradient(model)

    def method2(model):
        return Gradient(model)

    dryrun.test_equal_analyzer(method1, method2, "trivia.*:mnist.log_reg")


@pytest.mark.base
@pytest.mark.mnist
@pytest.mark.precommit
def test_precommit__BasicGraphReversal():
    tf.keras.backend.clear_session()

    def method1(model):
        return BaselineGradient(model)

    def method2(model):
        return Gradient(model)

    dryrun.test_equal_analyzer(method1, method2, "mnist.*")
