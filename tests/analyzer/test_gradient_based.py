import pytest
import tensorflow as tf

from innvestigate.analyzer import (
    BaselineGradient,
    Deconvnet,
    Gradient,
    GuidedBackprop,
    InputTimesGradient,
    IntegratedGradients,
    SmoothGrad,
)

from tests import dryrun

# Dict that maps test name to tuple of method and kwargs
methods = {
    "Gradient": (Gradient, {}),
    "Gradient_pp_None": (Gradient, {"postprocess": None}),
    "Gradient_pp_square": (Gradient, {"postprocess": "square"}),
    "BaselineGradient": (BaselineGradient, {}),
    "BaselineGradient_pp_None": (BaselineGradient, {"postprocess": None}),
    "BaselineGradient_pp_square": (BaselineGradient, {"postprocess": "square"}),
    "InputTimesGradient": (InputTimesGradient, {}),
    "Deconvnet": (Deconvnet, {}),
    "GuidedBackprop": (GuidedBackprop, {}),
    "IntegratedGradients": (IntegratedGradients, {}),
    "SmoothGrad": (SmoothGrad, {}),
}


@pytest.mark.gradient
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_fast(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "trivia.*:mnist.log_reg")


@pytest.mark.gradient
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_fast_serialize(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_serialize_analyzer(analyzer, "trivia.*:mnist.log_reg")


@pytest.mark.gradient
@pytest.mark.mnist
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_precommit(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "mnist.*")


@pytest.mark.gradient
@pytest.mark.resnet50
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_precommit_resnet50(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "imagenet.resnet50")


@pytest.mark.gradient
@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_imagenet(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "imagenet.*")
