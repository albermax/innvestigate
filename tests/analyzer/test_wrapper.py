import pytest
import tensorflow as tf

from innvestigate.analyzer import (
    AugmentReduceBase,
    GaussianSmoother,
    Gradient,
    PathIntegrator,
    WrapperBase,
)

from tests import dryrun

# Dict that maps test name to tuple of method and kwargs
methods = {
    "WrapperBase": (WrapperBase, {}),
    "GaussianSmoother": (GaussianSmoother, {}),
    "PathIntegrator": (PathIntegrator, {}),
    "AugmentReduceBase": (AugmentReduceBase, {}),
}


@pytest.mark.wrapper
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_fast(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(Gradient(model), **kwargs)

    dryrun.test_analyzer(analyzer, "trivia.*:mnist.log_reg")


@pytest.mark.wrapper
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_fast_serialize(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(Gradient(model), **kwargs)

    dryrun.test_serialize_analyzer(analyzer, "trivia.*:mnist.log_reg")


@pytest.mark.wrapper
@pytest.mark.mnist
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_precommit(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(Gradient(model), **kwargs)

    dryrun.test_analyzer(analyzer, "mnist.*")
