import pytest
import tensorflow as tf

from innvestigate.analyzer import (
    LRPZ,
    BaselineLRPZ,
    LRPAlpha1Beta0,
    LRPAlpha1Beta0IgnoreBias,
    LRPAlpha2Beta1,
    LRPAlpha2Beta1IgnoreBias,
    LRPEpsilon,
    LRPFlat,
    LRPSequentialPresetA,
    LRPSequentialPresetAFlat,
    LRPSequentialPresetB,
    LRPSequentialPresetBFlat,
    LRPWSquare,
    LRPZPlus,
    LRPZPlusFast,
)

from tests import dryrun

methods = {
    "LRPZ": (LRPZ, {}),
    "LRPZ_Flat_input_layer_rule": (LRPZ, {"input_layer_rule": "Flat"}),
    "LRPZ_boxed_input_layer_rule": (LRPZ, {"input_layer_rule": (-10, 10)}),
    "LRPZPlus": (LRPZPlus, {}),
    "LRPZPlusFast": (LRPZPlusFast, {}),
    "BaselineLRPZ": (BaselineLRPZ, {}),
    "LRPAlpha1Beta0": (LRPAlpha1Beta0, {}),
    "LRPAlpha1Beta0IgnoreBias": (LRPAlpha1Beta0IgnoreBias, {}),
    "LRPAlpha2Beta1": (LRPAlpha2Beta1, {}),
    "LRPAlpha2Beta1IgnoreBias": (LRPAlpha2Beta1IgnoreBias, {}),
    "LRPEpsilon": (LRPEpsilon, {}),
    "LRPFlat": (LRPFlat, {}),
    "LRPWSquare": (LRPWSquare, {}),
    "LRPSequentialPresetA": (LRPSequentialPresetA, {}),
    "LRPSequentialPresetB": (LRPSequentialPresetB, {}),
    "LRPSequentialPresetAFlat": (LRPSequentialPresetAFlat, {}),
    "LRPSequentialPresetBFlat": (LRPSequentialPresetBFlat, {}),
}


@pytest.mark.lrp
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_fast(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "trivia.*:mnist.log_reg")


@pytest.mark.lrp
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_fast_serialize(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_serialize_analyzer(analyzer, "trivia.*:mnist.log_reg")


@pytest.mark.lrp
@pytest.mark.mnist
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_precommit(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "mnist.*")


@pytest.mark.lrp
@pytest.mark.resnet50
@pytest.mark.precommit
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_precommit_resnet50(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "imagenet.resnet50")


@pytest.mark.lrp
@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
@pytest.mark.parametrize("method, kwargs", methods.values(), ids=list(methods.keys()))
def test_imagenet(method, kwargs):
    tf.keras.backend.clear_session()

    def analyzer(model):
        return method(model, **kwargs)

    dryrun.test_analyzer(analyzer, "imagenet.*")


###############################################################################


@pytest.mark.lrp
@pytest.mark.fast
@pytest.mark.precommit
def test_fast_LRPZ_equal_BaselineLRPZ():
    tf.keras.backend.clear_session()

    def method1(model):
        return BaselineLRPZ(model)

    def method2(model):
        # LRP-Z with bias
        return LRPZ(model)

    dryrun.test_equal_analyzer(
        method1,
        method2,
        # mind this only works for networks with relu, max, activations
        # and no skip connections!
        "trivia.dot:mnist.log_reg",
    )
