import pytest

from innvestigate.analyzer import PatternAttribution, PatternNet

from tests import dryrun


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_fast__PatternNet():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_analyzer(method, "mnist.log_reg")


@pytest.mark.mnist
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_precommit__PatternNet():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
@pytest.mark.pattern_based
def test_imagenet__PatternNet():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_analyzer(method, "imagenet.vgg16:imagenet.vgg19")


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_fast__PatternAttribution():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternAttribution(model, patterns=patterns)

    dryrun.test_analyzer(method, "mnist.log_reg")


@pytest.mark.mnist
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_precommit__PatternAttribution():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternAttribution(model, patterns=patterns)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
@pytest.mark.pattern_based
def test_imagenet__PatternAttribution():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternAttribution(model, patterns=patterns)

    dryrun.test_analyzer(method, "imagenet.vgg16:imagenet.vgg19")


###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.pattern_based
def test_fast__SerializePatternNet():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_serialize_analyzer(method, "mnist.log_reg")
