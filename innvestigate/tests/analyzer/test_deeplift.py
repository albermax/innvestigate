# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest
try:
    import deeplift
except ImportError:
    deeplift = None

from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import DeepLIFTWrapper


###############################################################################
###############################################################################
###############################################################################


require_deeplift = pytest.mark.skipif(deeplift is None,
                                      reason="Package deeplift is required.")


@require_deeplift
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.skip(reason="DeepLIFT does not work with skip connection.")
def test_fast__DeepLIFTWrapper():

    def method(model):
        return DeepLIFTWrapper(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@require_deeplift
@pytest.mark.precommit
def test_precommit__DeepLIFTWrapper():

    def method(model):
        return DeepLIFTWrapper(model)

    dryrun.test_analyzer(method, "mnist.*")


@require_deeplift
@pytest.mark.precommit
def test_precommit__DeepLIFTWrapper_neuron_selection_index():

    class CustomAnalyzer(DeepLIFTWrapper):

        def analyze(self, X):
            index = 0
            return super(CustomAnalyzer, self).analyze(X, index)

    def method(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(method, "mnist.*")


@require_deeplift
@pytest.mark.precommit
def test_precommit__DeepLIFTWrapper_larger_batch_size():

    class CustomAnalyzer(DeepLIFTWrapper):

        def analyze(self, X):
            X = np.concatenate((X, X), axis=0)
            return super(CustomAnalyzer, self).analyze(X)[0:1]

    def method(model):
        return CustomAnalyzer(model)

    dryrun.test_analyzer(method, "mnist.*")


@require_deeplift
@pytest.mark.precommit
def test_precommit__DeepLIFTWrapper_larger_batch_size_with_index():

    class CustomAnalyzer(DeepLIFTWrapper):

        def analyze(self, X):
            index = 0
            X = np.concatenate((X, X), axis=0)
            return super(CustomAnalyzer, self).analyze(X, index)[0:1]

    def method(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(method, "mnist.*")


@require_deeplift
@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__DeepLIFTWrapper():

    def method(model):
        return DeepLIFTWrapper(model)

    dryrun.test_analyzer(method, "imagenet.*")
