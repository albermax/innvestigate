# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import pytest
try:
    import deeplift
except ImportError:
    deeplift = None

from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import DeepLIFT
from innvestigate.analyzer import DeepLIFTWrapper


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__DeepLIFT():

    def method(model):
        return DeepLIFT(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__DeepLIFT():

    def method(model):
        return DeepLIFT(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.precommit
def test_precommit__DeepLIFT_neuron_selection_index():

    class CustomAnalyzer(DeepLIFT):

        def analyze(self, X):
            index = 0
            return super(CustomAnalyzer, self).analyze(X, index)

    def method(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__DeepLIFT():

    def method(model):
        return DeepLIFT(model)

    dryrun.test_analyzer(method, "imagenet.*")


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
@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__DeepLIFTWrapper():

    def method(model):
        return DeepLIFTWrapper(model)

    dryrun.test_analyzer(method, "imagenet.*")
