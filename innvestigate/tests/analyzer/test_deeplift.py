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


from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import DeepLIFTCore


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__DeepLIFTCore():

    def method(model):
        return DeepLIFTCore(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__DeepLIFTCore():

    def method(model):
        return DeepLIFTCore(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.precommit
def test_precommit__DeepLIFTCore_neuron_selection_index():

    class CustomAnalyzer(DeepLIFTCore):

        def analyze(self, X):
            index = 0
            return super(CustomAnalyzer, self).analyze(X, index)

    def method(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__DeepLIFTCore():

    def method(model):
        return DeepLIFTCore(model)

    dryrun.test_analyzer(method, "imagenet.*")
