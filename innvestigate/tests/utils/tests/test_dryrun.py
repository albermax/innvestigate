# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from innvestigate.utils.tests import dryrun


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__DryRunAnalyzerTestCase():
    """
    Sanity test for the TestCase.
    """

    def method(output_layer):

        class TestAnalyzer(object):
            def analyze(self, X):
                return X

        return TestAnalyzer()

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")
