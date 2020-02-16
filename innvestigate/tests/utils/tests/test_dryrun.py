# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from innvestigate.utils.tests import cases
from innvestigate.utils.tests import dryrun


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST+cases.PRECOMMIT)
def test_fast__DryRunAnalyzerTestCase(case_id):
    """
    Sanity test for the TestCase.
    """

    def create_analyzer_f(output_layer):

        class TestAnalyzer(object):
            def fit(self, X):
                pass

            def analyze(self, X):
                return X

        return TestAnalyzer()

    dryrun.test_analyzer(case_id, create_analyzer_f)
