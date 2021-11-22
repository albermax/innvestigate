from __future__ import annotations

import pytest

from tests import dryrun


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
