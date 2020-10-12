# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import numpy as np
import pytest
try:
    import deeplift
except ImportError:
    deeplift = None

from innvestigate import backend
from innvestigate.utils.tests import cases
from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import DeepLIFTWrapper


###############################################################################
###############################################################################
###############################################################################


# Cases that are known to not work with deeplift package:
DEEPLIFT_EXPECTED_FAILURES = [
    "skip_connection",
    "cnn_3dim_c1_d1",
    "cnn_3dim_c2_d1",
    "lc_cnn_1dim_c1_d1",
    "lc_cnn_1dim_c2_d1",
    "lc_cnn_2dim_c1_d1",
    "lc_cnn_2dim_c2_d1",
]


def mark_xfails(case_ids):
    return cases.mark_as_xfail(case_ids, DEEPLIFT_EXPECTED_FAILURES)


require_deeplift = pytest.mark.skipif(deeplift is None,
                                      reason="Package deeplift is required.")
require_tf = pytest.mark.skipif(backend.name() != "tensorflow",
                                reason="Package deeplift requires tensorflow.")


###############################################################################
###############################################################################
###############################################################################


@require_deeplift
@require_tf
@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", mark_xfails(cases.FAST))
def test_fast__DeepLIFTWrapper(case_id):

    def create_analyzer_f(model):
        return DeepLIFTWrapper(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@require_deeplift
@require_tf
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", mark_xfails(cases.PRECOMMIT))
def test_precommit__DeepLIFTWrapper(case_id):

    def create_analyzer_f(model):
        return DeepLIFTWrapper(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@require_deeplift
@require_tf
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", mark_xfails(cases.FAST+cases.PRECOMMIT))
def atest_precommit__DeepLIFTWrapper_neuron_selection_index(case_id):

    class CustomAnalyzer(DeepLIFTWrapper):

        def analyze(self, X):
            index = 0
            return super(CustomAnalyzer, self).analyze(X, index)

    def create_analyzer_f(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(case_id, create_analyzer_f)


@require_deeplift
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", mark_xfails(cases.FAST+cases.PRECOMMIT))
def atest_precommit__DeepLIFTWrapper_larger_batch_size(case_id):

    class CustomAnalyzer(DeepLIFTWrapper):

        def analyze(self, X):
            X = np.concatenate((X, X), axis=0)
            return super(CustomAnalyzer, self).analyze(X)[0:1]

    def create_analyzer_f(model):
        return CustomAnalyzer(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@require_deeplift
@require_tf
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", mark_xfails(cases.FAST+cases.PRECOMMIT))
def atest_precommit__DeepLIFTWrapper_larger_batch_size_with_index(case_id):

    class CustomAnalyzer(DeepLIFTWrapper):

        def analyze(self, X):
            index = 0
            X = np.concatenate((X, X), axis=0)
            return super(CustomAnalyzer, self).analyze(X, index)[0:1]

    def create_analyzer_f(model):
        return CustomAnalyzer(model, neuron_selection_mode="index")

    dryrun.test_analyzer(case_id, create_analyzer_f)
