# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from innvestigate.utils.tests import cases
from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import Input
from innvestigate.analyzer import Random


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST+cases.PRECOMMIT)
def test_fast__Input(case_id):

    def create_analyzer_f(model):
        return Input(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST+cases.PRECOMMIT)
def test_fast__Random(case_id):

    def create_analyzer_f(model):
        return Random(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)
