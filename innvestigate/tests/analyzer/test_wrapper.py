# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from innvestigate.utils.tests import cases
from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import WrapperBase
from innvestigate.analyzer import AugmentReduceBase
from innvestigate.analyzer import GaussianSmoother
from innvestigate.analyzer import PathIntegrator

from innvestigate.analyzer import Gradient


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__WrapperBase(case_id):

    def create_analyzer_f(model):
        return WrapperBase(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__WrapperBase(case_id):

    def create_analyzer_f(model):
        return WrapperBase(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__AugmentReduceBase(case_id):

    def create_analyzer_f(model):
        return AugmentReduceBase(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__AugmentReduceBase(case_id):

    def create_analyzer_f(model):
        return AugmentReduceBase(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__GaussianSmoother(case_id):

    def create_analyzer_f(model):
        return GaussianSmoother(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__GaussianSmoother(case_id):

    def create_analyzer_f(model):
        return GaussianSmoother(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__PathIntegrator(case_id):

    def create_analyzer_f(model):
        return PathIntegrator(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__PathIntegrator(case_id):

    def create_analyzer_f(model):
        return PathIntegrator(Gradient(model))

    dryrun.test_analyzer(case_id, create_analyzer_f)
