# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from innvestigate.utils.tests import cases
from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import BaselineGradient
from innvestigate.analyzer import Gradient

from innvestigate.analyzer import InputTimesGradient

from innvestigate.analyzer import Deconvnet
from innvestigate.analyzer import GuidedBackprop

from innvestigate.analyzer import IntegratedGradients

from innvestigate.analyzer import SmoothGrad


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__BaselineGradient(case_id):

    def create_analyzer_f(model):
        return BaselineGradient(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__BaselineGradient(case_id):

    def create_analyzer_f(model):
        return BaselineGradient(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__BaselineGradient_pp_None(case_id):

    def create_analyzer_f(model):
        return BaselineGradient(model, postprocess=None)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__BaselineGradient_pp_None(case_id):

    def create_analyzer_f(model):
        return BaselineGradient(model, postprocess=None)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__BaselineGradient_pp_square(case_id):

    def create_analyzer_f(model):
        return BaselineGradient(model, postprocess="square")

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__BaselineGradient_pp_square(case_id):

    def create_analyzer_f(model):
        return BaselineGradient(model, postprocess="square")

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__Gradient(case_id):

    def create_analyzer_f(model):
        return Gradient(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__Gradient(case_id):

    def create_analyzer_f(model):
        return Gradient(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__Gradient_pp_None(case_id):

    def create_analyzer_f(model):
        return Gradient(model, postprocess=None)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__Gradient_pp_None(case_id):

    def create_analyzer_f(model):
        return Gradient(model, postprocess=None)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__Gradient_pp_square(case_id):

    def create_analyzer_f(model):
        return Gradient(model, postprocess="square")

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__Gradient_pp_square(case_id):

    def create_analyzer_f(model):
        return Gradient(model, postprocess="square")

    dryrun.test_analyzer(case_id, create_analyzer_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__InputTimesGradient(case_id):

    def create_analyzer_f(model):
        return InputTimesGradient(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__InputTimesGradient(case_id):

    def create_analyzer_f(model):
        return InputTimesGradient(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__Deconvnet(case_id):

    def create_analyzer_f(model):
        return Deconvnet(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__Deconvnet(case_id):

    def create_analyzer_f(model):
        return Deconvnet(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__GuidedBackprop(case_id):

    def create_analyzer_f(model):
        return GuidedBackprop(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__GuidedBackprop(case_id):

    def create_analyzer_f(model):
        return GuidedBackprop(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__IntegratedGradients(case_id):

    def create_analyzer_f(model):
        return IntegratedGradients(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__IntegratedGradients(case_id):

    def create_analyzer_f(model):
        return IntegratedGradients(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__SmoothGrad(case_id):

    def create_analyzer_f(model):
        return SmoothGrad(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_precommit__SmoothGrad(case_id):

    def create_analyzer_f(model):
        return SmoothGrad(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)
