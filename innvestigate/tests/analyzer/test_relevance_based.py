# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import pytest


from innvestigate.utils.tests import cases
from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import BaselineLRPZ
from innvestigate.analyzer import LRPZ
from innvestigate.analyzer import LRPZIgnoreBias
from innvestigate.analyzer import LRPZPlus
from innvestigate.analyzer import LRPZPlusFast
from innvestigate.analyzer import LRPEpsilon
from innvestigate.analyzer import LRPEpsilonIgnoreBias
from innvestigate.analyzer import LRPWSquare
from innvestigate.analyzer import LRPFlat
from innvestigate.analyzer import LRPAlpha2Beta1
from innvestigate.analyzer import LRPAlpha2Beta1IgnoreBias
from innvestigate.analyzer import LRPAlpha1Beta0
from innvestigate.analyzer import LRPAlpha1Beta0IgnoreBias


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__BaselineLRPZ(case_id):

    def create_analyzer_f(model):
        return BaselineLRPZ(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__BaselineLRPZ(case_id):

    def create_analyzer_f(model):
        return BaselineLRPZ(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPZ(case_id):

    def create_analyzer_f(model):
        return LRPZ(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPZ(case_id):

    def create_analyzer_f(model):
        return LRPZ(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
# mind this only works for
# networks with relu, max,
# activations and no
# skip connections!
@pytest.mark.parametrize(
    "case_id", cases.filter(cases.FAST, ["skip_connection"]))
def test_fast__LRPZ__equal_BaselineLRPZ(case_id):

    def create_analyzer1_f(model):
        return BaselineLRPZ(model)

    def create_analyzer2_f(model):
        # LRP-Z with bias
        return LRPZ(model)

    dryrun.test_analyzers_for_same_output(
        case_id, create_analyzer1_f, create_analyzer2_f)


@pytest.mark.precommit
# mind this only works for
# networks with relu, max,
# activations and no
# skip connections!
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPZ__equal_BaselineLRPZ(case_id):

    def create_analyzer1_f(model):
        return BaselineLRPZ(model)

    def create_analyzer2_f(model):
        # LRP-Z with bias
        return LRPZ(model)

    dryrun.test_analyzers_for_same_output(
        case_id, create_analyzer1_f, create_analyzer2_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPZ__with_input_layer_rule(case_id):

    def create_analyzer_f(model):
        return LRPZ(model, input_layer_rule="Flat")

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPZ__with_input_layer_rule(case_id):

    def create_analyzer_f(model):
        return LRPZ(model, input_layer_rule="Flat")

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPZ__with_boxed_input_layer_rule(case_id):

    def create_analyzer_f(model):
        return LRPZ(model, input_layer_rule=(-10, 10))

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPZ__with_boxed_input_layer_rule(case_id):

    def create_analyzer_f(model):
        return LRPZ(model, input_layer_rule=(-10, 10))

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPZIgnoreBias(case_id):

    def create_analyzer_f(model):
        return LRPZIgnoreBias(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPZIgnoreBias(case_id):

    def create_analyzer_f(model):
        return LRPZIgnoreBias(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPZPlus(case_id):

    def create_analyzer_f(model):
        return LRPZPlus(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPZPlus(case_id):

    def create_analyzer_f(model):
        return LRPZPlus(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPZPlusFast(case_id):

    def create_analyzer_f(model):
        return LRPZPlusFast(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPZPlusFast(case_id):

    def create_analyzer_f(model):
        return LRPZPlusFast(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPEpsilon(case_id):

    def create_analyzer_f(model):
        return LRPEpsilon(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPEpsilon(case_id):

    def create_analyzer_f(model):
        return LRPEpsilon(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPEpsilonIgnoreBias(case_id):

    def create_analyzer_f(model):
        return LRPEpsilonIgnoreBias(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPEpsilonIgnoreBias(case_id):

    def create_analyzer_f(model):
        return LRPEpsilonIgnoreBias(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPWSquare(case_id):

    def create_analyzer_f(model):
        return LRPWSquare(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPWSquare(case_id):

    def create_analyzer_f(model):
        return LRPWSquare(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPFlat(case_id):

    def create_analyzer_f(model):
        return LRPFlat(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPFlat(case_id):

    def create_analyzer_f(model):
        return LRPFlat(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPAlpha2Beta1(case_id):

    def create_analyzer_f(model):
        return LRPAlpha2Beta1(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPAlpha2Beta1(case_id):

    def create_analyzer_f(model):
        return LRPAlpha2Beta1(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPAlpha2Beta1IgnoreBias(case_id):

    def create_analyzer_f(model):
        return LRPAlpha2Beta1IgnoreBias(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPAlpha2Beta1IgnoreBias(case_id):

    def create_analyzer_f(model):
        return LRPAlpha2Beta1IgnoreBias(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPAlpha1Beta0(case_id):

    def create_analyzer_f(model):
        return LRPAlpha1Beta0(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPAlpha1Beta0(case_id):

    def create_analyzer_f(model):
        return LRPAlpha1Beta0(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.fast
@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.FAST)
def test_fast__LRPAlpha1Beta0IgnoreBias(case_id):

    def create_analyzer_f(model):
        return LRPAlpha1Beta0IgnoreBias(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)


@pytest.mark.precommit
@pytest.mark.parametrize("case_id", cases.PRECOMMIT)
def test_fast__LRPAlpha1Beta0IgnoreBias(case_id):

    def create_analyzer_f(model):
        return LRPAlpha1Beta0IgnoreBias(model)

    dryrun.test_analyzer(case_id, create_analyzer_f)
