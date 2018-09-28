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

from innvestigate.analyzer import *


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__BaselineLRPZ():

    def method(model):
        return BaselineLRPZ(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPZ():

    def method(model):
        return LRPZ(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPZ__equal_BaselineLRPZ():

    _all_close_rtol = 10**-5
    _all_close_atol = 10**-5

    def method1(model):
        return BaselineLRPZ(model)

    def method2(model):
        # LRP-Z with bias
        return LRPZ(model)

    dryrun.test_equal_analyzer(method1,
                                      method2,
                                      # mind this only works for
                                      # networks with relu, max,
                                      # activations and no
                                      # skip connections!
                                      "trivia.dot:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPZ__with_input_layer_rule():

    def method(model):
        return LRPZ(model, input_layer_rule="Flat")

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPZ__with_boxed_input_layer_rule():

    def method(model):
        return LRPZ(model, input_layer_rule=(-10, 10))

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPZIgnoreBias():

    def method(model):
        return LRPZIgnoreBias(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPZPlus():

    def method(model):
        return LRPZPlus(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPEpsilon():

    def method(model):
        return LRPEpsilon(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPEpsilonIgnoreBias():

    def method(model):
        return LRPEpsilonIgnoreBias(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPWSquare():

    def method(model):
        return LRPWSquare(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPFlat():

    def method(model):
        return LRPFlat(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPAlpha2Beta1():

    def method(model):
        return LRPAlpha2Beta1(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPAlpha2Beta1IgnoreBias():

    def method(model):
        return LRPAlpha2Beta1IgnoreBias(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPAlpha1Beta0():

    def method(model):
        return LRPAlpha1Beta0(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__LRPAlpha1Beta0IgnoreBias():

    def method(model):
        return LRPAlpha1Beta0IgnoreBias(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__DeepTaylor():

    def method(model):
        return DeepTaylor(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__BoundedDeepTaylor():

    def method(model):
        return BoundedDeepTaylor(model, low=-1, high=1)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__SerializeLRPZ():

    def method(model):
        return LRPZ(model)

    dryrun.test_serialize_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__SerializeLRPAlpha2Beta1():

    def method(model):
        return LRPAlpha2Beta1(model)

    dryrun.test_serialize_analyzer(method, "trivia.*:mnist.log_reg")
