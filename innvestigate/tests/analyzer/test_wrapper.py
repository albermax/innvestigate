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

from innvestigate.analyzer import WrapperBase
from innvestigate.analyzer import AugmentReduceBase
from innvestigate.analyzer import GaussianSmoother
from innvestigate.analyzer import PathIntegrator

from innvestigate.analyzer import Input
from innvestigate.analyzer import Gradient


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__WrapperBase():

    def method(model):
        return WrapperBase(Gradient(model))

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__WrapperBase():

    def method(model):
        return WrapperBase(Gradient(model))

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__SerializeWrapperBase():

    def method(model):
        return WrapperBase(Gradient(model))

    dryrun.test_serialize_analyzer(method, "trivia.*:mnist.log_reg")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.skip("Deprecated feature.")
@pytest.mark.fast
@pytest.mark.precommit
def test_fast__AugmentReduceBase__python_based():

    def method(model):
        return AugmentReduceBase(Input(model))

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__AugmentReduceBase__keras_based():

    def method(model):
        return AugmentReduceBase(Gradient(model))

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__AugmentReduceBase__keras_based():

    def method(model):
        return AugmentReduceBase(Gradient(model))

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__SerializeAugmentReduceBase():

    def method(model):
        return AugmentReduceBase(Gradient(model))

    dryrun.test_serialize_analyzer(method, "trivia.*:mnist.log_reg")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.skip("Deprecated feature.")
@pytest.mark.fast
@pytest.mark.precommit
def test_fast__GaussianSmoother__python_based():

    def method(model):
        return GaussianSmoother(Input(model))

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__GaussianSmoother__keras_based():

    def method(model):
        return GaussianSmoother(Gradient(model))

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__GaussianSmoother__keras_based():

    def method(model):
        return GaussianSmoother(Gradient(model))

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__SerializeGaussianSmoother():

    def method(model):
        return GaussianSmoother(Gradient(model))

    dryrun.test_serialize_analyzer(method, "trivia.*:mnist.log_reg")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.skip("Deprecated feature.")
@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PathIntegrator__python_based():

    def method(model):
        return PathIntegrator(Input(model))

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PathIntegrator__keras_based():

    def method(model):
        return PathIntegrator(Gradient(model))

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__PathIntegrator__keras_based():

    def method(model):
        return PathIntegrator(Gradient(model))

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__SerializePathIntegrator():

    def method(model):
        return PathIntegrator(Gradient(model))

    dryrun.test_serialize_analyzer(method, "trivia.*:mnist.log_reg")
