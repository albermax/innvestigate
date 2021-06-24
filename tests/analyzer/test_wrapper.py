# Get Python six functionality:
from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from innvestigate.analyzer import (
    AugmentReduceBase,
    GaussianSmoother,
    Gradient,
    PathIntegrator,
    WrapperBase,
)

from tests import dryrun


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


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__AugmentReduceBase():
    def method(model):
        return AugmentReduceBase(Gradient(model))

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__AugmentReduceBase():
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


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__GaussianSmoother():
    def method(model):
        return GaussianSmoother(Gradient(model))

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__GaussianSmoother():
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


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PathIntegrator():
    def method(model):
        return PathIntegrator(Gradient(model))

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__PathIntegrator():
    def method(model):
        return PathIntegrator(Gradient(model))

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__SerializePathIntegrator():
    def method(model):
        return PathIntegrator(Gradient(model))

    dryrun.test_serialize_analyzer(method, "trivia.*:mnist.log_reg")
