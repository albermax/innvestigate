# Get Python six functionality:
from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from innvestigate.analyzer import (
    BaselineGradient,
    Deconvnet,
    Gradient,
    GuidedBackprop,
    InputTimesGradient,
    IntegratedGradients,
    SmoothGrad,
)

from tests import dryrun


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__BaselineGradient():
    def method(model):
        return BaselineGradient(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__BaselineGradient():
    def method(model):
        return BaselineGradient(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__BaselineGradient():
    def method(model):
        return BaselineGradient(model)

    dryrun.test_analyzer(method, "imagenet.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__BaselineGradient_pp_None():
    def method(model):
        return BaselineGradient(model, postprocess=None)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__BaselineGradient_pp_None():
    def method(model):
        return BaselineGradient(model, postprocess=None)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__BaselineGradient_pp_square():
    def method(model):
        return BaselineGradient(model, postprocess="square")

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__BaselineGradient_pp_square():
    def method(model):
        return BaselineGradient(model, postprocess="square")

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__Gradient():
    def method(model):
        return Gradient(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__Gradient():
    def method(model):
        return Gradient(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__Gradient():
    def method(model):
        return Gradient(model)

    dryrun.test_analyzer(method, "imagenet.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__Gradient_pp_None():
    def method(model):
        return Gradient(model, postprocess=None)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__Gradient_pp_None():
    def method(model):
        return Gradient(model, postprocess=None)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__Gradient_pp_square():
    def method(model):
        return Gradient(model, postprocess="square")

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__Gradient_pp_square():
    def method(model):
        return Gradient(model, postprocess="square")

    dryrun.test_analyzer(method, "mnist.*")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__InputTimesGradient():
    def method(model):
        return InputTimesGradient(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__InputTimesGradient():
    def method(model):
        return InputTimesGradient(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__InputTimesGradient():
    def method(model):
        return InputTimesGradient(model)

    dryrun.test_analyzer(method, "imagenet.*")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__Deconvnet():
    def method(model):
        return Deconvnet(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__Deconvnet():
    def method(model):
        return Deconvnet(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__Deconvnet():
    def method(model):
        return Deconvnet(model)

    dryrun.test_analyzer(method, "imagenet.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__GuidedBackprop():
    def method(model):
        return GuidedBackprop(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__GuidedBackprop():
    def method(model):
        return GuidedBackprop(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__GuidedBackprop():
    def method(model):
        return GuidedBackprop(model)

    dryrun.test_analyzer(method, "imagenet.*")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__IntegratedGradients():
    def method(model):
        return IntegratedGradients(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__IntegratedGradients():
    def method(model):
        return IntegratedGradients(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__IntegratedGradients():
    def method(model):
        return IntegratedGradients(model, steps=2)

    dryrun.test_analyzer(method, "imagenet.*")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__SmoothGrad():
    def method(model):
        return SmoothGrad(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__SmoothGrad():
    def method(model):
        return SmoothGrad(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__SmoothGrad():
    def method(model):
        return SmoothGrad(model, augment_by_n=2)

    dryrun.test_analyzer(method, "imagenet.*")
