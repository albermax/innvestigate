# Get Python six functionality:
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

from innvestigate.analyzer import Input
from innvestigate.analyzer import Random
from innvestigate.utils.tests import dryrun

###############################################################################
###############################################################################
###############################################################################


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__Input():
    def method(model):
        return Input(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__Random():
    def method(model):
        return Random(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__SerializeRandom():
    def method(model):
        return Random(model)

    dryrun.test_serialize_analyzer(method, "trivia.*:mnist.log_reg")
