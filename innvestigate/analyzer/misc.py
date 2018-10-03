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


import numpy as np

from .base import AnalyzerNetworkBase
from .. import layers as ilayers
from .. import utils as iutils


__all__ = ["Random", "Input"]


###############################################################################
###############################################################################
###############################################################################


class Input(AnalyzerNetworkBase):

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        tensors_to_analyze = [x for x in iutils.to_list(model.inputs)
                              if x not in stop_analysis_at_tensors]
        return [ilayers.Identity()(x) for x in tensors_to_analyze]


class Random(AnalyzerNetworkBase):

    def __init__(self, model, stddev=1):
        self._stddev = 1

        super(Random, self).__init__(model)

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        noise = ilayers.TestPhaseGaussianNoise(stddev=1)
        tensors_to_analyze = [x for x in iutils.to_list(model.inputs)
                              if x not in stop_analysis_at_tensors]
        return [noise(x) for x in tensors_to_analyze]

    def _get_state(self):
        state = super(Random, self)._get_state()
        state.update({"stddev": self._stddev})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        stddev = state.pop("stddev")
        kwargs = super(Random, clazz)._state_to_kwargs(state)
        kwargs.update({"stddev": stddev})
        return kwargs
