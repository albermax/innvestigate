# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


from .base import AnalyzerNetworkBase
from .. import layers as ilayers
from .. import utils as iutils


__all__ = ["Random", "Input"]


###############################################################################
###############################################################################
###############################################################################

#TODO: tf2.*
class Input(AnalyzerNetworkBase):
    """Returns the input.

    Returns the input as analysis.

    :param model: A Keras model.
    """

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        tensors_to_analyze = [x for x in iutils.to_list(model.inputs)
                              if x.experimental_ref() not in [a.experimental_ref() for a in stop_analysis_at_tensors]]
        return [ilayers.Identity()(x) for x in tensors_to_analyze]

#TODO: tf2.*
class Random(AnalyzerNetworkBase):
    """Returns noise.

    Returns the Gaussian noise as analysis.

    :param model: A Keras model.
    :param stddev: The standard deviation of the noise.
    """

    def __init__(self, model, stddev=1, **kwargs):
        self._stddev = stddev

        super(Random, self).__init__(model, **kwargs)

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        noise = ilayers.TestPhaseGaussianNoise(stddev=self._stddev)
        tensors_to_analyze = [x for x in iutils.to_list(model.inputs)
                              if x.experimental_ref() not in [a.experimental_ref() for a in stop_analysis_at_tensors]]
        return [noise(x) for x in tensors_to_analyze]
