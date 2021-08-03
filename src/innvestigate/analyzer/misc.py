from __future__ import annotations

import innvestigate.layers as ilayers
import innvestigate.utils as iutils
from innvestigate.analyzer.network_base import AnalyzerNetworkBase

__all__ = ["Random", "Input"]


class Input(AnalyzerNetworkBase):
    """Returns the input.

    Returns the input as analysis.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs) -> None:
        super().__init__(model, **kwargs)

    def _create_analysis(self, model, stop_analysis_at_tensors=None):
        if stop_analysis_at_tensors is None:
            stop_analysis_at_tensors = []

        tensors_to_analyze = [
            x for x in iutils.to_list(model.inputs) if x not in stop_analysis_at_tensors
        ]
        return [ilayers.Identity()(x) for x in tensors_to_analyze]


class Random(AnalyzerNetworkBase):
    """Returns noise.

    Returns the Gaussian noise as analysis.

    :param model: A Keras model.
    :param stddev: The standard deviation of the noise.
    """

    def __init__(self, model, stddev=1, **kwargs):
        self._stddev = stddev

        super().__init__(model, **kwargs)

    def _create_analysis(self, model, stop_analysis_at_tensors=None):
        if stop_analysis_at_tensors is None:
            stop_analysis_at_tensors = []

        noise = ilayers.TestPhaseGaussianNoise(stddev=self._stddev)
        tensors_to_analyze = [
            x for x in iutils.to_list(model.inputs) if x not in stop_analysis_at_tensors
        ]
        return [noise(x) for x in tensors_to_analyze]

    def _get_state(self):
        state = super()._get_state()
        state.update({"stddev": self._stddev})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        stddev = state.pop("stddev")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update({"stddev": stddev})
        return kwargs
