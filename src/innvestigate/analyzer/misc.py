from __future__ import annotations

import innvestigate.backend as ibackend
import innvestigate.layers as ilayers
from innvestigate.analyzer.network_base import AnalyzerNetworkBase
from innvestigate.backend.types import Tensor

__all__ = ["Random", "Input"]


class Input(AnalyzerNetworkBase):
    """Returns the input as analysis.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs) -> None:
        super().__init__(model, **kwargs)

    def _create_analysis(self, model, stop_analysis_at_tensors=None):
        if stop_analysis_at_tensors is None:
            stop_analysis_at_tensors = []

        tensors_to_analyze = [
            x
            for x in ibackend.to_list(model.inputs)
            if x not in stop_analysis_at_tensors
        ]
        return [ilayers.Identity()(x) for x in tensors_to_analyze]


class Random(AnalyzerNetworkBase):
    """Returns the input with added zero-mean Gaussian noise as analysis.

    :param model: A Keras model.
    :param stddev: The standard deviation of the noise.
    """

    def __init__(self, model, stddev=1, **kwargs):
        self._stddev = stddev

        super().__init__(model, **kwargs)

    def _create_analysis(self, model, stop_analysis_at_tensors=None) -> list[Tensor]:
        if stop_analysis_at_tensors is None:
            stop_analysis_at_tensors = []

        tensors_to_analyze = [
            X
            for X in ibackend.to_list(model.inputs)
            if X not in stop_analysis_at_tensors
        ]
        tensors_with_noise = [
            ibackend.add_gaussian_noise(X, stddev=self._stddev)
            for X in tensors_to_analyze
        ]
        return tensors_with_noise

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
