from __future__ import annotations

from typing import Dict, Optional

import keras
import keras.models

import innvestigate.layers as ilayers
import innvestigate.utils as iutils
import innvestigate.utils.keras as kutils
import innvestigate.utils.keras.checks as kchecks
import innvestigate.utils.keras.graph as kgraph
from innvestigate.analyzer.network_base import AnalyzerNetworkBase
from innvestigate.analyzer.reverse_base import ReverseAnalyzerBase
from innvestigate.analyzer.wrapper import GaussianSmoother, PathIntegrator

__all__ = [
    "BaselineGradient",
    "Gradient",
    "InputTimesGradient",
    "Deconvnet",
    "GuidedBackprop",
    "IntegratedGradients",
    "SmoothGrad",
]


class BaselineGradient(AnalyzerNetworkBase):
    """Gradient analyzer based on build-in gradient.

    Returns as analysis the function value with respect to the input.
    The gradient is computed via the build in function.
    Is mainly used for debugging purposes.

    :param model: A Keras model.
    """

    def __init__(self, model, postprocess=None, **kwargs):
        super().__init__(model, **kwargs)

        if postprocess not in [None, "abs", "square"]:
            raise ValueError(
                "Parameter 'postprocess' must be either " "None, 'abs', or 'square'."
            )
        self._postprocess = postprocess

        self._add_model_softmax_check()
        self._do_model_checks()

    def _create_analysis(self, model, stop_analysis_at_tensors=None):
        if stop_analysis_at_tensors is None:
            stop_analysis_at_tensors = []

        tensors_to_analyze = [
            x for x in iutils.to_list(model.inputs) if x not in stop_analysis_at_tensors
        ]
        ret = iutils.to_list(
            ilayers.Gradient()(tensors_to_analyze + [model.outputs[0]])
        )

        if self._postprocess == "abs":
            ret = ilayers.Abs()(ret)
        elif self._postprocess == "square":
            ret = ilayers.Square()(ret)

        return iutils.to_list(ret)

    def _get_state(self):
        state = super()._get_state()
        state.update({"postprocess": self._postprocess})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        postprocess = state.pop("postprocess")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update(
            {
                "postprocess": postprocess,
            }
        )
        return kwargs


class Gradient(ReverseAnalyzerBase):
    """Gradient analyzer.

    Returns as analysis the function value with respect to the input.
    The gradient is computed via the librarie's network reverting.

    :param model: A Keras model.
    """

    def __init__(self, model, postprocess: Optional[str] = None, **kwargs):
        super(Gradient, self).__init__(model, **kwargs)

        if postprocess not in [None, "abs", "square"]:
            raise ValueError(
                """Parameter 'postprocess' must be either None, "abs", or "square"."""
            )
        self._postprocess = postprocess

        # Add and run model checks
        self._add_model_softmax_check()
        self._do_model_checks()

    def _head_mapping(self, X):
        return ilayers.OnesLike()(X)

    def _postprocess_analysis(self, X):
        ret = super()._postprocess_analysis(X)

        if self._postprocess == "abs":
            ret = ilayers.Abs()(ret)
        elif self._postprocess == "square":
            ret = ilayers.Square()(ret)

        return iutils.to_list(ret)

    def _get_state(self):
        state = super()._get_state()
        state.update({"postprocess": self._postprocess})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        postprocess = state.pop("postprocess")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update(
            {
                "postprocess": postprocess,
            }
        )
        return kwargs


###############################################################################


class InputTimesGradient(Gradient):
    """Input*Gradient analyzer.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):

        super(InputTimesGradient, self).__init__(model, **kwargs)

    def _create_analysis(self, model, stop_analysis_at_tensors=None):
        if stop_analysis_at_tensors is None:
            stop_analysis_at_tensors = []

        tensors_to_analyze = [
            x for x in iutils.to_list(model.inputs) if x not in stop_analysis_at_tensors
        ]
        gradients = super()._create_analysis(
            model, stop_analysis_at_tensors=stop_analysis_at_tensors
        )
        return [
            keras.layers.Multiply()([i, g])
            for i, g in zip(tensors_to_analyze, gradients)
        ]


###############################################################################


class DeconvnetReverseReLULayer(kgraph.ReverseMappingBase):
    def __init__(self, layer, state):
        self._activation = keras.layers.Activation("relu")
        self._layer_wo_relu = kgraph.copy_layer_wo_activation(
            layer,
            name_template="reversed_%s",
        )

    def apply(self, Xs, Ys, reversed_Ys, reverse_state: Dict):
        # Apply relus conditioned on backpropagated values.
        reversed_Ys = kutils.apply(self._activation, reversed_Ys)

        # Apply gradient of forward pass without relus.
        Ys_wo_relu = kutils.apply(self._layer_wo_relu, Xs)
        return ilayers.GradientWRT(len(Xs))(Xs + Ys_wo_relu + reversed_Ys)


class Deconvnet(ReverseAnalyzerBase):
    """Deconvnet analyzer.

    Applies the "deconvnet" algorithm to analyze the model.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

        # Add and run model checks
        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            "Deconvnet is only specified for networks with ReLU activations.",
            check_type="exception",
        )
        self._do_model_checks()

    def _create_analysis(self, *args, **kwargs):

        self._add_conditional_reverse_mapping(
            lambda layer: kchecks.contains_activation(layer, "relu"),
            DeconvnetReverseReLULayer,
            name="deconvnet_reverse_relu_layer",
        )

        return super()._create_analysis(*args, **kwargs)


def GuidedBackpropReverseReLULayer(Xs, Ys, reversed_Ys, reverse_state: Dict):
    activation = keras.layers.Activation("relu")
    # Apply relus conditioned on backpropagated values.
    reversed_Ys = kutils.apply(activation, reversed_Ys)

    # Apply gradient of forward pass.
    return ilayers.GradientWRT(len(Xs))(Xs + Ys + reversed_Ys)


class GuidedBackprop(ReverseAnalyzerBase):
    """Guided backprop analyzer.

    Applies the "guided backprop" algorithm to analyze the model.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

        # Add and run model checks
        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            "GuidedBackprop is only specified for " "networks with ReLU activations.",
            check_type="exception",
        )
        self._do_model_checks()

    def _create_analysis(self, *args, **kwargs):

        self._add_conditional_reverse_mapping(
            lambda layer: kchecks.contains_activation(layer, "relu"),
            GuidedBackpropReverseReLULayer,
            name="guided_backprop_reverse_relu_layer",
        )

        return super()._create_analysis(*args, **kwargs)


###############################################################################


class IntegratedGradients(PathIntegrator):
    """Integrated gradient analyzer.

    Applies the "integrated gradient" algorithm to analyze the model.

    :param model: A Keras model.
    :param steps: Number of steps to use average along integration path.
    """

    def __init__(self, model, steps=64, **kwargs):
        # If initialized through serialization:
        if "subanalyzer" in kwargs:
            subanalyzer = kwargs.pop("subanalyzer")
        # If initialized normally:
        else:
            subanalyzer_kwargs = {}
            kwargs_keys = ["neuron_selection_mode", "postprocess"]
            for key in kwargs_keys:
                if key in kwargs:
                    subanalyzer_kwargs[key] = kwargs.pop(key)
            subanalyzer = Gradient(model, **subanalyzer_kwargs)

        super().__init__(subanalyzer, steps=steps, **kwargs)


###############################################################################


class SmoothGrad(GaussianSmoother):
    """Smooth grad analyzer.

    Applies the "smooth grad" algorithm to analyze the model.

    :param model: A Keras model.
    :param augment_by_n: Number of distortions to average for smoothing.
    """

    def __init__(self, model, augment_by_n=64, **kwargs):
        # If initialized through serialization:
        if "subanalyzer" in kwargs:
            subanalyzer = kwargs.pop("subanalyzer")
        # If initialized normally:
        else:
            subanalyzer_kwargs = {}
            kwargs_keys = ["neuron_selection_mode", "postprocess"]
            for key in kwargs_keys:
                if key in kwargs:
                    subanalyzer_kwargs[key] = kwargs.pop(key)
            subanalyzer = Gradient(model, **subanalyzer_kwargs)

        super().__init__(subanalyzer, augment_by_n=augment_by_n, **kwargs)
