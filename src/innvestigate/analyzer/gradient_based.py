from __future__ import annotations

import tensorflow as tf
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers

import innvestigate.backend as ibackend
import innvestigate.backend.checks as ichecks
import innvestigate.backend.graph as igraph
from innvestigate.analyzer.network_base import AnalyzerNetworkBase
from innvestigate.analyzer.reverse_base import ReverseAnalyzerBase
from innvestigate.analyzer.wrapper import GaussianSmoother, PathIntegrator
from innvestigate.backend.types import List, OptionalList, Tensor

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
            x
            for x in ibackend.to_list(model.inputs)
            if x not in stop_analysis_at_tensors
        ]
        ret = ibackend.to_list(kbackend.gradients(model.outputs[0], tensors_to_analyze))

        if self._postprocess == "abs":
            ret = [kbackend.abs(r) for r in ret]
        elif self._postprocess == "square":
            ret = [kbackend.square(r) for r in ret]

        return ret

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

    def __init__(self, model, postprocess: str | None = None, **kwargs):
        super().__init__(model, **kwargs)

        if postprocess not in [None, "abs", "square"]:
            raise ValueError(
                """Parameter 'postprocess' must be either None, "abs", or "square"."""
            )
        self._postprocess = postprocess

        # Add and run model checks
        self._add_model_softmax_check()
        self._do_model_checks()

    def _head_mapping(self, X: Tensor) -> Tensor:
        return tf.ones_like(X)

    def _postprocess_analysis(self, Xs: OptionalList[Tensor]) -> List[Tensor]:
        ret = super()._postprocess_analysis(Xs)

        if self._postprocess == "abs":
            ret = [kbackend.abs(r) for r in ret]
        elif self._postprocess == "square":
            ret = [kbackend.square(r) for r in ret]

        return ret

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

        super().__init__(model, **kwargs)

    def _create_analysis(self, model, stop_analysis_at_tensors=None):
        if stop_analysis_at_tensors is None:
            stop_analysis_at_tensors = []

        tensors_to_analyze = [
            x
            for x in ibackend.to_list(model.inputs)
            if x not in stop_analysis_at_tensors
        ]
        gradients = super()._create_analysis(
            model, stop_analysis_at_tensors=stop_analysis_at_tensors
        )
        return [
            klayers.Multiply()([i, g]) for i, g in zip(tensors_to_analyze, gradients)
        ]


###############################################################################


class DeconvnetReverseReLULayer(igraph.ReverseMappingBase):
    def __init__(self, layer, _state):
        self._activation = klayers.Activation("relu")
        self._layer_wo_relu = igraph.copy_layer_wo_activation(
            layer,
            name_template="reversed_%s",
        )

    def apply(self, Xs, Ys, Rs, reverse_state: dict) -> List[Tensor]:
        # Apply relus conditioned on backpropagated values.
        Rs = ibackend.apply(self._activation, Rs)

        # Apply gradient of forward pass without relus.
        Ys_wo_relu = ibackend.apply(self._layer_wo_relu, Xs)
        return ibackend.gradients(Xs, Ys_wo_relu, Rs)


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
            lambda layer: not ichecks.only_relu_activation(layer),
            "Deconvnet is only specified for networks with ReLU activations.",
            check_type="exception",
        )
        self._do_model_checks()

    def _create_analysis(self, *args, **kwargs):

        self._add_conditional_reverse_mapping(
            lambda layer: ichecks.contains_activation(layer, "relu"),
            DeconvnetReverseReLULayer,
            name="deconvnet_reverse_relu_layer",
        )

        return super()._create_analysis(*args, **kwargs)


def guided_backprop_reverse_relu_layer(Xs, Ys, reversed_Ys, _reverse_state: dict):
    activation = klayers.Activation("relu")
    # Apply relus conditioned on backpropagated values.
    reversed_Ys = ibackend.apply(activation, reversed_Ys)

    # Apply gradient of forward pass.
    return ibackend.gradients(Xs, Ys, reversed_Ys)


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
            lambda layer: not ichecks.only_relu_activation(layer),
            "GuidedBackprop is only specified for networks with ReLU activations.",
            check_type="exception",
        )
        self._do_model_checks()

    def _create_analysis(self, *args, **kwargs):

        self._add_conditional_reverse_mapping(
            lambda layer: ichecks.contains_activation(layer, "relu"),
            guided_backprop_reverse_relu_layer,
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

    def __init__(
        self,
        model,
        steps=64,
        neuron_selection_mode="max_activation",
        postprocess=None,
        **kwargs,
    ):
        # If initialized through serialization:
        if "subanalyzer" in kwargs:
            subanalyzer = kwargs.pop("subanalyzer")
        # If initialized normally:
        else:
            subanalyzer = Gradient(
                model,
                neuron_selection_mode=neuron_selection_mode,
                postprocess=postprocess,
            )

        super().__init__(
            subanalyzer,
            steps=steps,
            neuron_selection_mode=neuron_selection_mode,
            **kwargs,
        )


###############################################################################


class SmoothGrad(GaussianSmoother):
    """Smooth grad analyzer.

    Applies the "smooth grad" algorithm to analyze the model.

    :param model: A Keras model.
    :param augment_by_n: Number of distortions to average for smoothing.
    """

    def __init__(
        self,
        model,
        augment_by_n=64,
        neuron_selection_mode="max_activation",
        postprocess=None,
        **kwargs,
    ):
        # If initialized through serialization:
        if "subanalyzer" in kwargs:
            subanalyzer = kwargs.pop("subanalyzer")
        # If initialized normally:
        else:

            subanalyzer = Gradient(
                model,
                neuron_selection_mode=neuron_selection_mode,
                postprocess=postprocess,
            )

        super().__init__(
            subanalyzer,
            augment_by_n=augment_by_n,
            neuron_selection_mode=neuron_selection_mode,
            **kwargs,
        )
