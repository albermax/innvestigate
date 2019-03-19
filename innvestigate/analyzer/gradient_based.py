# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import keras.models
import keras


from . import base
from . import wrapper
from .. import layers as ilayers
from .. import utils as iutils
from ..utils import keras as kutils
from ..utils.keras import checks as kchecks
from ..utils.keras import graph as kgraph

__all__ = [
    "BaselineGradient",
    "Gradient",

    "InputTimesGradient",

    "Deconvnet",
    "GuidedBackprop",

    "IntegratedGradients",

    "SmoothGrad",
]


###############################################################################
###############################################################################
###############################################################################


class BaselineGradient(base.AnalyzerNetworkBase):
    """Gradient analyzer based on build-in gradient.

    Returns as analysis the function value with respect to the input.
    The gradient is computed via the build in function.
    Is mainly used for debugging purposes.

    :param model: A Keras model.
    """

    def __init__(self, model, postprocess=None, **kwargs):

        if postprocess not in [None, "abs", "square"]:
            raise ValueError("Parameter 'postprocess' must be either "
                             "None, 'abs', or 'square'.")
        self._postprocess = postprocess

        self._add_model_softmax_check()

        super(BaselineGradient, self).__init__(model, **kwargs)

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        tensors_to_analyze = [x for x in iutils.to_list(model.inputs)
                              if x not in stop_analysis_at_tensors]
        ret = iutils.to_list(ilayers.Gradient()(
            tensors_to_analyze+[model.outputs[0]]))

        if self._postprocess == "abs":
            ret = ilayers.Abs()(ret)
        elif self._postprocess == "square":
            ret = ilayers.Square()(ret)

        return iutils.to_list(ret)

    def _get_state(self):
        state = super(BaselineGradient, self)._get_state()
        state.update({"postprocess": self._postprocess})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        postprocess = state.pop("postprocess")
        kwargs = super(BaselineGradient, clazz)._state_to_kwargs(state)
        kwargs.update({
            "postprocess": postprocess,
        })
        return kwargs


class Gradient(base.ReverseAnalyzerBase):
    """Gradient analyzer.

    Returns as analysis the function value with respect to the input.
    The gradient is computed via the librarie's network reverting.

    :param model: A Keras model.
    """

    def __init__(self, model, postprocess=None, **kwargs):

        if postprocess not in [None, "abs", "square"]:
            raise ValueError("Parameter 'postprocess' must be either "
                             "None, 'abs', or 'square'.")
        self._postprocess = postprocess

        self._add_model_softmax_check()

        super(Gradient, self).__init__(model, **kwargs)

    def _head_mapping(self, X):
        return ilayers.OnesLike()(X)

    def _postprocess_analysis(self, X):
        ret = super(Gradient, self)._postprocess_analysis(X)

        if self._postprocess == "abs":
            ret = ilayers.Abs()(ret)
        elif self._postprocess == "square":
            ret = ilayers.Square()(ret)

        return iutils.to_list(ret)

    def _get_state(self):
        state = super(Gradient, self)._get_state()
        state.update({"postprocess": self._postprocess})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        postprocess = state.pop("postprocess")
        kwargs = super(Gradient, clazz)._state_to_kwargs(state)
        kwargs.update({
            "postprocess": postprocess,
        })
        return kwargs


###############################################################################
###############################################################################
###############################################################################


class InputTimesGradient(Gradient):
    """Input*Gradient analyzer.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):

        self._add_model_softmax_check()

        super(InputTimesGradient, self).__init__(model, **kwargs)

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        tensors_to_analyze = [x for x in iutils.to_list(model.inputs)
                              if x not in stop_analysis_at_tensors]
        gradients = super(InputTimesGradient, self)._create_analysis(
            model, stop_analysis_at_tensors=stop_analysis_at_tensors)
        return [keras.layers.Multiply()([i, g])
                for i, g in zip(tensors_to_analyze, gradients)]


###############################################################################
###############################################################################
###############################################################################


class DeconvnetReverseReLULayer(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        self._activation = keras.layers.Activation("relu")
        self._layer_wo_relu = kgraph.copy_layer_wo_activation(
            layer,
            name_template="reversed_%s",
        )

    def apply(self, Xs, Ys, reversed_Ys, reverse_state):
        # Apply relus conditioned on backpropagated values.
        reversed_Ys = kutils.apply(self._activation, reversed_Ys)

        # Apply gradient of forward pass without relus.
        Ys_wo_relu = kutils.apply(self._layer_wo_relu, Xs)
        return ilayers.GradientWRT(len(Xs))(Xs+Ys_wo_relu+reversed_Ys)


class Deconvnet(base.ReverseAnalyzerBase):
    """Deconvnet analyzer.

    Applies the "deconvnet" algorithm to analyze the model.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):

        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            "Deconvnet is only specified for networks with ReLU activations.",
            check_type="exception",
        )

        super(Deconvnet, self).__init__(model, **kwargs)

    def _create_analysis(self, *args, **kwargs):

        self._add_conditional_reverse_mapping(
            lambda layer: kchecks.contains_activation(layer, "relu"),
            DeconvnetReverseReLULayer,
            name="deconvnet_reverse_relu_layer",
        )

        return super(Deconvnet, self)._create_analysis(*args, **kwargs)


def GuidedBackpropReverseReLULayer(Xs, Ys, reversed_Ys, reverse_state):
    activation = keras.layers.Activation("relu")
    # Apply relus conditioned on backpropagated values.
    reversed_Ys = kutils.apply(activation, reversed_Ys)

    # Apply gradient of forward pass.
    return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)


class GuidedBackprop(base.ReverseAnalyzerBase):
    """Guided backprop analyzer.

    Applies the "guided backprop" algorithm to analyze the model.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):

        self._add_model_softmax_check()
        self._add_model_check(
            lambda layer: not kchecks.only_relu_activation(layer),
            "GuidedBackprop is only specified for "
            "networks with ReLU activations.",
            check_type="exception",
        )

        super(GuidedBackprop, self).__init__(model, **kwargs)

    def _create_analysis(self, *args, **kwargs):

        self._add_conditional_reverse_mapping(
            lambda layer: kchecks.contains_activation(layer, "relu"),
            GuidedBackpropReverseReLULayer,
            name="guided_backprop_reverse_relu_layer",
        )

        return super(GuidedBackprop, self)._create_analysis(*args, **kwargs)


###############################################################################
###############################################################################
###############################################################################


class IntegratedGradients(wrapper.PathIntegrator):
    """Integrated gradient analyzer.

    Applies the "integrated gradient" algorithm to analyze the model.

    :param model: A Keras model.
    :param steps: Number of steps to use average along integration path.
    """

    def __init__(self, model, steps=64, **kwargs):
        subanalyzer_kwargs = {}
        kwargs_keys = ["neuron_selection_mode", "postprocess"]
        for key in kwargs_keys:
            if key in kwargs:
                subanalyzer_kwargs[key] = kwargs.pop(key)
        subanalyzer = Gradient(model, **subanalyzer_kwargs)

        super(IntegratedGradients, self).__init__(subanalyzer,
                                                  steps=steps,
                                                  **kwargs)


###############################################################################
###############################################################################
###############################################################################


class SmoothGrad(wrapper.GaussianSmoother):
    """Smooth grad analyzer.

    Applies the "smooth grad" algorithm to analyze the model.

    :param model: A Keras model.
    :param augment_by_n: Number of distortions to average for smoothing.
    """

    def __init__(self, model, augment_by_n=64, **kwargs):
        subanalyzer_kwargs = {}
        kwargs_keys = ["neuron_selection_mode", "postprocess"]
        for key in kwargs_keys:
            if key in kwargs:
                subanalyzer_kwargs[key] = kwargs.pop(key)
        subanalyzer = Gradient(model, **subanalyzer_kwargs)

        super(SmoothGrad, self).__init__(subanalyzer,
                                         augment_by_n=augment_by_n,
                                         **kwargs)
