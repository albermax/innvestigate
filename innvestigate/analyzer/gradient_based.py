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


import keras.backend as K
import keras.models
import keras
import numpy as np


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

    def __init__(self, model, **kwargs):
        self._model_checks = [
            # todo: Check for non-linear output in general.
            {
                "check": lambda layer: kchecks.contains_activation(
                    layer, activation="softmax"),
                "type": "warning",
                "message": ("Typically models are analyzed with respect to "
                            "pre-softmax output."),
            },
        ]

        super(BaselineGradient, self).__init__(model, **kwargs)

    def _create_analysis(self, model):
        return ilayers.Gradient()(model.inputs+[model.outputs[0], ])


class Gradient(base.ReverseAnalyzerBase):
    """Gradient analyzer.

    Returns as analysis the function value with respect to the input.
    The gradient is computed via the librarie's network reverting.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):
        self._model_checks = [
            # todo: Check for non-linear output in general.
            {
                "check": lambda layer: kchecks.contains_activation(
                    layer, activation="softmax"),
                "type": "warning",
                "message": ("Typically models are analyzed with respect to "
                            "pre-softmax output."),
            },
        ]

        super(Gradient, self).__init__(model, **kwargs)

    def _head_mapping(self, X):
        return ilayers.OnesLike()(X)


###############################################################################
###############################################################################
###############################################################################


class Deconvnet(base.ReverseAnalyzerBase):
    """Deconvnet analyzer.

    Applies the "deconvnet" algorithm to analyze the model.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):
        self._model_checks = [
            # todo: Check for non-linear output in general.
            {
                "check": lambda layer: kchecks.contains_activation(
                    layer, activation="softmax"),
                "type": "warning",
                "message": ("Typically models are analyzed with respect to "
                            "pre-softmax output."),
            },
            {
                "check":
                lambda layer: not kchecks.only_relu_activation(layer),
                "type": "warning",
                "message": ("Deconvnet is only well defined for "
                            "neural networks with "
                            "relu activations."),
            },
        ]

        class ReverseLayer(kgraph.ReverseMappingBase):

            def __init__(self, layer, state):
                self._activation = keras.layers.Activation("relu")
                self._layer_wo_relu = kgraph.copy_layer_wo_activation(
                    layer,
                    name_template="reversed_%s",
                )

            def apply(self, Xs, Ys, reversed_Ys, reverse_state):
                # apply relus conditioned on backpropagated values.
                reversed_Ys = kutils.apply(self._activation, reversed_Ys)

                # apply gradient of forward without relus
                Ys_wo_relu = kutils.apply(self._layer_wo_relu, Xs)
                return ilayers.GradientWRT(len(Xs))(Xs+Ys_wo_relu+reversed_Ys)

        # todo: add check for other non-linearities.
        self._conditional_mappings = [
            (lambda layer: kchecks.contains_activation(layer, "relu"),
             ReverseLayer),
        ]
        super(Deconvnet, self).__init__(model, **kwargs)


class GuidedBackprop(base.ReverseAnalyzerBase):
    """Guided backprop analyzer.

    Applies the "guided backprop" algorithm to analyze the model.

    :param model: A Keras model.
    """

    def __init__(self, model, **kwargs):
        self._model_checks = [
            # todo: Check for non-linear output in general.
            {
                "check": lambda layer: kchecks.contains_activation(
                    layer, activation="softmax"),
                "type": "warning",
                "message": ("Typically models are analyzed with respect to "
                            "pre-softmax output."),
            },
            {
                "check":
                lambda layer: not kchecks.only_relu_activation(layer),
                "type": "warning",
                "message": ("Guided Backprop is only well defined for "
                            "neural networks with "
                            "relu activations."),
            },
        ]

        def reverse_layer_instance(Xs, Ys, reversed_Ys, reverse_state):
            activation = keras.layers.Activation("relu")
            reversed_Ys = kutils.apply(activation, reversed_Ys)

            return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)

        # todo: add check for other non-linearities.
        self._conditional_mappings = [
            (lambda layer: kchecks.contains_activation(layer, "relu"),
             reverse_layer_instance),
        ]
        super(GuidedBackprop, self).__init__(model, **kwargs)


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
        subanalyzer = Gradient(model)
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
        subanalyzer = Gradient(model)
        super(SmoothGrad, self).__init__(subanalyzer,
                                         augment_by_n=augment_by_n,
                                         **kwargs)
