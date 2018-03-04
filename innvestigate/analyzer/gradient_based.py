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

    def _create_analysis(self, model):
        return ilayers.Gradient()(model.inputs+[model.outputs[0], ])


class Gradient(base.ReverseAnalyzerBase):

    def _head_mapping(self, X):
        return ilayers.OnesLike()(X)


###############################################################################
###############################################################################
###############################################################################


class Deconvnet(base.ReverseAnalyzerBase):

    def __init__(self, *args, **kwargs):
        self._model_checks = [
            # todo: Check for non-linear output in general.
            {
                "check": lambda layer: kgraph.contains_activation(
                    layer, activation="softmax"),
                "type": "exception",
                "message": "Model should not contain a softmax.",
            },
            {
                "check": lambda layer: not kgraph.is_relu_convnet_layer(layer),
                "type": "warning",
                "mesage": ("Deconvnet is only well defined for "
                           "convolutional neural networks with "
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
            (lambda layer: kgraph.contains_activation(layer, "relu"),
             ReverseLayer),
        ]
        return super(Deconvnet, self).__init__(*args, **kwargs)


class GuidedBackprop(base.ReverseAnalyzerBase):

    def __init__(self, *args, **kwargs):
        self._model_checks = [
            # todo: Check for non-linear output in general.
            {
                "check": lambda layer: kgraph.contains_activation(
                    layer, activation="softmax"),
                "type": "exception",
                "message": "Model should not contain a softmax.",
            },
            {
                "check": lambda layer: not kgraph.is_relu_convnet_layer(layer),
                "type": "warning",
                "mesage": ("Guided Backprop is only well defined for "
                           "convolutional neural networks with "
                           "relu activations."),
            },
        ]

        def reverse_layer_instance(Xs, Ys, reversed_Ys, reverse_state):
            activation = keras.layers.Activation("relu")
            reversed_Ys = kutils.apply(activation, reversed_Ys)

            return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)

        # todo: add check for other non-linearities.
        self._conditional_mappings = [
            (lambda layer: kgraph.contains_activation(layer, "relu"),
             reverse_layer_instance),
        ]
        return super(GuidedBackprop, self).__init__(*args, **kwargs)


###############################################################################
###############################################################################
###############################################################################


class IntegratedGradients(wrapper.PathIntegrator):

    def __init__(self, model, steps=64, *args, **kwargs):
        subanalyzer = Gradient(model)
        return super(IntegratedGradients, self).__init__(subanalyzer,
                                                         *args,
                                                         steps=steps,
                                                         **kwargs)


###############################################################################
###############################################################################
###############################################################################


class SmoothGrad(wrapper.GaussianSmoother):

    def __init__(self, model, augment_by_n=64, *args, **kwargs):
        subanalyzer = Gradient(model)
        return super(SmoothGrad, self).__init__(
            subanalyzer,
            *args,
            augment_by_n=augment_by_n,
            **kwargs)
