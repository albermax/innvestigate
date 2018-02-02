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


from . import base
from .. import layers as ilayers
from .. import utils
from ..utils import keras as kutils
from ..utils.keras import graph as kgraph

__all__ = [
    "BaselineGradient",
    "Gradient",

    "Deconvnet",
    "GuidedBackprop",
]


###############################################################################
###############################################################################
###############################################################################


class BaselineGradient(base.AnalyzerNetworkBase):

    properties = {
        "name": "BaselineGradient",
        "show_as": "rgb",
    }

    def _create_analysis(self, model):
        return ilayers.Gradient()(model.inputs+[model.outputs[0], ])


class Gradient(base.ReverseAnalyzerBase):

    properties = {
        "name": "Gradient",
        "show_as": "rgb",
    }

    def _head_mapping(self, X):
        return ilayers.OnesLike()(X)


###############################################################################
###############################################################################
###############################################################################


class Deconvnet(base.ReverseAnalyzerBase):

    properties = {
        "name": "Deconvnet",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, *args, **kwargs):
        self._model_checks = [
            lambda layer: not kgraph.is_relu_convnet_layer(layer),
        ]
        self._model_checks_msg = (
            "Deconvnet is only well defined for "
            "convluational neural networks with non-relu activations."
            )

        def reverse_layer(Xs, Ys, reversed_Ys, reverse_state):
            activation = keras.layers.Activation("relu")
            layer = reverse_state["layer"]
            layer_wo_relu = kgraph.get_layer_wo_activation(
                layer,
                name_template="reversed_%s",
            )

            def reverse_layer_instance(Xs, Ys, reversed_Ys, reverse_state):
                # apply relus conditioned on backpropagated values.
                reversed_Ys = kutils.easy_apply(activation, reversed_Ys)

                # apply gradient of forward without relus
                Ys_wo_relu = kutils.easy_apply(layer_wo_relu, Xs)
                return ilayers.GradientWRT(len(Xs))(Xs+Ys_wo_relu+reversed_Ys)

            return reverse_layer_instance

        # todo: add check for other non-linearities.
        self._conditional_mappings = [
            (lambda layer: kgraph.contains_activation(layer, "relu"),
             reverse_layer),
        ]
        return super(Deconvnet, self).__init__(*args, **kwargs)


class GuidedBackprop(base.ReverseAnalyzerBase):

    properties = {
        "name": "GuidedBackprop",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, *args, **kwargs):
        self._model_checks = [
            lambda layer: not kgraph.is_relu_convnet_layer(layer),
        ]
        self._model_checks_msg = (
            "GuidedBackprop is only well defined for "
            "convluational neural networks with non-relu activations."
            )

        def reverse_layer_instance(Xs, Ys, reversed_Ys, reverse_state):
            activation = keras.layers.Activation("relu")
            reversed_Ys = kutils.easy_apply(activation, reversed_Ys)

            return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)

        # todo: add check for other non-linearities.
        self._conditional_mappings = [
            (lambda layer: kgraph.contains_activation(layer, "relu"),
             reverse_layer_instance),
        ]
        return super(GuidedBackprop, self).__init__(*args, **kwargs)
