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


class BaselineGradient(base.BaseNetwork):

    properties = {
        "name": "BaselineGradient",
        "show_as": "rgb",
    }


    def _create_analysis(self, model):
        return ilayers.Gradient()(model.inputs+[model.outputs[0],])


class Gradient(base.BaseReverseNetwork):

    properties = {
        "name": "Gradient",
        "show_as": "rgb",
    }

    def __init__(self, *args, **kwargs):
        # we assume there is only one head!
        head_processed = [False]
        def reverse(Xs, Ys, reversed_Ys, reverse_state):
            if head_processed[0] is not True:
                # replace function value with ones as the last element
                # chain rule is a one.
                head_processed[0] = True
                reversed_Ys = utils.listify(ilayers.OnesLike()(reversed_Ys))
            return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)

        self.default_reverse = reverse
        return super(Gradient, self).__init__(*args, **kwargs)


###############################################################################
###############################################################################
###############################################################################


class Deconvnet(base.BaseReverseNetwork):

    properties = {
        "name": "Deconvnet",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, *args, **kwargs):
        # we assume there is only one head!
        head_processed = [False]
        layer_cache = {}

        def reverse(Xs, Ys, reversed_Ys, reverse_state):
            if head_processed[0] is not True:
                # replace function value with ones as the last element
                # chain rule is a one.
                head_processed[0] = True
                reversed_Ys = utils.listify(ilayers.OnesLike()(reversed_Ys))

            layer = reverse_state["layer"]
            # todo: add check for other non-linearities. 
            if kgraph.contains_activation(layer, "relu"):
                activation = keras.layers.Activation("relu")
                reversed_Ys = kutils.easy_apply(activation, reversed_Ys)

                # layers can be applied to several nodes.
                # but we need to revert it only once.
                if layer in layer_cache:
                    layer_wo_relu = layer_cache[layer]
                    Ys_wo_relu = kutils.easy_apply(layer_wo_relu, Xs)
                else:
                    layer_wo_ = kgraph.get_layer_wo_activation(
                        layer,
                        name_template="reversed_%s",
                    )
                    Ys_wo_relu = kutils.easy_apply(layer_wo_relu, Xs)
                    layer_cache[layer] = layer_wo_relu

                return ilayers.GradientWRT(len(Xs))(Xs+Ys_wo_relu+reversed_Ys)
            else:
                return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)   

        self.default_reverse = reverse
        return super(Deconvnet, self).__init__(*args, **kwargs)


class GuidedBackprop(base.BaseReverseNetwork):

    properties = {
        "name": "GuidedBackprop",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, *args, **kwargs):
        # we assume there is only one head!
        head_processed = [False]
        layer_cache = {}

        def reverse(Xs, Ys, reversed_Ys, reverse_state):
            if head_processed[0] is not True:
                # replace function value with ones as the last element
                # chain rule is a one.
                head_processed[0] = True
                reversed_Ys = utils.listify(ilayers.OnesLike()(reversed_Ys))

            # todo: add check for other non-linearities.
            layer = reverse_state["layer"]
            if kgraph.contains_activation(layer, "relu"):
                activation = keras.layers.Activation("relu")
                reversed_Ys = kutils.easy_apply(activation, reversed_Ys)

            return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)

        self.default_reverse = reverse
        return super(GuidedBackprop, self).__init__(*args, **kwargs)
