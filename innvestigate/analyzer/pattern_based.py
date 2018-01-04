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

import keras.activations
import keras.backend as K
import keras.models
import keras
import numpy as np


from . import base
from .. import layers as ilayers
from .. import utils
from ..utils import keras as kutils
from ..utils.keras import graph as kgraph


__all__ = [
    "PatternNet",
    "PatternAttribution",
]


###############################################################################
###############################################################################
###############################################################################


class PatternNet(base.BaseReverseNetwork):

    properties = {
        "name": "PatternNet",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, *args, patterns=None, **kwargs):
        layer_cache = {}

        if patterns is None:
            raise ValueError("Patterns are required.")

        # copy patterns, will be used as stack
        self.patterns = list(patterns)

        def reverse(Xs, Ys, reversed_Ys, reverse_state):
            layer = reverse_state["layer"]
            if kgraph.contains_kernel(layer):
                # we want to apply the filter weights on the forward pass
                # on the backward pass we want copy the gradient computation
                # except that using the pattern weights instead of the filter w
                # i.e. we need to revert the activation with the forward
                # activations to keep the relu patterns of the gradient comp.
                # this is the reason for splitting the gradient mimicking:

                # layers can be applied to several nodes.
                # but we need to revert it only once.
                if layer in layer_cache:
                    kernel_layer, pattern_layer, act_layer = layer_cache[layer]
                    act_Xs = kutils.easy_apply(kernel_layer, Xs)
                    act_Ys = kutils.easy_apply(act_layer, act_Xs)
                    pattern_Ys = kutils.easy_apply(pattern_layer, Xs)
                else:
                    config = layer.get_config()

                    activation = None
                    if "activation" in config:
                        activation = config["activation"]
                        config["activation"] = None
                    act_layer = keras.layers.Activation(
                        activation,
                        name="reversed_act_%s" % config["name"])

                    kernel_layer = kgraph.get_layer_wo_activation(
                        layer, name_template="reversed_kernel_%s")
                    act_Xs = kutils.easy_apply(kernel_layer, Xs)
                    act_Ys = kutils.easy_apply(act_layer, act_Xs)


                    # replace kernel weights with pattern weights
                    # assume that only one matches
                    pattern_weights = layer.get_weights()
                    pattern = self.patterns.pop()
                    tmp = [pattern.shape == x.shape for x in pattern_weights]
                    if np.sum(tmp) != 1:
                        raise Exception("Cannot match pattern to kernel.")
                    pattern_weights[np.argmax(tmp)] = pattern
                    pattern_layer = kgraph.get_layer_wo_activation(
                        layer,
                        name_template="reversed_pattern_%s",
                        weights=pattern_weights)
                    pattern_Ys = kutils.easy_apply(pattern_layer, Xs)

                    layer_cache[layer] = (kernel_layer,
                                          pattern_layer,
                                          act_layer)

                grad_act = ilayers.GradientWRT(len(act_Xs))
                grad_pattern = ilayers.GradientWRT(len(Xs))

                if act_layer.activation in [None,
                                            keras.activations.get("linear")]:
                    tmp = reversed_Ys
                else:
                    # if linear activation this behaves strange
                    tmp = utils.listify(grad_act(act_Xs+act_Ys+reversed_Ys))
                return grad_pattern(Xs+pattern_Ys+tmp)
            else:
                return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)   

        self.default_reverse = reverse
        return super(PatternNet, self).__init__(*args, **kwargs)

    def _create_analysis(self, *args, **kwargs):
        ret = super(PatternNet, self)._create_analysis(*args, **kwargs)

        if len(self.patterns) == 0:
            del self.patterns
        else:
            raise Exception("Not all patterns consumed. Something is wrong.")

        return ret


class PatternAttribution(PatternNet):

    properties = {
        "name": "PatternAttribution",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, *args, patterns=None, **kwargs):
        if patterns is None:
            raise ValueError("Patterns are required.")

        # copy patterns, will be used as stack
        patterns = list(patterns)

        # copy patterns
        new_patterns = []
        for W in model.get_weights():
            if W.shape == patterns[0].shape:
                new_patterns.append(np.multiply(W, patterns[0]))
                patterns.pop(0)
            if len(patterns) == 0:
                break

        if len(patterns) != 0:
            raise Exception("Not all patterns consumed. Something is wrong.")

        return super(PatternAttribution, self).__init__(model,
                                                        *args,
                                                        patterns=new_patterns,
                                                        **kwargs)
