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


from . import base
from .. import layers as ilayers
from .. import utils
from ..utils import keras as kutils

import keras.backend as K
import keras.models
import keras
import numpy as np


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
        # we assume there is only one head!
        gradient_head_processed = [False]
        layer_cache = {}

        if patterns is None:
            raise ValueError("Patterns are required.")

        # copy patterns
        self.patterns = list(patterns)

        def gradient_reverse(Xs, Ys, reversed_Ys, reverse_state):
            if gradient_head_processed[0] is not True:
                # replace function value with ones as the last element
                # chain rule is a one.
                gradient_head_processed[0] = True
                reversed_Ys = utils.listify(ilayers.OnesLike()(reversed_Ys))

            layer = reverse_state["layer"]
            if kutils.contains_kernel(layer):
                # layers can be applied to several nodes.
                # but we need to revert it only once.
                if layer in layer_cache:
                    layer_wo_relu = layer_cache[layer]
                    Ys_wo_relu = kutils.easy_apply(layer_wo_relu, Xs)
                else:
                    config = layer.get_config()
                    config["name"] = "reversed_%s" % config["name"]
                    layer_wo_relu = layer.__class__.from_config(config)
                    Ys_wo_relu = kutils.easy_apply(layer_wo_relu, Xs)

                    # replace kernel weights with pattern weights
                    # assume that only one matches
                    pattern = self.patterns.pop()
                    weights = layer.get_weights()
                    tmp = [pattern.shape == x.shape for x in weights]
                    if np.sum(tmp) != 1:
                        raise Exception("Cannot match pattern to kernel.")
                    idx = np.argmax(tmp)
                    weights[idx] = pattern
                    layer_wo_relu.set_weights(weights)
                    layer_cache[layer] = layer_wo_relu

                return ilayers.GradientWRT(len(Xs))(Xs+Ys_wo_relu+reversed_Ys)
            else:
                return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)   

        self.default_reverse = gradient_reverse
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
