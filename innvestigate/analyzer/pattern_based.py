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


class PatternNet(base.ReverseAnalyzerBase):

    properties = {
        "name": "PatternNet",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, *args, patterns=None, **kwargs):
        self._model_checks = [
            lambda layer: not kgraph.is_relu_convnet_layer(layer),
        ]
        self._model_checks_msg = (
            "PatternNet is only well defined for "
            "convluational neural networks with non-relu activations."
            )

        if patterns is None:
            raise ValueError("Patterns are required.")
        # copy pattern references
        self._patterns = list(patterns)

        return super(PatternNet, self).__init__(*args, **kwargs)

    def _create_analysis(self, *args, **kwargs):

        # shared information among the created reverse mappings
        patterns = self._patterns
        tmp_pattern_idx_stack = list(range(len(self._patterns)))

        class ReverseLayer(kgraph.ReverseMappingBase):

            def __init__(self, layer, state):
                # we want to apply the filter weights on the forward pass
                # on the backward pass we want copy the gradient computation
                # except that using the pattern weights instead of the filter w
                # i.e. we need to revert the activation with the forward
                # activations to keep the relu patterns of the gradient comp.
                # this is the reason for splitting the gradient mimicking:
                config = layer.get_config()

                activation = None
                if "activation" in config:
                    activation = config["activation"]
                    config["activation"] = None
                self._act_layer = keras.layers.Activation(
                    activation,
                    name="reversed_act_%s" % config["name"])

                self._kernel_layer = kgraph.get_layer_wo_activation(
                    layer, name_template="reversed_kernel_%s")

                # replace kernel weights with pattern weights
                pattern_weights = layer.get_weights()
                pattern = patterns[tmp_pattern_idx_stack.pop()]
                # assume that only one matches
                tmp = [pattern.shape == x.shape for x in pattern_weights]
                if np.sum(tmp) != 1:
                    raise Exception("Cannot match pattern to kernel.")
                pattern_weights[np.argmax(tmp)] = pattern
                self._pattern_layer = kgraph.get_layer_wo_activation(
                    layer,
                    name_template="reversed_pattern_%s",
                    weights=pattern_weights)

            def apply(self, Xs, Ys, reversed_Ys, reverse_state):
                act_Xs = kutils.easy_apply(self._kernel_layer, Xs)
                act_Ys = kutils.easy_apply(self._act_layer, act_Xs)
                pattern_Ys = kutils.easy_apply(self._pattern_layer, Xs)

                grad_act = ilayers.GradientWRT(len(act_Xs))
                grad_pattern = ilayers.GradientWRT(len(Xs))

                linear_activations = [None, keras.activations.get("linear")]
                if self._act_layer.activation in linear_activations:
                    tmp = reversed_Ys
                else:
                    # if linear activation this behaves strange
                    tmp = utils.listify(grad_act(act_Xs+act_Ys+reversed_Ys))

                return grad_pattern(Xs+pattern_Ys+tmp)

        self._conditional_mappings = [
            (kgraph.contains_kernel, ReverseLayer),
        ]

        ret = super(PatternNet, self)._create_analysis(*args, **kwargs)

        if len(tmp_pattern_idx_stack) == 0:
            del tmp_pattern_idx_stack
        else:
            raise Exception("Not all patterns consumed. Something is wrong.")

        return ret

    def _get_state(self):
        state = super(PatternNet, self)._get_state()
        state.update({"patterns": self._patterns})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        patterns = state.pop("patterns")
        kwargs = super(PatternNet, clazz)._state_to_kwargs(state)
        kwargs.update({"patterns": patterns})
        return kwargs


class PatternAttribution(PatternNet):

    properties = {
        "name": "PatternAttribution",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, *args, patterns=None, **kwargs):
        if patterns is None:
            raise ValueError("Patterns are required.")

        # copy pattern references
        self._patterns = list(patterns)
        # copy pattern references; will be used as stack
        patterns = list(patterns)

        # create PatternAttributions "patterns""
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

    def _get_state(self):
        state = super(PatternAttribution, self)._get_state()
        # This overwrites the patterns parameter from PatterNet.
        state.update({"patterns": self._patterns})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        # Don't pop patterns key-value-pair; PatternNet expects it.
        patterns = state["patterns"]
        kwargs = super(PatternAttribution, clazz)._state_to_kwargs(state)
        # This overwrites the patterns parameter from PatternNet.
        kwargs.update({"patterns": patterns})
        return kwargs
