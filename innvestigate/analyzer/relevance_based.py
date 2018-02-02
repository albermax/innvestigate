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
    "BaselineLRPZ",
]


###############################################################################
###############################################################################
###############################################################################


class BaselineLRPZ(base.AnalyzerNetworkBase):

    properties = {
        "name": "BaselineLRP-Z",
        "show_as": "rgb",
    }

    def __init__(self, *args, **kwargs):
        self._model_checks = [
            lambda layer: not kgraph.is_convnet_layer(layer),
        ]
        self._model_checks_msg = (
            "LRP-Z only collapses to gradient times input for "
            "(convluational) relu neural networks."
            )
        super(BaselineLRPZ, self).__init__(*args, **kwargs)

    def _create_analysis(self, model):
        gradients = utils.listify(ilayers.Gradient()(
            model.inputs+[model.outputs[0], ]))
        return [keras.layers.Multiply()([i, g])
                for i, g in zip(model.inputs, gradients)]


###############################################################################
###############################################################################
###############################################################################


class LRPBase(base.ReverseAnalyzerBase):

    properties = {
        "name": "Deconvnet",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self,
                 model, *args, rule=None,
                 first_layer_rule=None, first_layer_use_ZB=False, **kwargs):
        self._model_checks = [
            lambda layer: not kgraph.is_convnet_layer(layer),
        ]
        self._model_checks_msg = (
            "LRP is only tested for "
            "convluational neural networks."
            )

        if rule is None:
            raise ValueError("Need LRP rule.")

        def reverse_layer(Xs, Ys, Rs, reverse_state):
            # activations do not affect relevances
            # also biases are not used
            # remove them on the backward way
            layer = reverse_state["layer"]
            config = layer.get_config()
            # todo: create helper function that does this:
            config["name"] = "reversed_%s" % config["name"]
            if "activation" in config:
                config["activation"] = None
            # todo: ask for some lrp rules this might be wrong.
            if "bias" in config:
                config["use_bias"] = False
            layer_wo_a_b = layer.__class__.from_config(config)
            # filter bias weights
            # todo: this is not a secure way.
            layer_wo_a_b.set_weights([W for W in layer.get_weights()
                                      if len(W.shape) > 1])

            rule = select_rule(layer, reverse_state)

            def reverse_layer_instance(Xs, Ys, Rs, reverse_state):
                Ys = kutils.easy_apply(layer_wo_a_b, Xs)

                return ilayers.LRP(len(Xs), layer, rule)(Xs+Ys+Rs)

            return reverse_layer_instance

        self._conditional_mappings = [
            (kgraph.contains_kernel, reverse_layer),
        ]
        return super(BaseLRP, self).__init__(*args, **kwargs)

    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        # Expect Xs and Ys to have the same shapes.
        # There is not mixing of relevances as there is kernel,
        # therefore we pass them as they are.
        return reversed_Ys
