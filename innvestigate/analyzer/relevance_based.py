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


import inspect
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
    "LRPBase",
    "LRPZ",
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


class LRPZRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        self._layer_wo_act = kgraph.get_layer_wo_activation(
            layer, name_template="reversed_kernel_%s")

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))

        Zs = kutils.easy_apply(self._layer_wo_act, Xs)
        tmp = [ilayers.Divide()([a, b])
               for a, b in zip(Rs, Zs)]
        tmp = utils.listify(grad(Xs+Zs+tmp))
        return [keras.layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]


LRP_RULES = {
    "Z": LRPZRule,
}


class LRPBase(base.ReverseAnalyzerBase):

    properties = {
        "name": "LRPBase",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self,
                 model, *args,
                 rule=None, first_layer_rule=None, **kwargs):
        self._model_checks = [
            lambda layer: not kgraph.is_convnet_layer(layer),
        ]
        self._model_checks_msg = (
            "LRP is only tested for "
            "convluational neural networks."
            )

        if rule is None:
            raise ValueError("Need LRP rule(s).")

        if isinstance(rule, list):
            # copy refrences
            self._rule = list(rule)
        else:
            self._rule = rule
        self._first_layer_rule = first_layer_rule

        if(inspect.isclass(rule) and
           issubclass(rule, kgraph.ReverseMappingBase)):
            use_conditions = True
            rules = [(lambda a, b: True, rule)]
        elif not isinstance(rule[0], tuple):
            use_conditions = False
            rules = list(rule)
        else:
            use_conditions = True
            rules = rule

        def select_rule(layer, reverse_state):
            if use_conditions is True:
                for condition, rule in rules:
                    if condition(layer, reverse_state):
                        return rule
                raise Exception("No rule applies to layer: %s" % layer)
            else:
                return rules.pop()

        class ReverseLayer(kgraph.ReverseMappingBase):

            def __init__(self, layer, state):
                rule_class = select_rule(layer, state)
                if isinstance(rule_class, six.string_types):
                    rule_class = LRP_RULES[rule]
                self._rule = rule_class(layer, state)

            def apply(self, Xs, Ys, Rs, reverse_state):
                return self._rule.apply(Xs, Ys, Rs, reverse_state)

        self._conditional_mappings = [
            (kgraph.contains_kernel, ReverseLayer),
        ]
        return super(LRPBase, self).__init__(model, *args, **kwargs)

    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        if(len(Xs) == len(Ys) and
           all([K.int_shape(x) == K.int_shape(y) for x, y in zip(Xs, Ys)])):
            # Expect Xs and Ys to have the same shapes.
            # There is not mixing of relevances as there is kernel,
            # therefore we pass them as they are.
            return reversed_Ys
        else:
            # todo: make this more clear, here we assume to have rehape layers
            # todo: add assert
            return ilayers.GradientWRT(len(Xs))(Xs+Ys+reversed_Ys)

    def _get_state(self):
        state = super(LRPBase, self)._get_state()
        state.update({"rule": self._rule})
        state.update({"first_layer_rule": self._first_layer_rule})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        rule = state.pop("rule")
        first_layer_rule = state.pop("first_layer_rule")
        kwargs = super(LRPBase, clazz)._state_to_kwargs(state)
        kwargs.update({"rule": rule,
                       "first_layer_rule": first_layer_rule})
        return kwargs


class LRPZ(LRPBase):

    properties = {
        "name": "LRP-Z",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, *args, **kwargs):
        return super(LRPZ, self).__init__(model, *args, rule="Z", **kwargs)
