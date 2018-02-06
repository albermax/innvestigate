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
import numpy as np


from . import base
from .. import layers as ilayers
from .. import utils
from ..utils import keras as kutils
from ..utils.keras import graph as kgraph


__all__ = [
    "BaselineLRPZ",

    "LRP_RULES",

    "LRP",

    "LRPZ",
    "LRPZPlus",
    "LRPEpsilon",
    "LRPWSquare",
    "LRPFlat",
    "LRPAlphaBeta",
    "LRPAlpha1Beta1",
    "LRPAlpha2Beta1",
    "LRPAlpha1Beta0",
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


class EpsilonRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        self._layer_wo_act = kgraph.get_layer_wo_activation(
            layer, name_template="reversed_kernel_%s")

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        prepare_div = keras.layers.Lambda(lambda x: x + K.sign(x)*K.epsilon())

        Zs = kutils.easy_apply(self._layer_wo_act, Xs)
        tmp = [ilayers.Divide()([a, prepare_div(b)])
               for a, b in zip(Rs, Zs)]
        tmp = utils.listify(grad(Xs+Zs+tmp))
        return [keras.layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]


class ZRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        self._layer_wo_act = kgraph.get_layer_wo_activation(
            layer, name_template="reversed_kernel_%s")

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))

        Zs = kutils.easy_apply(self._layer_wo_act, Xs)
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        tmp = utils.listify(grad(Xs+Zs+tmp))
        return [keras.layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]


class ZPlusRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        self._layer_wo_act_b_positive = kgraph.get_layer_wo_activation(
            layer, keep_bias=False,
            name_template="reversed_kernel_positive_%s")
        tmp = [x * (x > 0)
               for x in self._layer_wo_act_b_positive.get_weights()]
        self._layer_wo_act_b_positive.set_weights(tmp)

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))

        Zs = kutils.easy_apply(self._layer_wo_act_b_positive, Xs)
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        tmp = utils.listify(grad(Xs+Zs+tmp))
        return [keras.layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]


class WSquareRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        self._layer_wo_act_b = kgraph.get_layer_wo_activation(
            layer, keep_bias=False, name_template="reversed_kernel_%s")
        tmp = [x**2 for x in self._layer_wo_act_b.get_weights()]
        self._layer_wo_act_b.set_weights(tmp)

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        ones_like = keras.layers.Lambda(lambda x: x * 0 + 1)
        Ys = kutils.easy_apply(self._layer_wo_act_b, Xs)

        ones = [ones_like(x) for x in Xs]
        Zs = kutils.easy_apply(self._layer_wo_act_b, ones)
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        tmp = utils.listify(grad(Xs+Ys+Rs))
        return tmp


class FlatRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        self._layer_wo_act_b = kgraph.get_layer_wo_activation(
            layer, keep_bias=False, name_template="reversed_kernel_%s")
        tmp = [np.ones_like(x) for x in self._layer_wo_act_b.get_weights()]
        self._layer_wo_act_b.set_weights(tmp)

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        ones_like = keras.layers.Lambda(lambda x: x * 0 + 1)
        Ys = kutils.easy_apply(self._layer_wo_act_b, Xs)

        ones = [ones_like(x) for x in Xs]
        Zs = kutils.easy_apply(self._layer_wo_act_b, ones)
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        tmp = utils.listify(grad(Xs+Ys+tmp))
        return tmp


class AlphaBetaRule(kgraph.ReverseMappingBase):
    # todo: this only works for relu networks, needs to be extended.
    def __init__(self, layer, state, alpha=1, beta=1):
        self._alpha = alpha
        self._beta = beta

        positive_params = [x * (x > 0) for x in layer.get_weights()]
        negative_params = [x * (x < 0) for x in layer.get_weights()]

        self._layer_wo_act_positive = kgraph.get_layer_wo_activation(
            layer,
            weights=positive_params,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act_negative = kgraph.get_layer_wo_activation(
            layer,
            weights=negative_params,
            name_template="reversed_kernel_negative_%s")

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        times_alpha = keras.layers.Lambda(lambda x: x * self._alpha)
        times_beta = keras.layers.Lambda(lambda x: x * self._beta)

        def f(layer):
            Zs = kutils.easy_apply(layer, Xs)
            tmp = [ilayers.SafeDivide()([a, b])
                   for a, b in zip(Rs, Zs)]
            tmp = utils.listify(grad(Xs+Zs+tmp))
            return [keras.layers.Multiply()([a, b])
                    for a, b in zip(Xs, tmp)]

        return [keras.layers.Subtract()([times_alpha(a), times_beta(b)])
                for a, b in zip(f(self._layer_wo_act_positive),
                                f(self._layer_wo_act_negative))]


class BoundedRule(kgraph.ReverseMappingBase):
    # todo: this only works for relu networks, needs to be extended.
    def __init__(self, layer, state, low=-1, high=1):
        self._low = low
        self._high = high

        positive_params = [x * (x > 0) for x in layer.get_weights()]
        negative_params = [x * (x < 0) for x in layer.get_weights()]

        self._layer_wo_act = kgraph.get_layer_wo_activation(
            layer,
            name_template="reversed_kernel_%s")
        self._layer_wo_act_positive = kgraph.get_layer_wo_activation(
            layer,
            weights=positive_params,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act_negative = kgraph.get_layer_wo_activation(
            layer,
            weights=negative_params,
            name_template="reversed_kernel_negative_%s")

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        to_low = keras.layers.Lambda(lambda x: x * 0 + self._low)
        to_high = keras.layers.Lambda(lambda x: x * 0 + self._high)

        def f(Xs):
            low = [to_low(x) for x in Xs]
            high = [to_high(x) for x in Xs]

            A = kutils.easy_apply(self._layer_wo_act, Xs)
            B = kutils.easy_apply(self._layer_wo_act_positive, low)
            C = kutils.easy_apply(self._layer_wo_act_negative, high)
            return [keras.layers.Subtract()([a, keras.layers.Add()([b, c])])
                    for a, b, c in zip(A, B, C)]

        Zs = f(Xs)
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        tmp = utils.listify(grad(Xs+Zs+tmp))
        return tmp

#        alpha-beta all networks
#        bias+- for some other rules
LRP_RULES = {
    "AlphaBeta": AlphaBetaRule,
    "Epsilon": EpsilonRule,
    "WSquare": WSquareRule,
    "Flat": FlatRule,
    "Z": ZRule,
    "ZPlus": ZPlusRule,
    "Bounded": BoundedRule,
}


###############################################################################
###############################################################################
###############################################################################


class LRP(base.ReverseAnalyzerBase):

    properties = {
        "name": "LRP",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self,
                 model, *args,
                 rule=None,
                 input_layer_rule=None,
                 **kwargs):
        self._model_checks = [
            lambda layer: not kgraph.is_convnet_layer(layer),
        ]
        self._model_checks_msg = (
            "LRP is only tested for "
            "convolutional neural networks."
            )

        if rule is None:
            raise ValueError("Need LRP rule(s).")

        if isinstance(rule, list):
            # copy refrences
            self._rule = list(rule)
        else:
            self._rule = rule
        self._input_layer_rule = input_layer_rule

        if(
                isinstance(rule, six.string_types) or
                (
                    inspect.isclass(rule) and
                    issubclass(rule, kgraph.ReverseMappingBase)
                )
        ):
            use_conditions = True
            rules = [(lambda a, b: True, rule)]
        elif not isinstance(rule[0], tuple):
            use_conditions = False
            rules = list(rule)
        else:
            use_conditions = True
            rules = rule

        if self._input_layer_rule is not None:
            input_layer_rule = self._input_layer_rule
            if isinstance(input_layer_rule, tuple):
                low, high = input_layer_rule

                class input_layer_rule(BoundedRule):
                    def __init__(self, *args, **kwars):
                        return super(input_layer_rule, self).__init__(
                            *args, low=low, high=high, **kwargs)

            if use_conditions is True:
                rules.insert(0,
                             (lambda layer, foo: kgraph.is_input_layer(layer),
                              input_layer_rule))
            else:
                rules.insert(0, input_layer_rule)

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
        return super(LRP, self).__init__(model, *args, **kwargs)

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
        state = super(LRP, self)._get_state()
        state.update({"rule": self._rule})
        state.update({"input_layer_rule": self._input_layer_rule})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        rule = state.pop("rule")
        input_layer_rule = state.pop("input_layer_rule")
        kwargs = super(LRP, clazz)._state_to_kwargs(state)
        kwargs.update({"rule": rule,
                       "input_layer_rule": input_layer_rule})
        return kwargs


###############################################################################
###############################################################################
###############################################################################


class LRPZ(LRP):

    properties = {
        "name": "LRP-Z",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, *args, **kwargs):
        return super(LRPZ, self).__init__(model, *args, rule="Z", **kwargs)


class LRPZPlus(LRP):

    properties = {
        "name": "LRP-ZPlus",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, *args, **kwargs):
        return super(LRPZPlus, self).__init__(model, *args,
                                              rule="ZPlus", **kwargs)


class LRPEpsilon(LRP):

    properties = {
        "name": "LRP-Epsilon",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, *args, **kwargs):
        return super(LRPEpsilon, self).__init__(model, *args,
                                                rule="Epsilon", **kwargs)


class LRPWSquare(LRP):

    properties = {
        "name": "LRP-WSquare",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, *args, **kwargs):
        return super(LRPWSquare, self).__init__(model, *args,
                                                rule="WSquare", **kwargs)


class LRPFlat(LRP):

    properties = {
        "name": "LRP-Flat",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, *args, **kwargs):
        return super(LRPFlat, self).__init__(model, *args,
                                             rule="Flat", **kwargs)


class LRPAlphaBeta(LRP):

    properties = {
        "name": "LRP-AlphaBeta",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, alpha=1, beta=1, *args, **kwargs):

        class CustomizedAlphaBetaRule(AlphaBetaRule):
            def __init__(self, *args, **kwargs):

                super(CustomizedAlphaBetaRule, self).__init__(*args,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              **kwargs)

        return super(LRPAlphaBeta, self).__init__(model, *args,
                                                  rule=CustomizedAlphaBetaRule,
                                                  **kwargs)


class LRPAlpha1Beta1(LRPAlphaBeta):

    properties = {
        "name": "LRP-Alpha1Beta1",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha1Beta1, self).__init__(model, *args,
                                                    alpha=1,
                                                    beta=1,
                                                    **kwargs)


class LRPAlpha2Beta1(LRPAlphaBeta):

    properties = {
        "name": "LRP-Alpha2Beta1",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha2Beta1, self).__init__(model, *args,
                                                    alpha=2,
                                                    beta=1,
                                                    **kwargs)


class LRPAlpha1Beta0(LRPAlphaBeta):

    properties = {
        "name": "LRP-Alpha1Beta0",
        # todo: set right value
        "show_as": "rgb",
    }

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha1Beta0, self).__init__(model, *args,
                                                    alpha=1,
                                                    beta=0,
                                                    **kwargs)
