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
from .. import utils as iutils
from ..utils import keras as kutils
from ..utils.keras import graph as kgraph


__all__ = [
    "BaselineLRPZ",

    "LRP_RULES",

    "LRP",

    "LRPZ",
    "LRPZWithBias",
    "LRPZPlus",
    "LRPEpsilon",
    "LRPEpsilonWithBias",
    "LRPWSquare",
    "LRPFlat",
    "LRPAlphaBeta",
    "LRPAlpha1Beta1",
    "LRPAlpha1Beta1WithBias",
    "LRPAlpha2Beta1",
    "LRPAlpha2Beta1WithBias",
    "LRPAlpha1Beta0",
    "LRPAlpha1Beta0WithBias",
]


###############################################################################
###############################################################################
###############################################################################


class BaselineLRPZ(base.AnalyzerNetworkBase):

    def __init__(self, *args, **kwargs):
        self._model_checks = [
            (lambda layer: not kgraph.is_convnet_layer(layer),
             "LRP-Z only collapses to gradient times input for "
             "(convolutional) relu neural networks."),
            # todo: Check for non-linear output in general.
            (lambda layer: kgraph.contains_activation(layer,
                                                      activation="softmax"),
             "Model should not contain a softmax.")
        ]
        super(BaselineLRPZ, self).__init__(*args, **kwargs)

    def _create_analysis(self, model):
        gradients = iutils.listify(ilayers.Gradient()(
            model.inputs+[model.outputs[0], ]))
        return [keras.layers.Multiply()([i, g])
                for i, g in zip(model.inputs, gradients)]


###############################################################################
###############################################################################
###############################################################################


class ZRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state, bias=False):
        self._layer_wo_act = kgraph.copy_layer_wo_activation(
            layer, keep_bias=bias, name_template="reversed_kernel_%s")

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))

        # Get activations.
        Zs = kutils.easy_apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.listify(grad(Xs+Zs+tmp))
        # Re-weight relevance with the input values.
        return [keras.layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]


class ZWithBiasRule(ZRule):
    def __init__(self, *args, **kwargs):
        return super(ZWithBiasRule, self).__init__(*args,
                                                   bias=True, **kwargs)


# todo: make subclass of ZRule
class ZPlusRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        # The z-plus rule only works with positive weights and
        # no biases.
        self._layer_wo_act_b_positive = kgraph.copy_layer_wo_activation(
            layer, keep_bias=False,
            name_template="reversed_kernel_positive_%s")
        tmp = [x * (x > 0)
               for x in self._layer_wo_act_b_positive.get_weights()]
        self._layer_wo_act_b_positive.set_weights(tmp)

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))

        # Get activations.
        Zs = kutils.easy_apply(self._layer_wo_act_b_positive, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.listify(grad(Xs+Zs+tmp))
        # Re-weight relevance with the input values.
        return [keras.layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]


class EpsilonRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state, bias=False):
        self._layer_wo_act = kgraph.copy_layer_wo_activation(
            layer, keep_bias=bias, name_template="reversed_kernel_%s")

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        # The epsilon rule aligns epsilon with the sign.
        prepare_div = keras.layers.Lambda(lambda x: x + K.sign(x)*K.epsilon())

        # Get activations.
        Zs = kutils.easy_apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.Divide()([a, prepare_div(b)])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.listify(grad(Xs+Zs+tmp))
        # Re-weight relevance with the input values.
        return [keras.layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]


class EpsilonWithBiasRule(EpsilonRule):
    def __init__(self, *args, **kwargs):
        return super(EpsilonWithBiasRule, self).__init__(*args,
                                                         bias=True, **kwargs)


class WSquareRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        # W-square rule works with squared weights and no biases.
        self._layer_wo_act_b = kgraph.copy_layer_wo_activation(
            layer, keep_bias=False, name_template="reversed_kernel_%s")
        tmp = [x**2 for x in self._layer_wo_act_b.get_weights()]
        self._layer_wo_act_b.set_weights(tmp)

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        # Create dummy forward path to take the derivative below.
        Ys = kutils.easy_apply(self._layer_wo_act_b, Xs)

        # Compute the sum of the squared weights.
        ones = ilayers.OnesLike()(Xs)
        Zs = iutils.listify(self._layer_wo_act_b(ones))
        # Weight the incoming relevance.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Redistribute the relevances along the gradient.
        tmp = iutils.listify(grad(Xs+Ys+Rs))
        return tmp


# todo: Make sublcass of WSquare rule
class FlatRule(kgraph.ReverseMappingBase):

    def __init__(self, layer, state):
        # The flat rule works with weights equal to one and
        # no biases.
        self._layer_wo_act_b = kgraph.copy_layer_wo_activation(
            layer, keep_bias=False, name_template="reversed_kernel_%s")
        tmp = [np.ones_like(x) for x in self._layer_wo_act_b.get_weights()]
        self._layer_wo_act_b.set_weights(tmp)

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        # Create dummy forward path to take the derivative below.
        Ys = kutils.easy_apply(self._layer_wo_act_b, Xs)

        # Compute the sum of the one-weights.
        ones = ilayers.OnesLike()(Xs)
        Zs = iutils.listify(self._layer_wo_act_b(ones))
        # Weight the incoming relevance.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Redistribute the relevances along the gradient.
        tmp = iutils.listify(grad(Xs+Ys+tmp))
        return tmp


# todo: could potentially inherit from and use ZRule.
class AlphaBetaRule(kgraph.ReverseMappingBase):
    # todo: this only works for relu networks, needs to be extended.
    def __init__(self, layer, state, alpha=1, beta=1, bias=False):
        self._alpha = alpha
        self._beta = beta

        # Positive part works with layer with only positive
        # weights and biases.
        # The negative part does accordingly.
        self._layer_wo_act_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act_negative = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            name_template="reversed_kernel_negative_%s")
        positive_weights = [x * (x > 0)
                            for x in self._layer_wo_act_positive.get_weights()]
        negative_weights = [x * (x < 0)
                            for x in self._layer_wo_act_negative.get_weights()]
        self._layer_wo_act_positive.set_weights(positive_weights)
        self._layer_wo_act_negative.set_weights(negative_weights)

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        times_alpha = keras.layers.Lambda(lambda x: x * self._alpha)
        times_beta = keras.layers.Lambda(lambda x: x * self._beta)

        def f(layer):
            # Get activations.
            Zs = kutils.easy_apply(layer, Xs)
            # Divide incoming relevance by the activations.
            tmp = [ilayers.SafeDivide()([a, b])
                   for a, b in zip(Rs, Zs)]
            # Propagate the relevance to input neurons
            # using the gradient.
            tmp = iutils.listify(grad(Xs+Zs+tmp))
            # Re-weight relevance with the input values.
            return [keras.layers.Multiply()([a, b])
                    for a, b in zip(Xs, tmp)]

        # Compute positive and negative relevance.'
        positive_part = f(self._layer_wo_act_positive)
        negative_part = f(self._layer_wo_act_negative)
        # Join weighted positive and negative relevance again.
        return [keras.layers.Subtract()([times_alpha(a), times_beta(b)])
                for a, b in zip(positive_part, negative_part)]


class AlphaBetaWithBiasRule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(AlphaBetaWithBiasRule, self).__init__(*args,
                                                           bias=True,
                                                           **kwargs)


class Alpha1Beta1Rule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(Alpha1Beta1Rule, self).__init__(*args,
                                                     alpha=1,
                                                     beta=1,
                                                     bias=False,
                                                     **kwargs)


class Alpha1Beta1WithBiasRule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(Alpha1Beta1WithBiasRule, self).__init__(*args,
                                                             alpha=1,
                                                             beta=1,
                                                             bias=True,
                                                             **kwargs)


class Alpha2Beta1Rule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(Alpha2Beta1Rule, self).__init__(*args,
                                                     alpha=2,
                                                     beta=1,
                                                     bias=False,
                                                     **kwargs)


class Alpha2Beta1WithBiasRule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(Alpha2Beta1WithBiasRule, self).__init__(*args,
                                                             alpha=2,
                                                             beta=1,
                                                             bias=True,
                                                             **kwargs)


class Alpha1Beta0Rule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(Alpha1Beta0Rule, self).__init__(*args,
                                                     alpha=1,
                                                     beta=0,
                                                     bias=False,
                                                     **kwargs)


class Alpha1Beta0WithBiasRule(AlphaBetaRule):
    def __init__(self, *args, **kwargs):
        return super(Alpha1Beta0WithBiasRule, self).__init__(*args,
                                                             alpha=1,
                                                             beta=0,
                                                             bias=True,
                                                             **kwargs)


class BoundedRule(kgraph.ReverseMappingBase):
    # todo: this only works for relu networks, needs to be extended.
    def __init__(self, layer, state, low=-1, high=1):
        self._low = low
        self._high = high

        # This rule works with three variants of the layer, all without biases.
        # One is the original form and two with only the positive or
        # negative weights.
        self._layer_wo_act = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            name_template="reversed_kernel_%s")
        self._layer_wo_act_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act_negative = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            name_template="reversed_kernel_negative_%s")
        positive_weights = [x * (x > 0)
                            for x in self._layer_wo_act_positive.get_weights()]
        negative_weights = [x * (x < 0)
                            for x in self._layer_wo_act_negative.get_weights()]
        self._layer_wo_act_positive.set_weights(positive_weights)
        self._layer_wo_act_negative.set_weights(negative_weights)

    # todo: check if this is a correct implementation.
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

        # Get values for the division.
        Zs = f(Xs)
        # Divide relevances with the value.    
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Distribute along the gradient.
        tmp = iutils.listify(grad(Xs+Zs+tmp))
        return tmp


#        alpha-beta all networks
#        bias+- for some other rules
LRP_RULES = {
    "Z": ZRule,
    "ZWithBias": ZWithBiasRule,
    "ZPlus": ZPlusRule,
    "Epsilon": EpsilonRule,
    "EpsilonWithBias": EpsilonWithBiasRule,
    "WSquare": WSquareRule,
    "Flat": FlatRule,
    "AlphaBeta": AlphaBetaRule,
    "AlphaWithBiasBeta": AlphaBetaWithBiasRule,
    "Alpha1Beta1": Alpha1Beta1Rule,
    "Alpha1Beta1WithBias": Alpha1Beta1WithBiasRule,
    "Alpha2Beta1": Alpha2Beta1Rule,
    "Alpha2Beta1WithBias": Alpha2Beta1WithBiasRule,
    "Alpha1Beta0": Alpha1Beta0Rule,
    "Alpha1Beta0WithBias": Alpha1Beta0WithBiasRule,
    "Bounded": BoundedRule,
}


###############################################################################
###############################################################################
###############################################################################


class LRP(base.ReverseAnalyzerBase):

    def __init__(self,
                 model, *args,
                 rule=None,
                 input_layer_rule=None,
                 **kwargs):
        self._model_checks = [
            (lambda layer: not kgraph.is_convnet_layer(layer),
             "LRP is only tested for "
             "convolutional neural networks."),
            # todo: Check for non-linear output in general.
            (lambda layer: kgraph.contains_activation(layer,
                                                      activation="softmax"),
             "Model should not contain a softmax.")
        ]

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
            # Todo: this is not necessarily true. Do explicit layer check.
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


class _LRPFixedParams(LRP):

    @classmethod
    def _state_to_kwargs(clazz, state):
        kwargs = super(_LRPFixedParams, clazz)._state_to_kwargs(state)
        del kwargs["rule"]
        return kwargs


class LRPZ(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPZ, self).__init__(model, *args, rule="Z", **kwargs)


class LRPZWithBias(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPZWithBias, self).__init__(model, *args,
                                                  rule="ZWithBias", **kwargs)


class LRPZPlus(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPZPlus, self).__init__(model, *args,
                                              rule="ZPlus", **kwargs)


class LRPEpsilon(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPEpsilon, self).__init__(model, *args,
                                                rule="Epsilon", **kwargs)


class LRPEpsilonWithBias(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPEpsilonWithBias, self).__init__(model, *args,
                                                        rule="EpsilonWithBias",
                                                        **kwargs)


class LRPWSquare(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPWSquare, self).__init__(model, *args,
                                                rule="WSquare", **kwargs)


class LRPFlat(_LRPFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPFlat, self).__init__(model, *args,
                                             rule="Flat", **kwargs)


class LRPAlphaBeta(LRP):

    def __init__(self, model, alpha=1, beta=1, bias=False, *args, **kwargs):
        self._alpha = alpha
        self._beta = beta
        self._bias = bias

        class CustomizedAlphaBetaRule(AlphaBetaRule):
            def __init__(self, *args, **kwargs):

                super(CustomizedAlphaBetaRule, self).__init__(*args,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              bias=bias,
                                                              **kwargs)

        return super(LRPAlphaBeta, self).__init__(model, *args,
                                                  rule=CustomizedAlphaBetaRule,
                                                  **kwargs)

    def _get_state(self):
        state = super(LRPAlphaBeta, self)._get_state()
        del state["rule"]
        state.update({"alpha": self._alpha})
        state.update({"beta": self._beta})
        state.update({"bias": self._bias})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        alpha = state.pop("alpha")
        beta = state.pop("beta")
        bias = state.pop("bias")
        state["rule"] = None
        kwargs = super(LRPAlphaBeta, clazz)._state_to_kwargs(state)
        del kwargs["rule"]
        kwargs.update({"alpha": alpha,
                       "beta": beta,
                       "bias": bias})
        return kwargs


class _LRPAlphaBetaFixedParams(LRPAlphaBeta):

    @classmethod
    def _state_to_kwargs(clazz, state):
        kwargs = super(_LRPAlphaBetaFixedParams, clazz)._state_to_kwargs(state)
        del kwargs["alpha"]
        del kwargs["beta"]
        del kwargs["bias"]
        return kwargs


class LRPAlpha1Beta1(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha1Beta1, self).__init__(model, *args,
                                                    alpha=1,
                                                    beta=1,
                                                    bias=False,
                                                    **kwargs)


class LRPAlpha1Beta1WithBias(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha1Beta1WithBias, self).__init__(model, *args,
                                                            alpha=1,
                                                            beta=1,
                                                            bias=True,
                                                            **kwargs)


class LRPAlpha2Beta1(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha2Beta1, self).__init__(model, *args,
                                                    alpha=2,
                                                    beta=1,
                                                    bias=False,
                                                    **kwargs)


class LRPAlpha2Beta1WithBias(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha2Beta1WithBias, self).__init__(model, *args,
                                                            alpha=2,
                                                            beta=1,
                                                            bias=True,
                                                            **kwargs)


class LRPAlpha1Beta0(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha1Beta0, self).__init__(model, *args,
                                                    alpha=1,
                                                    beta=0,
                                                    bias=False,
                                                    **kwargs)


class LRPAlpha1Beta0WithBias(_LRPAlphaBetaFixedParams):

    def __init__(self, model, *args, **kwargs):
        return super(LRPAlpha1Beta0WithBias, self).__init__(model, *args,
                                                            alpha=1,
                                                            beta=0,
                                                            bias=True,
                                                            **kwargs)
