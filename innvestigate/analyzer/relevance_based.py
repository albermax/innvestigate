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
import keras
import keras.backend as K
import keras.engine.topology
import keras.models
import keras.layers
import keras.layers.convolutional
import keras.layers.core
import keras.layers.local
import keras.layers.noise
import keras.layers.normalization
import keras.layers.pooling
import numpy as np


from . import base
from .. import layers as ilayers
from .. import utils as iutils
from ..utils import keras as kutils
from ..utils.keras import checks as kchecks
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
    """LRPZ analyzer.

    Applies the "LRP-Z" algorithm to analyze the model.
    Based on the gradient times the input formula.
    This formula holds only for ReLU/MaxPooling networks, for which
    LRP-Z collapses into the stated formula.

    :param model: A Keras model.
    :param allow_lambda_layers: Approximate lambda layers with the
      gradient.
    """

    def __init__(self, model, allow_lambda_layers=False, **kwargs):
        # Inside function to not break import if Keras changes.
        BASELINELRPZ_LAYERS = (
            keras.engine.topology.InputLayer,
            keras.layers.convolutional.Conv1D,
            keras.layers.convolutional.Conv2D,
            keras.layers.convolutional.Conv2DTranspose,
            keras.layers.convolutional.Conv3D,
            keras.layers.convolutional.Conv3DTranspose,
            keras.layers.convolutional.Cropping1D,
            keras.layers.convolutional.Cropping2D,
            keras.layers.convolutional.Cropping3D,
            keras.layers.convolutional.SeparableConv1D,
            keras.layers.convolutional.SeparableConv2D,
            keras.layers.convolutional.UpSampling1D,
            keras.layers.convolutional.UpSampling2D,
            keras.layers.convolutional.UpSampling3D,
            keras.layers.convolutional.ZeroPadding1D,
            keras.layers.convolutional.ZeroPadding2D,
            keras.layers.convolutional.ZeroPadding3D,
            keras.layers.core.Activation,
            keras.layers.core.ActivityRegularization,
            keras.layers.core.Dense,
            keras.layers.core.Dropout,
            keras.layers.core.Flatten,
            keras.layers.core.Lambda,
            keras.layers.core.Masking,
            keras.layers.core.Permute,
            keras.layers.core.RepeatVector,
            keras.layers.core.Reshape,
            keras.layers.core.SpatialDropout1D,
            keras.layers.core.SpatialDropout2D,
            keras.layers.core.SpatialDropout3D,
            keras.layers.local.LocallyConnected1D,
            keras.layers.local.LocallyConnected2D,
            keras.layers.Add,
            keras.layers.Concatenate,
            keras.layers.Dot,
            keras.layers.Maximum,
            keras.layers.Minimum,
            keras.layers.Subtract,
            keras.layers.noise.AlphaDropout,
            keras.layers.noise.GaussianDropout,
            keras.layers.noise.GaussianNoise,
            keras.layers.normalization.BatchNormalization,
            keras.layers.pooling.GlobalMaxPooling1D,
            keras.layers.pooling.GlobalMaxPooling2D,
            keras.layers.pooling.GlobalMaxPooling3D,
            keras.layers.pooling.MaxPooling1D,
            keras.layers.pooling.MaxPooling2D,
            keras.layers.pooling.MaxPooling3D,
        )

        self._model_checks = [
            # todo: Check for non-linear output in general.
            {
                "check": lambda layer: kchecks.contains_activation(
                    layer, activation="softmax"),
                "type": "exception",
                "message": "Model should not contain a softmax.",
            },
            {
                "check":
                lambda layer: not kchecks.only_relu_activation(layer),
                "type": "exception",
                "message": ("BaselineLRPZ is not working for "
                            "networks with non-ReLU activations."),
            },
            {
                "check":
                lambda layer: not isinstance(layer, BASELINELRPZ_LAYERS),
                "type": "exception",
                "message": ("BaselineLRPZ is only defined for "
                            "certain layers."),
            },
            {
                "check":
                lambda layer: (not allow_lambda_layers and
                               isinstance(layer, keras.layers.core.Lambda)),
                "type": "exception",
                "message": ("Lamda layers are not allowed. "
                            "To allow use allow_lambda_layers kw."),
            },
        ]

        self._allow_lambda_layers = allow_lambda_layers

        super(BaselineLRPZ, self).__init__(model, **kwargs)

    def _create_analysis(self, model):
        gradients = iutils.to_list(ilayers.Gradient()(
            model.inputs+[model.outputs[0], ]))
        return [keras.layers.Multiply()([i, g])
                for i, g in zip(model.inputs, gradients)]

    def _get_state(self):
        state = super(BaselineLRPZ, self)._get_state()
        state.update({"allow_lambda_layers": self._allow_lambda_layers})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        allow_lambda_layers = state.pop("allow_lambda_layers")
        kwargs = super(BaselineLRPZ, clazz)._state_to_kwargs(state)
        kwargs.update({"allow_lambda_layers": allow_lambda_layers})
        return kwargs


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
        Zs = kutils.apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.to_list(grad(Xs+Zs+tmp))
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
        Zs = kutils.apply(self._layer_wo_act_b_positive, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.to_list(grad(Xs+Zs+tmp))
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
        Zs = kutils.apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.Divide()([a, prepare_div(b)])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.to_list(grad(Xs+Zs+tmp))
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
        Ys = kutils.apply(self._layer_wo_act_b, Xs)

        # Compute the sum of the squared weights.
        ones = ilayers.OnesLike()(Xs)
        Zs = iutils.to_list(self._layer_wo_act_b(ones))
        # Weight the incoming relevance.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Redistribute the relevances along the gradient.
        tmp = iutils.to_list(grad(Xs+Ys+Rs))
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
        Ys = kutils.apply(self._layer_wo_act_b, Xs)

        # Compute the sum of the one-weights.
        ones = ilayers.OnesLike()(Xs)
        Zs = iutils.to_list(self._layer_wo_act_b(ones))
        # Weight the incoming relevance.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Redistribute the relevances along the gradient.
        tmp = iutils.to_list(grad(Xs+Ys+tmp))
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
            Zs = kutils.apply(layer, Xs)
            # Divide incoming relevance by the activations.
            tmp = [ilayers.SafeDivide()([a, b])
                   for a, b in zip(Rs, Zs)]
            # Propagate the relevance to input neurons
            # using the gradient.
            tmp = iutils.to_list(grad(Xs+Zs+tmp))
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

            A = kutils.apply(self._layer_wo_act, Xs)
            B = kutils.apply(self._layer_wo_act_positive, low)
            C = kutils.apply(self._layer_wo_act_negative, high)
            return [keras.layers.Subtract()([a, keras.layers.Add()([b, c])])
                    for a, b, c in zip(A, B, C)]

        # Get values for the division.
        Zs = f(Xs)
        # Divide relevances with the value.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Distribute along the gradient.
        tmp = iutils.to_list(grad(Xs+Zs+tmp))
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
    """
    Base class for LRP-based model analyzers


    :param model: A Keras model.

    :param rule: A rule can be a  string or a Rule object, lists thereof or a list of conditions [(Condition, Rule), ... ]
      gradient.

    :param input_layer_rule: either a Rule object, atuple of (low, high) the min/max pixel values of the inputs
    """

    def __init__(self,
                 model, *args,
                 rule=None,
                 input_layer_rule=None,
                 **kwargs):
        self._model_checks = [
            # TODO: Check for non-linear output in general.
            {
                "check": lambda layer: kchecks.contains_activation(
                    layer, activation="softmax"),
                "type": "exception",
                "message": "Model should not contain a softmax.",
            },
            {
                "check": lambda layer: not kchecks.is_convnet_layer(layer),
                "type": "warning",
                "message": ("LRP is only tested for "
                            "convolutional neural networks."),
            },
        ]


        # check if rule was given explicitly.
        # TODO: make LRP-Z the default behaviour / rule
        # TODO: create PRESETs for LRP, e.g. alphabeta for conv, eps/z für linear layers
        # rule can be a string, a list (of strings) or a list of conditions [(Condition, Rule), ... ] for each layer.
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
           (inspect.isclass(rule) and issubclass(rule, kgraph.ReverseMappingBase)) # NOTE: All LRP rules inherit from kgraph.ReverseMappingBase
        ):
            # the given rule is a single strig or single rule implementing class
            use_conditions = True
            rules = [(lambda a, b: True, rule)]

        elif not isinstance(rule[0], tuple):
            # rule list of rule strings or classes
            use_conditions = False
            rules = list(rule)
        else:
            # rule is list of conditioned rules
            use_conditions = True
            rules = rule


        #TODO: find out: what are the assumptions here?
        if self._input_layer_rule is not None:
            input_layer_rule = self._input_layer_rule
            if isinstance(input_layer_rule, tuple):
                low, high = input_layer_rule

                # TODO avoid ad-hoc class definition here. Try to use Bounded Rule.
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
            # TODO: check if use_conditions functions properly. should be class variable.
            if use_conditions is True:
                for condition, rule in rules:
                    if condition(layer, reverse_state):
                        return rule
                raise Exception("No rule applies to layer: %s" % layer)
            else:
                return rules.pop()


        class ReverseLayer(kgraph.ReverseMappingBase):
            # TODO: refactor as independent class?
            def __init__(self, layer, state):
                rule_class = select_rule(layer, state)
                if isinstance(rule_class, six.string_types):
                    rule_class = LRP_RULES[rule]
                self._rule = rule_class(layer, state) #TODO. add def call() to Rule base class, which is a setter for layer, state, to avoid above ad-hoc-input rule class gen.

            def apply(self, Xs, Ys, Rs, reverse_state):
                print(reverse_state['layer'].__class__.__name__, 'Rule.apply kicking in for rule', self._rule)
                return self._rule.apply(Xs, Ys, Rs, reverse_state)


        # conditional mappings layer_criterion -> Rule on how to handle backward passes through layers.
        self._conditional_mappings = [
            (kchecks.contains_kernel, ReverseLayer),
        ]

        # finalize constructor.
        return super(LRP, self).__init__(model, *args, **kwargs)




    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        print(reverse_state['layer'].__class__.__name__, '_default_reverse_layer kicking in', end=':')
        if(len(Xs) == len(Ys) and
           all([K.int_shape(x) == K.int_shape(y) for x, y in zip(Xs, Ys)])):
        #if isinstance(reverse_state['layer'], keras.layers.Activation): # TODO: complete this. Activation should not be everything.
            # TODO: this is not necessarily true. Do explicit layer check.
            # Expect Xs and Ys to have the same shapes.
            # There is not mixing of relevances as there is kernel,
            # therefore we pass them as they are.
            print(' just return') #TODO:DEBUG
            return reversed_Ys
        else:
            # TODO: make this more clear, here we assume to have reshape layers
            # TODO: add assert
            # TODO: BatchNorm layer should end up here (?): Implements an affine transformation.
            # TODO: Confirm that behaviour of GradientWRT. Flatten and BatchNorm are correct
            print(' ilayers.GradientWRT') #TODO:DEBUG
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
# RULE CLASSES ################################################################
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


###############################################################################
# LRP PARAM PRESETS CLASSES ###################################################
###############################################################################

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
