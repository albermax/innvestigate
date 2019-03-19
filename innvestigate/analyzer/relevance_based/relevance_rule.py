# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import zip


###############################################################################
###############################################################################
###############################################################################

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


from innvestigate import layers as ilayers
from innvestigate import utils as iutils
import innvestigate.utils.keras as kutils
from innvestigate.utils.keras import backend as iK
from innvestigate.utils.keras import graph as kgraph
from . import utils as rutils


# TODO: differentiate between LRP and DTD rules?
# DTD rules are special cases of LRP rules with additional assumptions
__all__ = [
    #dedicated treatment for special layers


    #general rules
    "ZRule",
    "ZIgnoreBiasRule",

    "EpsilonRule",
    "EpsilonIgnoreBiasRule",

    "WSquareRule",
    "FlatRule",

    "AlphaBetaRule",
    "AlphaBetaIgnoreBiasRule",

    "Alpha2Beta1Rule",
    "Alpha2Beta1IgnoreBiasRule",

    "Alpha1Beta0Rule",
    "Alpha1Beta0IgnoreBiasRule",

    "ZPlusRule",
    "ZPlusFastRule",
    "BoundedRule"
]



class ZRule(kgraph.ReverseMappingBase):
    """
    Basic LRP decomposition rule (for layers with weight kernels),
    which considers the bias a constant input neuron.
    """

    def __init__(self, layer, state, bias=True):
        self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                             keep_bias=bias,
                                                             name_template="reversed_kernel_%s")

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



class ZIgnoreBiasRule(ZRule):
    """
    Basic LRP decomposition rule, ignoring the bias neuron
    """
    def __init__(self, *args, **kwargs):
        super(ZIgnoreBiasRule, self).__init__(*args,
                                              bias=False,
                                              **kwargs)



class EpsilonRule(kgraph.ReverseMappingBase):
    """
    Similar to ZRule.
    The only difference is the addition of a numerical stabilizer term
    epsilon to the decomposition function's denominator.
    the sign of epsilon depends on the sign of the output activation
    0 is considered to be positive, ie sign(0) = 1
    """

    def __init__(self, layer, state, epsilon = 1e-7, bias=True):
        self._epsilon = rutils.assert_lrp_epsilon_param(epsilon, self)
        self._layer_wo_act = kgraph.copy_layer_wo_activation(
            layer, keep_bias=bias, name_template="reversed_kernel_%s")


    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        # The epsilon rule aligns epsilon with the (extended) sign: 0 is considered to be positive
        prepare_div = keras.layers.Lambda(lambda x: x + (K.cast(K.greater_equal(x,0), K.floatx())*2-1)*self._epsilon)

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



class EpsilonIgnoreBiasRule(EpsilonRule):
    """Same as EpsilonRule but ignores the bias."""
    def __init__(self, *args, **kwargs):
        super(EpsilonIgnoreBiasRule, self).__init__(*args,
                                                    bias=False,
                                                    **kwargs)



class WSquareRule(kgraph.ReverseMappingBase):
    """W**2 rule from Deep Taylor Decomposition"""

    def __init__(self, layer, state, copy_weights=False):
        # W-square rule works with squared weights and no biases.
        if copy_weights:
            weights = layer.get_weights()
        else:
            weights = layer.weights
        if layer.use_bias:
            weights = weights[:-1]
        weights = [x**2 for x in weights]

        self._layer_wo_act_b = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=weights,
            name_template="reversed_kernel_%s")


    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        # Create dummy forward path to take the derivative below.
        Ys = kutils.apply(self._layer_wo_act_b, Xs)

        # Compute the sum of the weights.
        ones = ilayers.OnesLike()(Xs)
        Zs = iutils.to_list(self._layer_wo_act_b(ones))
        # Weight the incoming relevance.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Redistribute the relevances along the gradient.
        tmp = iutils.to_list(grad(Xs+Ys+tmp))
        return tmp




class FlatRule(WSquareRule):
    """Same as W**2 rule but sets all weights to ones."""

    def __init__(self, layer, state, copy_weights=False):
        # The flat rule works with weights equal to one and
        # no biases.
        if copy_weights:
            weights = layer.get_weights()
            if layer.use_bias:
                weights = weights[:-1]
            weights = [np.ones_like(x) for x in weights]
        else:
            weights = layer.weights
            if layer.use_bias:
                weights = weights[:-1]
            weights = [K.ones_like(x) for x in weights]

        self._layer_wo_act_b = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=weights,
            name_template="reversed_kernel_%s")




class AlphaBetaRule(kgraph.ReverseMappingBase):
    """
    This decomposition rule handles the positive forward
    activations (x*w > 0) and negative forward activations
    (w * x < 0) independently, reducing the risk of zero
    divisions considerably. In fact, the only case where
    divisions by zero can happen is if there are either
    no positive or no negative parts to the activation
    at all.
    Corresponding parameterization of this rule implement
    methods such as Excitation Backpropagation with
    alpha=1, beta=0
    s.t.
    alpha - beta = 1 (after current param. scheme.)
    and
    alpha > 1
    beta > 0
    """


    def __init__(self,
                 layer,
                 state,
                 alpha=None,
                 beta=None,
                 bias=True,
                 copy_weights=False):
        alpha, beta = rutils.assert_infer_lrp_alpha_beta_param(alpha, beta, self)
        self._alpha = alpha
        self._beta = beta

        # prepare positive and negative weights for computing positive
        # and negative preactivations z in apply_accordingly.
        if copy_weights:
            weights = layer.get_weights()
            if not bias and layer.use_bias:
                weights = weights[:-1]
            positive_weights = [x * (x > 0) for x in weights]
            negative_weights = [x * (x < 0) for x in weights]
        else:
            weights = layer.weights
            if not bias and layer.use_bias:
                weights = weights[:-1]
            positive_weights = [x * iK.to_floatx(x > 0) for x in weights]
            negative_weights = [x * iK.to_floatx(x < 0) for x in weights]

        self._layer_wo_act_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=positive_weights,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act_negative = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=bias,
            weights=negative_weights,
            name_template="reversed_kernel_negative_%s")


    def apply(self, Xs, Ys, Rs, reverse_state):
        #this method is correct, but wasteful
        grad = ilayers.GradientWRT(len(Xs))
        times_alpha = keras.layers.Lambda(lambda x: x * self._alpha)
        times_beta = keras.layers.Lambda(lambda x: x * self._beta)
        keep_positives = keras.layers.Lambda(lambda x: x * K.cast(K.greater(x,0), K.floatx()))
        keep_negatives = keras.layers.Lambda(lambda x: x * K.cast(K.less(x,0), K.floatx()))


        def f(layer1, layer2, X1, X2):
            # Get activations of full positive or negative part.
            Z1 = kutils.apply(layer1, X1)
            Z2 = kutils.apply(layer2, X2)
            Zs = [keras.layers.Add()([a, b])
                    for a, b in zip(Z1, Z2)]
            # Divide incoming relevance by the activations.
            tmp = [ilayers.SafeDivide()([a, b])
                    for a, b in zip(Rs, Zs)]
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = iutils.to_list(grad(X1+Z1+tmp))
            tmp2 = iutils.to_list(grad(X2+Z2+tmp))
            # Re-weight relevance with the input values.
            tmp1 = [keras.layers.Multiply()([a, b])
                    for a, b in zip(X1, tmp1)]
            tmp2 = [keras.layers.Multiply()([a, b])
                    for a, b in zip(X2, tmp2)]
            #combine and return
            return [keras.layers.Add()([a, b])
                    for a, b in zip(tmp1, tmp2)]


        # Distinguish postive and negative inputs.
        Xs_pos = kutils.apply(keep_positives, Xs)
        Xs_neg = kutils.apply(keep_negatives, Xs)
        # xpos*wpos + xneg*wneg
        activator_relevances = f(self._layer_wo_act_positive,
                                 self._layer_wo_act_negative,
                                 Xs_pos, Xs_neg)

        if self._beta: #only compute beta-weighted contributions of beta is not zero
            # xpos*wneg + xneg*wpos
            inhibitor_relevances = f(self._layer_wo_act_negative,
                                     self._layer_wo_act_positive,
                                     Xs_pos, Xs_neg)
            return [keras.layers.Subtract()([times_alpha(a), times_beta(b)])
                        for a, b in zip(activator_relevances, inhibitor_relevances)]
        else:
            return activator_relevances



        
class AlphaBetaIgnoreBiasRule(AlphaBetaRule):
    """Same as AlphaBetaRule but ignores biases."""
    def __init__(self, *args, **kwargs):
        super(AlphaBetaIgnoreBiasRule, self).__init__(*args,
                                                      bias=False,
                                                      **kwargs)



class Alpha2Beta1Rule(AlphaBetaRule):
    """AlphaBetaRule with alpha=2, beta=1"""
    def __init__(self, *args, **kwargs):
        super(Alpha2Beta1Rule, self).__init__(*args,
                                              alpha=2,
                                              beta=1,
                                              bias=True,
                                              **kwargs)


class Alpha2Beta1IgnoreBiasRule(AlphaBetaRule):
    """AlphaBetaRule with alpha=2, beta=1 and ignores biases"""
    def __init__(self, *args, **kwargs):
        super(Alpha2Beta1IgnoreBiasRule, self).__init__(*args,
                                                        alpha=2,
                                                        beta=1,
                                                        bias=False,
                                                        **kwargs)


class Alpha1Beta0Rule(AlphaBetaRule):
    """AlphaBetaRule with alpha=1, beta=0"""
    def __init__(self, *args, **kwargs):
        super(Alpha1Beta0Rule, self).__init__(*args,
                                              alpha=1,
                                              beta=0,
                                              bias=True,
                                              **kwargs)


class Alpha1Beta0IgnoreBiasRule(AlphaBetaRule):
    """AlphaBetaRule with alpha=1, beta=0 and ignores biases"""
    def __init__(self, *args, **kwargs):
        super(Alpha1Beta0IgnoreBiasRule, self).__init__(*args,
                                                        alpha=1,
                                                        beta=0,
                                                        bias=False,
                                                        **kwargs)



class BoundedRule(kgraph.ReverseMappingBase):
    """Z_B rule from the Deep Taylor Decomposition"""
    # TODO: this only works for relu networks, needs to be extended.
    # TODO: check
    def __init__(self, layer, state, low=-1, high=1, copy_weights=False):
        self._low = low
        self._high = high

        # This rule works with three variants of the layer, all without biases.
        # One is the original form and two with only the positive or
        # negative weights.
        if copy_weights:
            weights = layer.get_weights()
            if layer.use_bias:
                weights = weights[:-1]
            positive_weights = [x * (x > 0) for x in weights]
            negative_weights = [x * (x < 0) for x in weights]
        else:
            weights = layer.weights
            if layer.use_bias:
                weights = weights[:-1]
            positive_weights = [x * iK.to_floatx(x > 0) for x in weights]
            negative_weights = [x * iK.to_floatx(x < 0) for x in weights]

        self._layer_wo_act = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            name_template="reversed_kernel_%s")
        self._layer_wo_act_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=positive_weights,
            name_template="reversed_kernel_positive_%s")
        self._layer_wo_act_negative = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=negative_weights,
            name_template="reversed_kernel_negative_%s")

    # TODO: clean up this implementation and add more documentation
    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))
        to_low = keras.layers.Lambda(lambda x: x * 0 + self._low)
        to_high = keras.layers.Lambda(lambda x: x * 0 + self._high)

        low = [to_low(x) for x in Xs]
        high = [to_high(x) for x in Xs]

        # Get values for the division.
        A = kutils.apply(self._layer_wo_act, Xs)
        B = kutils.apply(self._layer_wo_act_positive, low)
        C = kutils.apply(self._layer_wo_act_negative, high)
        Zs = [keras.layers.Subtract()([a, keras.layers.Add()([b, c])])
              for a, b, c in zip(A, B, C)]

        # Divide relevances with the value.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Distribute along the gradient.
        tmpA = iutils.to_list(grad(Xs+A+tmp))
        tmpB = iutils.to_list(grad(low+B+tmp))
        tmpC = iutils.to_list(grad(high+C+tmp))

        tmpA = [keras.layers.Multiply()([a, b]) for a, b in zip(Xs, tmpA)]
        tmpB = [keras.layers.Multiply()([a, b]) for a, b in zip(low, tmpB)]
        tmpC = [keras.layers.Multiply()([a, b]) for a, b in zip(high, tmpC)]

        tmp = [keras.layers.Subtract()([a, keras.layers.Add()([b, c])])
               for a, b, c in zip(tmpA, tmpB, tmpC)]

        return tmp



class ZPlusRule(Alpha1Beta0IgnoreBiasRule):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0, which assumes inputs x >= 0
    and ignores the bias.
    CAUTION! Results differ from Alpha=1, Beta=0
    if inputs are not strictly >= 0
    """
    #TODO: assert that layer inputs are always >= 0
    def __init__(self, *args, **kwargs):
        super(ZPlusRule, self).__init__(*args, **kwargs)



class ZPlusFastRule(kgraph.ReverseMappingBase):
    """
    The ZPlus rule is a special case of the AlphaBetaRule
    for alpha=1, beta=0 and assumes inputs x >= 0.
    """

    def __init__(self, layer, state, copy_weights=False):
        # The z-plus rule only works with positive weights and
        # no biases.
        #TODO: assert that layer inputs are always >= 0
        if copy_weights:
            weights = layer.get_weights()
            if layer.use_bias:
                weights = weights[:-1]
            weights = [x * (x > 0) for x in weights]
        else:
            weights = layer.weights
            if layer.use_bias:
                weights = weights[:-1]
            weights = [x * iK.to_floatx(x > 0) for x in weights]

        self._layer_wo_act_b_positive = kgraph.copy_layer_wo_activation(
            layer,
            keep_bias=False,
            weights=weights,
            name_template="reversed_kernel_positive_%s")

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))

        #TODO: assert all inputs are positive, instead of only keeping the positives.
        #keep_positives = keras.layers.Lambda(lambda x: x * K.cast(K.greater(x,0), K.floatx()))
        #Xs = kutils.apply(keep_positives, Xs)

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
