# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import zip


###############################################################################
###############################################################################
###############################################################################

import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as keras_layers
import numpy as np


from innvestigate import layers as ilayers
from innvestigate import utils as iutils
import innvestigate.utils.keras as kutils
from innvestigate.utils.keras import backend as iK
from innvestigate.utils.keras import graph as kgraph
from . import utils as rutils
from .. import new_base as base


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

    "AlphaBetaXRule",
    "AlphaBetaX1000Rule",
    "AlphaBetaX1010Rule",
    "AlphaBetaX1001Rule",
    "AlphaBetaX2m100Rule",

    "ZPlusRule",
    "ZPlusFastRule",
    "BoundedRule"
]


class ZRule(base.ReplacementLayer):
    def __init__(self, *args, **kwargs):
        layer = kwargs.pop("layer", None)
        bias = kwargs.pop("bias", True)
        self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                             keep_bias=bias,
                                                             name_template="no_act_%s")
        super(ZRule, self).__init__(layer, *args, **kwargs)

    def wrap_hook(self, ins, neuron_selection):
        with tf.GradientTape() as tape:
            tape.watch(ins)
            outs = self.layer_func(ins)

            # check if final layer (i.e., no next layers)
            if len(self.layer_next) == 0:
                outs = self._neuron_select(outs, neuron_selection)

        return outs, tape

    def explain_hook(self, ins, reversed_outs, args):
        outs, tape = args
        # Get activations.
        Zs = self._layer_wo_act(ins)
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([a, b]) for a, b in zip(reversed_outs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = tape.gradient(Zs, ins, output_gradients=tmp)
        # Re-weight relevance with the input values.
        ret = [keras_layers.Multiply()([a, b]) for a, b in zip(Xs, tmp)]
        return ret

#TODO: tf2.0
class ZIgnoreBiasRule(ZRule):
    """
    Basic LRP decomposition rule, ignoring the bias neuron
    """
    def __init__(self, *args, **kwargs):
        super(ZIgnoreBiasRule, self).__init__(*args,
                                              bias=False,
                                              **kwargs)


#TODO: tf2.0
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
        prepare_div = keras_layers.Lambda(lambda x: x + (K.cast(K.greater_equal(x,0), K.floatx())*2-1)*self._epsilon)

        # Get activations.
        Zs = kutils.apply(self._layer_wo_act, Xs)

        # Divide incoming relevance by the activations.
        tmp = [ilayers.Divide()([a, prepare_div(b)])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.to_list(grad(Xs+Zs+tmp))
        # Re-weight relevance with the input values.
        return [keras_layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]


#TODO: tf2.0
class EpsilonIgnoreBiasRule(EpsilonRule):
    """Same as EpsilonRule but ignores the bias."""
    def __init__(self, *args, **kwargs):
        super(EpsilonIgnoreBiasRule, self).__init__(*args,
                                                    bias=False,
                                                    **kwargs)


#TODO: tf2.0
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
        Zs = kutils.apply(self._layer_wo_act_b, ones)
        # Weight the incoming relevance.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Redistribute the relevances along the gradient.
        tmp = iutils.to_list(grad(Xs+Ys+tmp))
        return tmp



#TODO: tf2.0
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



#TODO: tf2.0
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
        times_alpha = keras_layers.Lambda(lambda x: x * self._alpha)
        times_beta = keras_layers.Lambda(lambda x: x * self._beta)
        keep_positives = keras_layers.Lambda(lambda x: x * K.cast(K.greater(x,0), K.floatx()))
        keep_negatives = keras_layers.Lambda(lambda x: x * K.cast(K.less(x,0), K.floatx()))


        def f(layer1, layer2, X1, X2):
            # Get activations of full positive or negative part.
            Z1 = kutils.apply(layer1, X1)
            Z2 = kutils.apply(layer2, X2)
            Zs = [keras_layers.Add()([a, b])
                    for a, b in zip(Z1, Z2)]
            # Divide incoming relevance by the activations.
            tmp = [ilayers.SafeDivide()([a, b])
                    for a, b in zip(Rs, Zs)]
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = iutils.to_list(grad(X1+Z1+tmp))
            tmp2 = iutils.to_list(grad(X2+Z2+tmp))
            # Re-weight relevance with the input values.
            tmp1 = [keras_layers.Multiply()([a, b])
                    for a, b in zip(X1, tmp1)]
            tmp2 = [keras_layers.Multiply()([a, b])
                    for a, b in zip(X2, tmp2)]
            #combine and return
            return [keras_layers.Add()([a, b])
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
            return [keras_layers.Subtract()([times_alpha(a), times_beta(b)])
                        for a, b in zip(activator_relevances, inhibitor_relevances)]
        else:
            return activator_relevances



#TODO: tf2.0
class AlphaBetaIgnoreBiasRule(AlphaBetaRule):
    """Same as AlphaBetaRule but ignores biases."""
    def __init__(self, *args, **kwargs):
        super(AlphaBetaIgnoreBiasRule, self).__init__(*args,
                                                      bias=False,
                                                      **kwargs)


#TODO: tf2.0
class Alpha2Beta1Rule(AlphaBetaRule):
    """AlphaBetaRule with alpha=2, beta=1"""
    def __init__(self, *args, **kwargs):
        super(Alpha2Beta1Rule, self).__init__(*args,
                                              alpha=2,
                                              beta=1,
                                              bias=True,
                                              **kwargs)

#TODO: tf2.0
class Alpha2Beta1IgnoreBiasRule(AlphaBetaRule):
    """AlphaBetaRule with alpha=2, beta=1 and ignores biases"""
    def __init__(self, *args, **kwargs):
        super(Alpha2Beta1IgnoreBiasRule, self).__init__(*args,
                                                        alpha=2,
                                                        beta=1,
                                                        bias=False,
                                                        **kwargs)

#TODO: tf2.0
class Alpha1Beta0Rule(AlphaBetaRule):
    """AlphaBetaRule with alpha=1, beta=0"""
    def __init__(self, *args, **kwargs):
        super(Alpha1Beta0Rule, self).__init__(*args,
                                              alpha=1,
                                              beta=0,
                                              bias=True,
                                              **kwargs)

#TODO: tf2.0
class Alpha1Beta0IgnoreBiasRule(AlphaBetaRule):
    """AlphaBetaRule with alpha=1, beta=0 and ignores biases"""
    def __init__(self, *args, **kwargs):
        super(Alpha1Beta0IgnoreBiasRule, self).__init__(*args,
                                                        alpha=1,
                                                        beta=0,
                                                        bias=False,
                                                        **kwargs)

#TODO: tf2.0
class AlphaBetaXRule(kgraph.ReverseMappingBase):
    """
    AlphaBeta advanced as proposed by Alexander Binder.
    """

    def __init__(self,
                 layer,
                 state,
                 alpha=(0.5, 0.5),
                 beta=(0.5, 0.5),
                 bias=True,
                 copy_weights=False):
        self._alpha = alpha
        self._beta = beta

        # prepare positive and negative weights for computing positive
        # and negative preactivations z in apply_accordingly.
        if copy_weights:
            weights = layer.get_weights()
            if not bias and getattr(layer, "use_bias", False):
                weights = weights[:-1]
            positive_weights = [x * (x > 0) for x in weights]
            negative_weights = [x * (x < 0) for x in weights]
        else:
            weights = layer.weights
            if not bias and getattr(layer, "use_bias", False):
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
        times_alpha0 = keras_layers.Lambda(lambda x: x * self._alpha[0])
        times_alpha1 = keras_layers.Lambda(lambda x: x * self._alpha[1])
        times_beta0 = keras_layers.Lambda(lambda x: x * self._beta[0])
        times_beta1 = keras_layers.Lambda(lambda x: x * self._beta[1])
        keep_positives = keras_layers.Lambda(
            lambda x: x * K.cast(K.greater(x,0), K.floatx()))
        keep_negatives = keras_layers.Lambda(
            lambda x: x * K.cast(K.less(x,0), K.floatx()))

        def f(layer, X):
            Zs = kutils.apply(layer, X)
            # Divide incoming relevance by the activations.
            tmp = [ilayers.SafeDivide()([a, b])
                    for a, b in zip(Rs, Zs)]
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp = iutils.to_list(grad(X+Zs+tmp))
            # Re-weight relevance with the input values.
            tmp = [keras_layers.Multiply()([a, b])
                    for a, b in zip(X, tmp)]
            return tmp

        # Distinguish postive and negative inputs.
        Xs_pos = kutils.apply(keep_positives, Xs)
        Xs_neg = kutils.apply(keep_negatives, Xs)

        # xpos*wpos
        r_pp = f(self._layer_wo_act_positive, Xs_pos)
        # xneg*wneg
        r_nn = f(self._layer_wo_act_negative, Xs_neg)
        # a0 * r_pp + a1 * r_nn
        r_pos = [keras_layers.Add()([times_alpha0(pp), times_beta1(nn)])
                 for pp, nn in zip(r_pp, r_nn)]

        # xpos*wneg
        r_pn = f(self._layer_wo_act_negative, Xs_pos)
        # xneg*wpos
        r_np = f(self._layer_wo_act_positive, Xs_neg)
        # b0 * r_pn + b1 * r_np
        r_neg = [keras_layers.Add()([times_beta0(pn), times_beta1(np)])
                 for pn, np in zip(r_pn, r_np)]

        return [keras_layers.Subtract()([a, b]) for a, b in zip(r_pos, r_neg)]

#TODO: tf2.0
class AlphaBetaX1000Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX1000Rule, self).__init__(*args,
                                                 alpha=(1, 0),
                                                 beta=(0, 0),
                                                 bias=True,
                                                 **kwargs)

#TODO: tf2.0
class AlphaBetaX1010Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX1010Rule, self).__init__(*args,
                                                 alpha=(1, 0),
                                                 beta=(0, -1),
                                                 bias=True,
                                                 **kwargs)

#TODO: tf2.0
class AlphaBetaX1001Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX1001Rule, self).__init__(*args,
                                                 alpha=(1, 1),
                                                 beta=(0, 0),
                                                 bias=True,
                                                 **kwargs)

#TODO: tf2.0
class AlphaBetaX2m100Rule(AlphaBetaXRule):
    def __init__(self, *args, **kwargs):
        super(AlphaBetaX2m100Rule, self).__init__(*args,
                                                  alpha=(2, 0),
                                                  beta=(1, 0),
                                                  bias=True,
                                                  **kwargs)

#TODO: tf2.0
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
        to_low = keras_layers.Lambda(lambda x: x * 0 + self._low)
        to_high = keras_layers.Lambda(lambda x: x * 0 + self._high)

        low = [to_low(x) for x in Xs]
        high = [to_high(x) for x in Xs]

        # Get values for the division.
        A = kutils.apply(self._layer_wo_act, Xs)
        B = kutils.apply(self._layer_wo_act_positive, low)
        C = kutils.apply(self._layer_wo_act_negative, high)
        Zs = [keras_layers.Subtract()([a, keras_layers.Add()([b, c])])
              for a, b, c in zip(A, B, C)]

        # Divide relevances with the value.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Distribute along the gradient.
        tmpA = iutils.to_list(grad(Xs+A+tmp))
        tmpB = iutils.to_list(grad(low+B+tmp))
        tmpC = iutils.to_list(grad(high+C+tmp))

        tmpA = [keras_layers.Multiply()([a, b]) for a, b in zip(Xs, tmpA)]
        tmpB = [keras_layers.Multiply()([a, b]) for a, b in zip(low, tmpB)]
        tmpC = [keras_layers.Multiply()([a, b]) for a, b in zip(high, tmpC)]

        tmp = [keras_layers.Subtract()([a, keras_layers.Add()([b, c])])
               for a, b, c in zip(tmpA, tmpB, tmpC)]

        return tmp


#TODO: tf2.0
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


#TODO: tf2.0
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
        #keep_positives = keras_layers.Lambda(lambda x: x * K.cast(K.greater(x,0), K.floatx()))
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
        return [keras_layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]

#TODO: gamma-Rule tf2.0